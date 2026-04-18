import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import time

# 配置路径 - 请修改为你本地的实际路径
TARGET_MODEL_PATH = "./models/qwen/Qwen2.5-7B-Instruct"
DRAFT_MODEL_PATH = "./models/qwen/Qwen2.5-1.5B-Instruct"

class SimpleSpeculativeEngine:
    def __init__(self, target_path, draft_path, device="cuda"):
        self.device = device
        print(f"正在加载 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(target_path)
        
        print(f"正在加载 Target Model (7B)...")
        # 4090 使用 bfloat16 性能最强
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_path, dtype=torch.bfloat16, device_map=device
        )
        
        print(f"正在加载 Draft Model (0.5B)...")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_path, dtype=torch.bfloat16, device_map=device
        )

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, K=4):
        """
        K: 投机步数 (lookahead window)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        # 初始化 KV Cache
        target_past_key_values = None
        
        # 统计数据
        total_accepted_tokens = 0
        iteration_count = 0
        start_time = time.time()

        # 1. 预填充 (Prefill) Target Model，获取初始 KV Cache
        outputs = self.target_model(input_ids, use_cache=True)
        target_past_key_values = outputs.past_key_values
        
        # 取第一个生成的 token
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        while input_ids.shape[1] < inputs.input_ids.shape[1] + max_new_tokens:
            iteration_count += 1
            current_num_tokens = input_ids.shape[1]
            
            # --- 步骤 A: Draft Model 连续生成 K 个 Token ---
            # 为了简单，Draft 每次从头跑，后续优化可以用 Draft 的 KV Cache
            draft_candidate_ids = input_ids.clone()
            for _ in range(K):
                draft_outputs = self.draft_model(draft_candidate_ids)
                next_draft_token = torch.argmax(draft_outputs.logits[:, -1, :], dim=-1, keepdim=True)
                draft_candidate_ids = torch.cat([draft_candidate_ids, next_draft_token], dim=-1)
            
            # 提取投机的 K 个 token
            speculated_tokens = draft_candidate_ids[:, current_num_tokens:] 

            # --- 步骤 B: Target Model 一次性并行验证 ---
            # 我们把投机的 tokens 接在后面，一次性通过 Target Model
            target_outputs = self.target_model(
                draft_candidate_ids, # 迄今为止所有的 token（包括投机的）
                past_key_values=target_past_key_values, 
                use_cache=True
            ) # 这里怎么没有position_ids？因为我们直接输入了完整的 token 序列，模型会自动处理位置编码，不需要我们手动指定 position_ids。
            # 但是会产生重复计算吗？
            
            # 验证逻辑：
            # target_outputs.logits 包含了从 current_num_tokens 到最后的预测
            # 我们要对比的是：Target 对前 K 个位置的预测，是否等于 Draft 投机的 token
            full_target_logits = target_outputs.logits
            # 我们只关心新增部分的预测
            # target_predicted_ids 的长度也是 K+1 (K个验证 + 1个多出来的bonus)
            target_predicted_ids = torch.argmax(full_target_logits[:, -(K+1):, :], dim=-1)
            
            # --- 步骤 C: 比较与接受 (The "Rollback" Logic) ---
            n_accepted = 0
            for i in range(K):
                if speculated_tokens[0, i] == target_predicted_ids[0, i]:
                    n_accepted += 1
                else:
                    break
            
            # --- 步骤 D: 更新序列与 KV Cache ---
            # 接受成功的 tokens + Target Model 预测的那个“不一致位置”的正确 token (Bonus Token)
            accepted_ids = target_predicted_ids[:, :n_accepted + 1]
            input_ids = torch.cat([input_ids[:, :current_num_tokens], accepted_ids], dim=-1)
            
            # 关键：KV Cache 回滚
            # 在 HuggingFace 中，KV Cache 的 shape 是 [layer, 2, batch, num_heads, seq_len, head_dim]
            # 我们需要把 seq_len 裁剪到当前实际接受的长度
            new_len = current_num_tokens + n_accepted
            target_past_key_values = self._rollback_kv_cache(target_outputs.past_key_values, new_len)
            
            total_accepted_tokens += n_accepted
            print(f"Iteration {iteration_count}: Accepted {n_accepted}/{K} tokens")

        end_time = time.time()
        total_gen_len = input_ids.shape[1] - inputs.input_ids.shape[1]
        print(f"\n[结果] 总生成长度: {total_gen_len}")
        print(f"[平均接受率]: {total_accepted_tokens / (iteration_count * K):.2f}")
        print(f"[端到端速度]: {total_gen_len / (end_time - start_time):.2f} tokens/s")
        
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def _rollback_kv_cache(self, cache: DynamicCache, keep_len: int):
        """
        将 DynamicCache 裁剪到指定的长度 keep_len
        """

        print(f"正在回滚 KV Cache 到长度 {keep_len}...")

        if cache is None:
            return None
        
        # 1. 对每一层的 Tensor 进行切片 (dim=2 是 seq_len 维度)
        for i in range(len(cache.layers)):
            cache.layers[i].keys = cache.layers[i].keys[:, :, :keep_len, :]
            cache.layers[i].values = cache.layers[i].values[:, :, :keep_len, :]
        
        return cache

# --- 运行测试 ---
if __name__ == "__main__":
    engine = SimpleSpeculativeEngine(TARGET_MODEL_PATH, DRAFT_MODEL_PATH)
    
    prompt = "Artificial intelligence is "
    result = engine.generate(prompt, max_new_tokens=60, K=4)
    print("\n生成的文本内容: ")
    print(result)