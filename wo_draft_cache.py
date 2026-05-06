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

        eos_token_id = self.target_model.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            self.eos_token_ids = [eos_token_id]
        elif isinstance(eos_token_id, list):
            self.eos_token_ids = eos_token_id
        else:
            self.eos_token_ids = [self.tokenizer.eos_token_id]

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, K=4):
        """
        K: 投机步数 (lookahead window)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        # 初始化 KV Cache
        target_past_key_values = None
        draft_past_key_values = None
        
        # 统计数据
        total_accepted_tokens = 0
        iteration_count = 0
        torch.cuda.synchronize()
        start_time = time.time()

        # 1. 预填充 (Prefill) Target Model，获取初始 KV Cache
        outputs = self.target_model(input_ids, use_cache=True)
        target_past_key_values = outputs.past_key_values

        # draft_outputs = self.draft_model(input_ids, use_cache=True)
        # draft_past_key_values = draft_outputs.past_key_values
        
        # 取第一个生成的 token
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        avg_acceptance_rates = 1.0  # 初始化平均接受率

        stop_generation = False

        while input_ids.shape[1] < inputs.input_ids.shape[1] + max_new_tokens:
            iteration_count += 1
            current_num_tokens = input_ids.shape[1]

            # draft_outputs = self.draft_model(input_ids[:, -1:], 
            #                                  past_key_values=draft_past_key_values, 
            #                                  use_cache=True)
            # draft_past_key_values = draft_outputs.past_key_values

            
            # 直接调用 HF 的 generate 函数
            draft_outputs = self.draft_model.generate(
                input_ids,
                max_new_tokens=K,
                use_cache=False,  # 设为 False 满足你不使用 KV Cache 的测试需求
                do_sample=False,  # 贪心解码
                # 屏蔽一些不必要的输出以稍微提升速度
                temperature=1.0, 
                top_p=1.0, 
                top_k=0,
                
                # 2. 关闭任何惩罚和额外处理
                repetition_penalty=1.0,
                length_penalty=1.0,
                
                # 3. 忽略模型自带的 generation_config.json 中的其他设置
                renormalize_logits=False,
            )
            
            # generate 函数默认返回的是[完整原始 input_ids + 新生成的 token]
            # 我们只需要提取新生成的那部分
            speculated_tokens = draft_outputs[:, input_ids.shape[1]:]
            
            # 注意：generate 遇到 EOS 会提前停止，因此生成的长度可能小于 K
            actual_k = speculated_tokens.shape[1]
            
            if actual_k == 0:
                # 极端情况：Draft 直接吐出了 EOS，无 token 可验证
                break

            # --- 步骤 B: Target Model 一次性并行验证 ---
            # 准备验证输入: 上一轮最后确定的 1 个 token + 投机生成的 actual_k 个 token
            target_input_ids = torch.cat([input_ids[:, -1:], speculated_tokens], dim=-1)
            
            target_outputs = self.target_model(
                target_input_ids, 
                past_key_values=target_past_key_values, 
                use_cache=True
            ) 
            
            target_predicted_ids = torch.argmax(target_outputs.logits, dim=-1)
            
            # --- 步骤 C: 比较与接受 (The "Rollback" Logic) ---
            n_accepted = 0
            for i in range(actual_k):  # 这里的上限变成了 actual_k
                if speculated_tokens[0, i] == target_predicted_ids[0, i]:
                    n_accepted += 1
                    if speculated_tokens[0, i] in self.eos_token_ids:
                        stop_generation = True
                        break
                else:
                    break
            
            # --- 步骤 D: 更新序列与 KV Cache ---
            # 接受成功的 tokens + Target Model 预测的那个“不一致位置”的正确 token (Bonus Token)
            # --- 步骤 D: 更新序列与 KV Cache ---
            if stop_generation:
                # 修复1: 如果遇到 EOS 结束了，只接受到 EOS 为止，不要后面的 Bonus Token
                accepted_ids = target_predicted_ids[:, :n_accepted]
            else:
                # 正常情况：接受 n_accepted 个 drafted token + 1个 bonus token
                accepted_ids = target_predicted_ids[:, :n_accepted+1]
            
            input_ids = torch.cat([input_ids, accepted_ids], dim=-1)
            
            # 关键：KV Cache 回滚
            # 我们需要把 seq_len 裁剪到当前实际接受的长度
            new_len = current_num_tokens + n_accepted
            target_past_key_values = self._rollback_kv_cache(target_outputs.past_key_values, new_len)
            # draft_past_key_values = self._rollback_kv_cache(draft_past_key_values, new_len)  # +1 因为 draft 还多了一个 token
            
            total_accepted_tokens += n_accepted
            if stop_generation:
                break
            # print(f"Iteration {iteration_count}: Accepted {n_accepted}/{K} tokens")


        torch.cuda.synchronize()
        end_time = time.time()
        total_gen_len = input_ids.shape[1] - inputs.input_ids.shape[1]
        print(f"\n[结果] 总生成长度: {total_gen_len}")
        print(f"[平均接受率]: {total_accepted_tokens / (iteration_count * K):.2f}")
        print(f"[端到端速度]: {total_gen_len / (end_time - start_time):.2f} tokens/s")
        
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True), end_time - start_time, total_gen_len, total_accepted_tokens / (iteration_count * K)

    def _rollback_kv_cache(self, cache: DynamicCache, keep_len: int):
        """
        将 DynamicCache 裁剪到指定的长度 keep_len
        """

        if cache is not None:
            cache.crop(keep_len) 
        
        return cache

# --- 运行测试 ---
if __name__ == "__main__":
    engine = SimpleSpeculativeEngine(TARGET_MODEL_PATH, DRAFT_MODEL_PATH)
    
    prompt = "Explain KV Cache and speculative decoding. "
    result, elapsed_time, total_gen_len, avg_acceptance_rate = engine.generate(prompt, max_new_tokens=2048, K=4)
    print("\n生成的文本内容: ")
    print(result)