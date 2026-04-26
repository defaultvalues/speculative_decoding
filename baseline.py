import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import time

# 配置路径 - 请修改为你本地的实际路径
TARGET_MODEL_PATH = "./models/qwen/Qwen2.5-7B-Instruct"
DRAFT_MODEL_PATH = "./models/qwen/Qwen2.5-1.5B-Instruct"

class SimpleEngine:
    """
    基础的推理引擎，直接使用 Target Model 进行生成，不进行投机。
    主要用于性能基线对比。
    """
    def __init__(self, target_path, draft_path, device="cuda"):
        self.device = device
        print(f"正在加载 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(target_path)
        
        print(f"正在加载 Target Model (7B)...")
        # 4090 使用 bfloat16 性能最强
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_path, dtype=torch.bfloat16, device_map=device
        )
        
        # print(f"正在加载 Draft Model (1.5B)...")
        # self.draft_model = AutoModelForCausalLM.from_pretrained(
        #     draft_path, dtype=torch.bfloat16, device_map=device
        # )

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, K=4):
        """
        K: 投机步数 (lookahead window，基线不使用，保持接口一致)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids

        torch.cuda.synchronize()
        start_time = time.time()

        output_ids = self.target_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 使用贪心解码，确保生成结果的确定性，便于性能对比
            use_cache=True
        )

        torch.cuda.synchronize()
        end_time = time.time()
        total_gen_len = output_ids.shape[1] - input_ids.shape[1]
        print(f"生成完成，耗时 {end_time - start_time:.2f} 秒，生成了 {total_gen_len} 个新 token。")
        print(f"生成速度: {total_gen_len / (end_time - start_time):.2f} tokens/s")

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True), end_time - start_time, total_gen_len, 1.0  # 基线引擎接受率固定为 100%

# --- 运行测试 ---
if __name__ == "__main__":
    engine = SimpleEngine(TARGET_MODEL_PATH, DRAFT_MODEL_PATH)
    
    prompt = "Compute the eigenvalues of the following matrix: [[2, 1], [1, 2]]."
    result, elapsed_time, total_gen_len, avg_acceptance_rate = engine.generate(prompt, max_new_tokens=128, K=8)
    print("\n生成的文本内容: ")
    print(result)