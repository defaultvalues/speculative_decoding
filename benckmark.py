from speculative_eigen import SimpleSpeculativeEngine
from baseline import SimpleEngine

TARGET_MODEL_PATH = "./models/qwen/Qwen2.5-7B-Instruct"
DRAFT_MODEL_PATH = "./models/qwen/Qwen2.5-1.5B-Instruct"
# --- 运行测试 ---
if __name__ == "__main__":
    # engine = SimpleEngine(TARGET_MODEL_PATH, DRAFT_MODEL_PATH)
    engine = SimpleSpeculativeEngine(TARGET_MODEL_PATH, DRAFT_MODEL_PATH)

    engine.generate("Warm up", max_new_tokens=512, K=8)  # 预热，加载模型和编译等开销只计算在第一次生成中，后续生成的性能数据更稳定可靠

    prompts = ["Repeat \"abc123\" for 50 times.",  # 简单的重复模式
               "Write a Python function to compute Fibonacci numbers.",  # 需要一定推理能力的编程题
               "Write a C++ program that implements a red-black tree with insert and delete operations.",  # 更复杂的编程题
               "请用中文写一首关于春天的诗，要求押韵，并且包含以下元素：花朵、鸟儿、微风、阳光。"
                ]   
    


    for prompt in prompts:
        result, elapsed_time, total_gen_len, avg_acceptance_rate = engine.generate(prompt, max_new_tokens=512, K=16)
        print("\n生成的文本内容: ")
        print(result)
        print(f"\n生成耗时: {elapsed_time:.2f} 秒")
        print(f"总生成长度: {total_gen_len}")
        print(f"生成速度: {total_gen_len / elapsed_time:.2f} tokens/s")
        print(f"平均接受率: {avg_acceptance_rate:.2f}")