from modelscope import snapshot_download

# 下载 Target (7B)
# target_dir = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir='./models')

# 下载 Draft (1.5B)
draft_dir = snapshot_download('qwen/Qwen2.5-1.5B-Instruct', cache_dir='./models')

# print(f"Target model saved at: {target_dir}")
print(f"Draft model saved at: {draft_dir}")
