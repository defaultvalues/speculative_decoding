# A Simple Inference Eigen base on Speculative Decoding

基于投机采样实现的一个大模型推理引擎，Target模型和Draft模型分别为Qwen 7B和1.5B。

## 主要功能

- K步前瞻的投机采样
- Target Model和Draft Model均采用KV Cache加速推理
- 发生错误时支持KV Cache回滚

##  KV Cache回滚逻辑示意图


## 运行

```
# 同步环境
uv sync

# 运行主代码（需要切换为本地的模型路径，手动调整prompt）
uv run python speculative_eigen.py
```


## TODO List

- [ ] Rejction Sampling
- [ ] 

