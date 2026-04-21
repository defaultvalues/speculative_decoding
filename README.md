# A Simple Inference Eigen base on Speculative Decoding

基于投机采样实现的一个大模型推理引擎，Target模型和Draft模型分别为Qwen 7B和1.5B。

## 主要功能

- K步前瞻的投机采样
- Target Model和Draft Model均采用KV Cache加速推理
- 发生错误时支持KV Cache回滚

##  KV Cache回滚逻辑示意图

1. 让draft model连续预测K的token，并将前K-1个token与初始输入送进target model进行验证

![](./images/SD.svg)

2. 将target model的输出结果与draft model的猜测结果比对，仅保留正确部分的KV cache

![](./images/kv_roll.svg)


## 运行

```
# 同步环境
uv sync

# 运行主代码（需要切换为本地的模型路径，手动调整prompt）
uv run python speculative_eigen.py
```


## Todo List

- [ ] Rejction Sampling


