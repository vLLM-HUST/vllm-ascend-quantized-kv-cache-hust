# Benchmark 数据（项目唯一保留）

> 约定：**整个项目只保留这一份 benchmark 数据**。以后每次新的
> benchmark 跑完，覆盖更新本文件和 `benchmarks/` 里的原始 JSON，
> 不另开新文档。

## 数据集

- **模型**：Qwen3.5-35B-A3B（打榜模型；此前在稠密小模型上的首跑
  只用于验证页大小数学，数据未保留）
- **日期/机器**：2026-10-04，vllm-hust-cyj（2×910B2，CANN 9.1.0）
- **配置**：TP2 + EP，`--enforce-eager --no-enable-chunked-prefill
  --no-enable-prefix-caching --max-model-len 8192 --max-num-seqs 16
  --gpu-memory-utilization 0.78`，bfloat16 权重，三组只差
  `--kv-cache-dtype`
- **负载**：8 条固定提示词并发 × max_tokens 128，温度 0；另测单路
  一条。Qwen3.5 是 thinking 模型，输出以推理过程为主。

## 结果

| | fp16 (auto) | int8 | kivi_int4 |
|---|---|---|---|
| 聚合吞吐 | 2.2 tok/s | 2.3 tok/s | 2.0 tok/s |
| 单路 128 token 延迟 | 87.1s | 85.9s | 99.8s |
| 8192 上下文可并发 | 90.14x | 126.20x | 126.20x |

三组输出全部连贯（量化集成正确）。

## 怎么读这些数

1. **吞吐 ~2 tok/s 与 KV 方案无关**：瓶颈在 Qwen3.5 的 GDN 层——
   这台宿主缺新 CANN 算子，GDN 全部走纯 torch 兜底（每层每步几十
   次 Python 循环）。三组同瓶颈，跨组对比仍公平；但要出打榜级
   数字，必须先把 GDN 换回真内核（升级宿主树或写融合内核）。
2. **KV 压缩的收益体现为并发 +40%**（90→126）：该模型只有部分层是
   全注意力，GDN 层状态定长不随 token 涨，稀释是预期。
3. **待查**：int8 与 kivi_int4 的并发数完全相同（126.20x）。稠密
   模型首跑里 int8 是精确 2.0x、kivi 是 3.56x，说明算子本身没问题，
   怀疑该宿主对 int8 模式的页大小处理与 kivi 混了，值得单独核。

## 原始数据与复现

- 原始 JSON：`benchmarks/qwen3.5-35b-a3b/bench-q35-{off,int8,kivi_int4}.json`
- 复现：服务器 `/root/bench/scripts/qkv35_run.sh`（等两卡空闲后
  三组连跑），客户端 `bench_qkv.py`；宿主侧需要
  [serving-verification-20261003.md](serving-verification-20261003.md)
  §3/§4 的那批修复。
