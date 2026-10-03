# 量化 KV cache：方案一览

KV cache 是长上下文推理的显存大头（128k 上下文的 70B 模型，fp16 KV
约 40 GiB）。量化它 = 不换硬件扩上下文和并发。
每个方案单独一页，从下面的表点进去；实现细节在
[npu-implementation.md](npu-implementation.md)，怎么跑在
[how-to-run.md](how-to-run.md)。

## 现状一览（feat/int4，2026-10-03）

| 方案 | 一句话 | 压缩 | 代码 | 910B2 验证 | 命令行可选 |
|---|---|---|---|---|---|
| [`int8_dynamic`](schemes/int8-dynamic.md) | 每通道动态 scale 的 int8 | 2x | ✅ | 选实现对过，引擎内没端到端跑过 | ✅ `int8` |
| [`kivi_int4`](schemes/kivi-int4.md) | 近期 token 保 fp16，更早压 int4 | ~3.6x | ✅ | ✅ 13 节设备记录；端到端 serve 已跑通（[实测](serving-verification-20261003.md)） | ✅ `kivi_int4` |
| [`fp8_per_token_head`](schemes/fp8-per-token-head.md) | 每 (token, head) 一个 scale 的 E4M3 | 1.94x | 写入内核有，读路径没有 | ❌ | ❌ |
| [四个纯格式](schemes/packed-formats.md) | 只声明格式，不算 | 2–3.6x | 只在 dev 分支 | ❌ | ❌ |

复现：`PYTHONPATH=src python -m pytest -q` → 108 passed。
没有 NPU 的环境跑设备代码会直接报错，不会猜。

## 接下来做哪个（按性价比）

1. **fp8_per_token_head 补完**：上机比对 → 写读路径 → 接命令行。
   宿主 triton 后端原生声明这个 dtype，不用等厂商算子。
2. **int8_per_token_head**：同一个内核的 int8 变体，几乎免费。
3. **TurboQuant**：去 vllm-ascend #15198/#15821 认领对齐。
   注意只省显存——GPU 实测吞吐只有 BF16 的 66–80%。
4. **kivi_int4 + 旋转**：Hadamard 融进内核才能做，融不进就不做。
5. **GEAR 式误差修正**：写历史区时误差本来就有，做稀疏修正最便宜。

nvfp4 / FP8 attention 等 950 代硬件；格码、熵编码不做。
详细论证见 [kvquant-schemes-beyond-int8-int4.md](kvquant-schemes-beyond-int8-int4.md)。

## 怎么选

- 省一半、最快跑通 → **int8**
- 长上下文、要 4x → **kivi_int4**
- 精度经验排序：int8 ≈ 动态 fp8 > KIVI int4 > 纯 int4 > fp4 系
  （上线前务必用目标模型实测）
