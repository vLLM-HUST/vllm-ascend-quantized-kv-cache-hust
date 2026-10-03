# 量化 KV cache：方案一览

KV cache 是长上下文推理的显存大头（128k 上下文的 70B 模型，fp16 KV
约 40 GiB）。量化它 = 不换硬件扩上下文和并发。
实现细节见 [npu-implementation.md](npu-implementation.md)，怎么跑见
[how-to-run.md](how-to-run.md)，方案出处见
[../PROVENANCE.md](../PROVENANCE.md)。

## 现状一览（feat/int4，2026-10-03）

| 方案 | 一句话 | 压缩 | 代码 | 910B2 验证 | 命令行可选 |
|---|---|---|---|---|---|
| `int8_dynamic` | 每通道动态 scale 的 int8 | 2x | ✅ | 选实现对过，引擎内没端到端跑过 | ✅ `int8` |
| `kivi_int4` | 近期 token 保原精度，更早压 int4 | ~4x | ✅ | ✅ 13 节设备记录 | ✅ `kivi_int4`（缺端到端 serve） |
| `fp8_per_token_head` | 每 token 每头一个 scale 的 E4M3 | 1.94x | 写入内核有，读路径没有 | ❌ | ❌ |
| `int4_packed` / `fp8_e4m3` / `nvfp4` / `fp4_e2m1` | 纯格式声明 | 2–3.6x | 只在 dev 分支 | ❌ | ❌ |

复现：`PYTHONPATH=src python -m pytest -q` → 108 passed。
没有 NPU 的环境跑设备代码会直接报错，不会猜。

## 两个能用的方案

**int8**：第一次 prefill 按通道算一次 scale，之后不变。不用打包、
fused attention 原生支持，最省事。要求 head_size 是 8 的倍数。

**int4（KIVI）**：键对位置敏感、值对通道敏感、新 token 重要——
所以每个请求分两块：最近 `residual_length`（默认 128）个 token 保持
fp16；更早的按组压成 int4（键按 token 分组，值按 head 维分组）。
写入走 triton 内核，读取时 gather 解压成稠密再算注意力。
几何要求：group / head / block 互相整除（默认 128 全满足；
Qwen3.5-35B-A3B 的 head 256 也满足）。跑法必须
`--enforce-eager`、不开 chunked prefill。

每个 dtype 实际存成什么，只有 `dtypes.py` 的 `resolve_layout` 说了算
（head_size=128 时）：

| dtype | 存储 | 相对 fp16 |
|---|---|---|
| int8 | int8，不打包 | 2x |
| kivi_int4 | uint8，4bit 数据 + 每组 scale/min | ~3.6x（含残差区） |
| fp8_per_token_head | uint8，每 (token,head) 1B 数据 + 4B scale | 1.94x |

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
