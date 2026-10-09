# 量化 KV cache：方案一览

KV cache 是长上下文推理的显存大头（128k 上下文的 70B 模型，fp16 KV
约 40 GiB）。量化它 = 不换硬件扩上下文和并发。
每个方案单独一页，从下面的表点进去；实现细节在
[npu-implementation.md](npu-implementation.md)，怎么跑在
[how-to-run.md](how-to-run.md)。

## 现状一览（feat/int4，2026-10-03）

| 方案 | 一句话 | 压缩 | 代码 | 910B2 验证 | 命令行可选 |
|---|---|---|---|---|---|
| [`int8_dynamic`](schemes/int8-dynamic.md) | 每通道动态 scale 的 int8 | 2x | ✅ | ✅ 端到端 serve 已跑通（[实测](serving-verification-20261003.md)） | ✅ `int8` |
| [`kivi_int4`](schemes/kivi-int4.md) | 近期 token 保 fp16，更早压 int4 | ~3.6x | ✅ | ✅ 13 节设备记录；端到端 serve 已跑通（[实测](serving-verification-20261003.md)） | ✅ `kivi_int4` |
| [`fp8_per_token_head`](schemes/fp8-per-token-head.md) | 每 (token, head) 一个 scale 的 E4M3 | 1.94x | 写入内核有，读路径没有 | ❌ | ❌ |
| [四个纯格式](schemes/packed-formats.md) | 只声明格式，不算 | 2–3.6x | 只在 dev 分支 | ❌ | ❌ |

复现：`PYTHONPATH=src python -m pytest -q` → 111 passed。
没有 NPU 的环境跑设备代码会直接报错，不会猜。

## 接下来做哪个（按性价比）

1. **fp8_per_token_head 补完**：上机比对 → 写读路径 → 接命令行。
   宿主 triton 后端原生声明这个 dtype，不用等厂商算子。
2. **int8_per_token_head**：同一个内核的 int8 变体，几乎免费。
3. **TurboQuant**：去 vllm-ascend #15198/#15821 认领对齐。
   注意只省显存——GPU 实测吞吐只有 BF16 的 66–80%。
4. **kivi_int4 + 旋转**：Hadamard 融进内核才能做，融不进就不做。
5. **GEAR 式误差修正**：写历史区时误差本来就有，做稀疏修正最便宜。

## 学术方案要什么 vs 我们手里有什么

2bit 三件套的积木基本齐全，缺的是接线（论证见
[kvquant-survey.md](kvquant-survey.md) §5.4、§5.5）：

| 学术方案要什么 | 我们手里有什么 | 差距 |
|---|---|---|
| 全精度兜底（sink / 近期窗口） | KIVI 残差窗口，已验证 | 无 |
| 旋转（Hadamard） | 宿主 `single_rht`（int4_per_token_head 在用） | 没接进 KIVI 的 flush 与内核 |
| 非均匀编码 | 宿主 TurboQuant 有完整 Lloyd-Max + 纯 triton 内核 | 没搬到 NPU；上游 #15198/#15821 等 NPU 实现 |
| FP8 存储 | 已写好的 fp8_per_token_head 写入内核 | 读路径 |
| GEAR 的"误差已知" | KIVI flush 时 min/max 本来就算了 | 稀疏修正逻辑没写 |

**明确不做**：nvfp4 / FP8 attention / MXFP8 / HiF8（这代 910B2 没有该硬件
能力，等 950 代）；格码与熵编码（解码串行、依赖数据内容，不可服务化）；需要
权重 / 校准集 / 梯度搜索的离线方案（与"不读 checkpoint、免校准"的定位冲突）；
基于注意力分数的显著性方案（我们走 fused attention 算子，拿不到分数）。

**基线待补**：端到端已通，还差 Qwen3.5 上的真实数字与模型级精度评测；跑长
上下文评测前先确认绕开 FIA 掩码 bug #15503。外部数字未经复测不对外引用。

## 怎么选

- 省一半、最快跑通 → **int8**
- 长上下文、要 4x → **kivi_int4**
- 精度经验排序：int8 ≈ 动态 fp8 > KIVI int4 > 纯 int4 > fp4 系
  （上线前务必用目标模型实测）
