# 量化 KV cache：模块含义与六个方案的差异

本文回答三个问题：这个模块**是干什么的**（§1）、KV cache 量化的
**基本原理**（§2）、六个方案的**语义差异与选型**（§3–§5）。
运行方法见 [how-to-run.md](how-to-run.md)，NPU 实现细节见
[npu-implementation.md](npu-implementation.md)。

## 1. 这个模块是干什么的

### 1.1 问题：KV cache 是长上下文推理的显存大头

自回归推理要把每个 token 的 K/V 存进分页缓存供后续注意力读取。每 token
每层的缓存字节数：

```
2 (K+V) × num_kv_heads × head_size × 每元素字节数
```

以 Llama-3-70B 形状的模型为例（80 层、8 个 KV 头、head_size 128、fp16）：
`2 × 8 × 128 × 2 B = 4 KiB/token/层`，全模型约 **320 KiB/token**。上下文
128k 时单个请求仅 KV cache 就要 ~40 GiB——批大小和上下文长度直接被
缓存容量卡死。

KV cache 量化把 K/V 从 fp16/bf16 压成 int8（2x）、int4/fp4（~4x–4.5x），
是**不换硬件扩大上下文与并发**的最直接手段。

### 1.2 这个库的定位

- **不是**离线模型权重量化，也不是 Adaptive Quantized KV observer 项目
  （见 README 的技术所有权声明）。
- **是**一个方案库 + 宿主适配层：把 legacy 分支里验证过的量化实现
  （出处见 [../PROVENANCE.md](../PROVENANCE.md)）整理成六个自注册
  "方案"，统一暴露三层能力：
  1. **契约**（`resolve_layout`）：dtype 字符串 → 存储布局
     （存储 dtype / 打包维度），全库唯一裁决处 `dtypes.py`；
  2. **语义**（`sol.semantics`）：纯 torch 数学（scale 计算、打包/解包、
     窗口簿记），CPU 可单测，同时是 NPU 内核的数值参考；
  3. **宿主适配**（`sol.host_adapter(host)`）：把方案插进
     vllm-ascend-hust 或 vllm-hust 的注册表。
- **设备边界**：内核层（triton-ascend / torch_npu）只在 Ascend NPU 上
  执行；任何设备路径在非 NPU 环境 fail-closed 并给出明确报错。

## 2. 原理速览：KV cache 量化在量化什么

注意力计算需要的是浮点 K/V。量化方案要回答四个问题：

1. **粒度**：一组数值共享一个 scale（per-tensor / per-channel /
   per-token-head / per-16-元素块）？粒度越细精度越好、开销越大。
2. **scale 来源**：静态（checkpoint 里带）还是动态（在线 amax）？
3. **存储格式**：int8 / int4 / fp8_e4m3 / fp4_e2m1；4bit 要两两打包进
   一个字节（uint8 存储）。
4. **注意力在哪一步反量化**：写入前量化、读取后反量化成稠密再算注意力
   （dequant-gather），还是直接把 scale 喂给 fused attention 算子
   （在线 antiquant，本库 NPU 路径的做法）。

**打包维度**由契约层 `resolve_layout(dtype, head_size)` 给出
（head_size=128 时）：

| dtype | storage_dtype | packed_last_dim | 有效位宽 | 相对 fp16 |
|---|---|---|---|---|
| `int8_per_token_head` | int8 | 128 | 8 bit | **2x** |
| `kivi_int4` | uint8 | 64 | 4 bit（历史区） | **4x**（残差区 fp16） |
| `int4` | uint8 | 64 | 4 bit | **4x** |
| `fp4_e2m1` | uint8 | 72（8×9B：数据+scale） | ~7.1 bit/元素 | **~4.5x** |
| `nvfp4` | uint8 | 72（64B 数据+8B scale） | ~7.1 bit/元素 | **~4.5x** |
| `fp8_e4m3` | uint8 | 128 | 8 bit | **2x** |

## 3. 六个方案分述

六个方法分两个家族：**有状态家族**（int8_dynamic、kivi_int4，自带完整
NPU attention 前向实现）与 **格式方法家族**（packed format：int4_packed、
fp4_e2m1、fp8_e4m3、
nvfp4，语义类只负责存储 dtype 与 scale 元数据，量化计算由宿主
attention backend 内核执行）。

### 3.1 `int8_dynamic` — 动态 per-channel INT8

- **出处**：ascend#116/0001。dtype 契约键：`int8_per_token_head`。
- **语义**：首次 prefill 时沿 token 维（dim=0）取 amax，每个
  (kv_head, head_dim) 通道得到一个 scale，`inv_scale = 127/amax`，
  零偏移恒为 0（对称量化）；`clamp(round(x·inv_scale), -128, 127)`。
  scale 之后固定不变（在线 amax 只算一次）。
- **NPU 路径**：不做打包（真 int8 存储，2x）。注意力走
  `npu_fused_infer_attention_score` 在线 antiquant：decode 用 BNSD 布局
  直接读分页 int8 缓存并携带 antiquant_scale；prefill/chunked-prefill
  用 TND 布局，必要时把分页缓存 gather 反量化成稠密（见
  [npu-implementation.md](npu-implementation.md) §3.2）。
- **约束**：head_size 是 8 的倍数。
- **适合**：想要最稳、最简单的 2x 节省；fused attention 直接支持
  antiquant，无需自定义解包内核。

### 3.2 `kivi_int4` — KIVI：int4 历史区 + 全精度残差窗口

- **出处**：ascend#116/0003–0013（最终状态）。dtype 契约键：`kivi_int4`。
- **核心思想**（KIVI 论文思路）：键对**位置**敏感、值对**通道**敏感，
  且越新的 token 越重要。于是每个请求的缓存分两个区域：
  - **残差窗口**：最近 `residual_length` 个 token 保持全精度（每请求
    一行 ring buffer）；
  - **历史区**：更早的 token 量化成 int4 打进分页缓存——**键**按
    token 组（每 `group_size` 个连续 token 一组，组内逐 (head, dim)
    求 min/max，非对称 [mn, scale] 量化）；**值**按 head 维组
    （每个 token 的 head_dim 按 group_size 分组）。
  键窗口写满后**整组 flush** 进历史区；注意力时把历史区 gather 出来
  反量化成稠密，与全精度残差尾部拼接后走 TND fused attention。
- **约束**（`validate_kivi_geometry`，全部 fail-closed）：
  `group_size % 8 == 0`（int32 打包 1 word = 8 lane）、
  `residual_length % group_size == 0`（整组 flush）、
  `head_size % 8 == 0`、`head_size % group_size == 0`（值按 head 维分组）、
  `block_size % group_size == 0`（flush 对齐检查）。
  两组旋钮对应宿主 `cache_config.kivi_group_size` /
  `kivi_residual_length`（默认 128/128）。
- **NPU 路径**：triton-ascend 打包内核（910B2 逐位验证）+ 纯 torch
  dequant-gather（**刻意不走**上游融合 gather 内核，原因见
  [npu-implementation.md](npu-implementation.md) §3.3）。
- **适合**：追求 4x 级节省且能接受残差窗口精度兜底的长上下文场景；
  这是六个方案里工程状态机最复杂的一个（残差 ring buffer、flush
  状态机、请求行分配，见 npu-implementation.md §4）。

### 3.3 格式方法家族：`int4_packed` / `fp4_e2m1` / `fp8_e4m3` / `nvfp4`

- **出处**：ascend#160/0001（+0004/0005/0007 gating 语境）。
- **共同结构**：每个格式语义类（`PackedFormatSemantics` 子类）只有四个类级元数据
  （`scheme_key` / `cache_dtype` / `storage_torch_dtype_name` /
  `uses_scales`）+ 三个行为：
  `create_weights`（在 attention layer 上挂存储 dtype 与可选 scale 参数）、
  `process_weights_after_loading`（scale 整理）、`apply`（**永远抛
  RuntimeError**——量化发生在 attention backend 内核里，apply 被调用
  说明接线错了，fail-closed）。
- 注册键命名空间化为 `VLLM_HUST_KV_*`，避免与宿主在树 scheme 撞键
  （宿主注册表重复键直接抛错）。

| 方案 | scheme_key | 存储 | scale | 语义 |
|---|---|---|---|---|
| `int4_packed` | `VLLM_HUST_KV_INT4` | uint8（2×int4/字节） | per-token-head，layer 上挂 k/v scale 参数 | 对称量化，backend 内核执行 |
| `fp4_e2m1` | `VLLM_HUST_KV_FP4_E2M1` | uint8 | **不挂** layer scale（scale 随块走） | MXFP4 微缩：每 16 元素共享 1 个 fp8 指数 scale |
| `fp8_e4m3` | `VLLM_HUST_KV_FP8_E4M3` | float8_e4m3fn | per-tensor，layer 上挂参数 | 支持静态（checkpoint）与动态两种 scale 来源 |
| `nvfp4` | `VLLM_HUST_KV_NVFP4` | uint8 | **不挂** layer scale（fp8 块 scale 随数据走） | fp4 数据 + 每 16 元素 1 个 fp8 scale |

四者的差异本质是**量化粒度与数值格式**：

- `int4_packed`：整数格式，per-token-head 粒度，粒度最细的一档；
- `fp8_e4m3`：浮点格式，per-tensor 粒度最粗，但 fp8 对动态范围友好，
  实现最简单；
- `fp4_e2m1`（MXFP4）与 `nvfp4`：都是 4bit 浮点 + 16 元素块 scale，
  压缩率最高；差别在块 scale 的编码约定（MXFP4 用 fp8 指数 scale、
  NVFP4 用 fp8 块 scale，打包布局也不同：fp4_e2m1 每块
  `8B 数据 + 1B scale` 交错，nvfp4 是数据 half + scale half 分离，
  见 `dtypes.fp4_e2m1_packed_dim` / `nvfp4_packed_dim`）。
- **宿主支持注意**：`fp4_e2m1` 在 vllm-hust 上**没有可协商的
  CacheDType 字面量**，适配器 fail-closed 拒绝（加字面量是宿主路线图
  项，不是运行时 hack，见 [integration.md](integration.md) §3.2）。

### 3.4 与两个有状态方案的本质区别

| | 有状态家族（int8_dynamic / kivi_int4） | 格式方法家族（packed format） |
|---|---|---|
| 方法库提供什么 | 完整 NPU attention 前向实现（attention mixin：存储路径 + 计算路径 + 状态机） | 只提供格式语义类（存储 dtype + scale 参数） |
| 量化计算在哪 | 方案自己的 mixin（torch_npu / triton-ascend，惰性导入） | 宿主 attention backend 内核 |
| 挂载方式 | 注册 scheme 后由 `create_weights` 做 C8 式 `layer.impl.__class__` 类手术 | 注册 scheme 即完成 |
| 本库测试覆盖 | 语义层 CPU 全测；内核 910B2 逐位验证 | 格式语义类行为 CPU stub 全测 |

## 4. 横向对比总表

| 方案 | 有效压缩 | 量化粒度 | scale 来源 | 有状态/无状态 | NPU 内核 | vllm-ascend-hust | vllm-hust |
|---|---|---|---|---|---|---|---|
| `int8_dynamic` | 2x | per-channel (head,dim) | 动态（首 prefill amax，之后固定） | 有状态 | torch_npu fused attention 在线 antiquant | scheme + impl 手术 | CUSTOM backend，字面量 `int8_per_token_head` |
| `kivi_int4` | 4x（历史区）+ fp16 残差 | 键 per-token-group / 值 per-head-dim-group，非对称 | 动态（flush 时算） | 有状态 | triton-ascend pack + torch gather | scheme + impl 手术 | CUSTOM backend，字面量 `int4_per_token_head` |
| `int4_packed` | 4x | per-token-head | 动态（backend 内） | 无状态 | backend 内核侧 | scheme（`VLLM_HUST_KV_INT4`） | CUSTOM backend，字面量 `int4_per_token_head` |
| `fp4_e2m1` | ~4.5x | 16 元素块 | 块内 fp8 指数 scale（随数据走） | 无状态 | backend 内核侧 | scheme（`VLLM_HUST_KV_FP4_E2M1`） | **不可用**（无 dtype 字面量，fail-closed） |
| `fp8_e4m3` | 2x | per-tensor | 静态或动态 | 无状态 | backend 内核侧 | scheme（`VLLM_HUST_KV_FP8_E4M3`） | CUSTOM backend，字面量 `fp8_e4m3` |
| `nvfp4` | ~4.5x | 16 元素块 | 块内 fp8 scale（随数据走） | 无状态 | backend 内核侧 | scheme（`VLLM_HUST_KV_NVFP4`） | CUSTOM backend，字面量 `nvfp4` |

成熟度口径（截至 0.2.0.dev0）：**语义层**全部方案 CPU 测试覆盖
（`pytest -q` 全绿）；**设备执行**只验证到端口保真度（pack/gather 在
910B2 上对照 CPU 参考逐位通过）；**端到端 serving** 需要 Ascend NPU
宿主环境，属宿主集成路线图项。vllm-hust 的 CUSTOM backend 适配器是
接口就绪的脚手架。

## 5. 怎么选

- **只想省一半显存、要最快跑通**：`int8_dynamic`。粒度细、无需打包、
  fused attention 原生支持 antiquant。
- **长上下文、追求 4x**：`kivi_int4`。残差窗口兜住近期 token 精度，
  历史区 int4；代价是配置约束多、状态机复杂。
- **在 vllm-ascend-hust 上做 scheme 级集成实验**：packed 四件套
  （`int4_packed` / `fp8_e4m3` / `nvfp4` / `fp4_e2m1`）。它们只定义存储契约，
  计算走宿主 backend，适合先打通注册/分发链路。
- **精度优先级**（经验法则，需按模型实测）：per-channel INT8 ≈
  fp8_e4m3（带动态 scale）> KIVI INT4 > 纯 INT4 > fp4/nvfp4 系。
  上线前务必用目标模型跑精度回归。
