# 量化 KV cache：这个仓库有什么、七个方案差在哪

本文回答三个问题：这个模块**是干什么的**（§1）、KV cache 量化的
**基本原理**（§2）、七个方案的**差异和怎么选**（§3–§5）。现在仓库
里到底有哪些代码、做到哪一步了，直接看 §4.2。运行方法见
[how-to-run.md](how-to-run.md)，NPU 实现细节见
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

KV cache 量化把 K/V 从 fp16/bf16 压成 int8/fp8（2x）、int4（~4x）、
fp4 块微缩（4.5 bit/元素，~3.6x），
是**不换硬件扩大上下文与并发**的最直接手段。

### 1.2 这个库的定位

- **不是**离线模型权重量化，也不是 Adaptive Quantized KV observer 项目
  （见 README 的技术所有权声明）。
- **是**一个方案库 + 一层宿主胶水：把老分支里验证过的量化实现
  （出处见 [../PROVENANCE.md](../PROVENANCE.md)）整理成几个能独立启用
  的"方案"。每个方案给三样东西：
  1. **dtype 解析**：`int8`、`kivi_int4` 这些字符串实际存成什么格式
     （uint8 还是 int8、每头压到多少字节），全库只有 `dtypes.py` 一处
     说了算；
  2. **量化数学**：scale 怎么算、怎么打包解包、残差窗口怎么记账——
     纯 torch 实现，CPU 上就能测，同时也是 NPU 内核的对照答案；
  3. **装进宿主**：挂在宿主"按 cache_dtype 选 attention 实现"的那个
     入口上（`adapters/vllm_ascend_hust`），不量化时宿主行为一点不变。
- **设备边界**：triton / torch_npu 内核只在 Ascend NPU 上执行；在
  没有 NPU 的环境里跑设备路径会直接报错，不会猜。
- **每个方案现在做到哪一步**（有没有代码、在 910B2 上验没验过、
  命令行能不能选）：见 §4.2。本文 §3 讲的是方案本身的设计，
  不代表代码都已经在这个分支上。

## 2. 原理速览：KV cache 量化在量化什么

注意力计算需要的是浮点 K/V。一个量化方案要回答四个问题：

1. **粒度**：多少个数共享一个 scale（整个张量一个 / 每个通道一个 /
   每 token 每头一个 / 每 16 个元素一个）？粒度越细精度越好、
   开销越大。
2. **scale 哪来**：静态（checkpoint 里带着）还是动态（写入时现场算）？
3. **存成什么格式**：int8 / int4 / fp8_e4m3 / fp4_e2m1；4bit 要两个
   挤进一个字节（uint8 存储）。
4. **在哪一步变回浮点**：写入前量化、读取后解压成稠密再算注意力
   （dequant-gather），还是把 scale 直接喂给 fused attention 算子、
   让它在内核里边解压边算（在线 antiquant，本库 int8 路径的做法）。

每个 dtype 实际存成什么，由 `dtypes.py` 的 `resolve_layout` 说了算
（head_size=128 时）：

| dtype | 存储成 | 每头最后一维 | 有效位宽 | 相对 fp16 |
|---|---|---|---|---|
| `int8_per_token_head` | int8 | 128 | 8 bit | **2x** |
| `kivi_int4` | uint8 | 64 | 4 bit（历史区） | **4x**（残差区 fp16） |
| `int4` | uint8 | 64 | 4 bit | **4x** |
| `fp4_e2m1` | uint8 | 72（8×9B：数据+scale） | 4.5 bit/元素 | **~3.6x** |
| `nvfp4` | uint8 | 72（64B 数据+8B scale） | 4.5 bit/元素 | **~3.6x** |
| `fp8_e4m3` | uint8 | 128 | 8 bit | **2x** |

## 3. 七个方案分述

七个方法分两类：**自己管状态的**（int8_dynamic、kivi_int4、
fp8_per_token_head——插件里有完整的 attention 前向实现，缓存怎么写、
怎么读、请求怎么记账都自己管）和 **只定格式的**（int4_packed、
fp4_e2m1、fp8_e4m3、nvfp4——只声明存储 dtype 和 scale 放哪，
量化计算由宿主 attention 内核做）。
fp8_per_token_head 是 2026-10 新起的：dtype 解析、字节布局、CPU 参考
数学和写入内核已经有了，读路径和宿主接线还没有，命令行还选不了
（§4.2 有细节）。

### 3.1 `int8_dynamic` — 动态 per-channel INT8

- **出处**：ascend#116/0001。dtype 字符串：`int8`（宿主字面量
  `int8_per_token_head`）。
- **怎么做**：第一次 prefill 时沿 token 维取一次绝对值最大值，每个
  (kv_head, head_dim) 通道得到一个 scale，`inv_scale = 127/amax`，
  对称量化（零偏移恒为 0）；`clamp(round(x·inv_scale), -128, 127)`。
  scale 算一次之后就不变了。
- **NPU 上怎么跑**：不做打包（真 int8 存储，2x）。注意力走
  `npu_fused_infer_attention_score` 在线 antiquant：decode 用 BNSD 布局
  直接读分页 int8 缓存并带上 scale；prefill/chunked-prefill 用 TND
  布局，必要时把分页缓存 gather 解压成稠密再算（见
  [npu-implementation.md](npu-implementation.md) §3.2）。
- **限制**：head_size 是 8 的倍数。
- **适合**：想要最稳、最简单的 2x 节省；fused attention 原生支持，
  不用自己写解包内核。

### 3.2 `kivi_int4` — KIVI：int4 历史区 + 全精度残差窗口

- **出处**：ascend#116/0003–0013（最终状态）。dtype 字符串：
  `kivi_int4`。
- **核心想法**（KIVI 论文思路）：键对**位置**敏感、值对**通道**敏感，
  而且越新的 token 越重要。所以每个请求的缓存分两块：
  - **残差窗口**：最近 `residual_length` 个 token 保持全精度（每请求
    一行环形缓冲）；
  - **历史区**：更早的 token 量化成 int4 打进分页缓存——**键**按
    token 分组（每 `group_size` 个连续 token 一组，组内逐 (head, dim)
    求 min/max，非对称量化）；**值**按 head 维分组（每个 token 的
    head_dim 按 group_size 分组）。
  键窗口写满后**一整组**写进历史区；算注意力时把历史区 gather 出来
  解压成稠密，拼上全精度残差尾部后走 TND fused attention。
- **几何限制**（不满足直接报错）：
  `group_size % 8 == 0`（int32 打包 1 word = 8 lane）、
  `residual_length % group_size == 0`（整组写进历史区）、
  `head_size % 8 == 0`、`head_size % group_size == 0`（值按 head 维
  分组）、`block_size % group_size == 0`。
  两个旋钮对应宿主 `cache_config.kivi_group_size` /
  `kivi_residual_length`（默认 128/128）。
- **NPU 上怎么跑**：triton-ascend 打包内核（910B2 上和 CPU 参考逐位
  比过）+ 纯 torch 的 dequant-gather（**故意不用**上游的融合 gather
  内核，原因见 [npu-implementation.md](npu-implementation.md) §3.3）。
- **适合**：追求 4x 级节省、能接受残差窗口兜底的长上下文；这是七件
  里状态最复杂的一个（环形缓冲、flush 时机、请求行分配，见
  npu-implementation.md §4）。

### 3.3 只定格式的四个：`int4_packed` / `fp4_e2m1` / `fp8_e4m3` / `nvfp4`

> **注意：这四个在 dev 分支上**（那时包还叫 `sol.*`）。现在这个分支
> （feat/int4）的代码里没有它们——`src/` 下搜不到
> `PackedFormatSemantics`，`dtypes.py` 也不认这四个 dtype（§4.2 的表
> 里有）。这一节留着对照用。

- **出处**：ascend#160/0001（+0004/0005/0007 gating 语境）。
- **共同结构**：每个格式一个语义类，只有四个元数据（名字 / dtype /
  存储 torch dtype / 带不带 scale）+ 三个行为：
  `create_weights`（在 attention layer 上挂存储 dtype 和可选 scale
  参数）、`process_weights_after_loading`（整理 scale）、`apply`
  （**永远抛 RuntimeError**——量化应该发生在 attention 内核里，
  这个方法被调到说明接线接错了）。
- 注册键带 `VLLM_HUST_KV_` 前缀，避免和宿主自带的 scheme 撞名
  （宿主注册表重名直接抛错）。

| 方案 | scheme_key | 存储 | scale | 语义 |
|---|---|---|---|---|
| `int4_packed` | `VLLM_HUST_KV_INT4` | uint8（2×int4/字节） | per-token-head，layer 上挂 k/v scale 参数 | 对称量化，backend 内核执行 |
| `fp4_e2m1` | `VLLM_HUST_KV_FP4_E2M1` | uint8 | **不挂** layer scale（scale 随块走） | MXFP4 微缩：每 16 元素共享 1 个 fp8 指数 scale |
| `fp8_e4m3` | `VLLM_HUST_KV_FP8_E4M3` | float8_e4m3fn | per-tensor，layer 上挂参数 | 支持静态（checkpoint）与动态两种 scale 来源 |
| `nvfp4` | `VLLM_HUST_KV_NVFP4` | uint8 | **不挂** layer scale（fp8 块 scale 随数据走） | fp4 数据 + 每 16 元素 1 个 fp8 scale |

四者的差别就是**粒度和数值格式**：

- `int4_packed`：整数格式，per-token-head 粒度，最细的一档；
- `fp8_e4m3`：浮点格式，per-tensor 粒度最粗，但浮点天生不怕动态
  范围，实现最简单；
- `fp4_e2m1`（MXFP4）和 `nvfp4`：都是 4bit 浮点 + 16 元素块 scale，
  压缩最高；区别在块 scale 怎么编（MXFP4 用 fp8 指数 scale、NVFP4
  用普通 fp8 scale），打包布局也不同：fp4_e2m1 每块
  `8B 数据 + 1B scale` 交错，nvfp4 是数据一半 + scale 一半分开，
  见 `dtypes.fp4_e2m1_packed_dim` / `nvfp4_packed_dim`。
- **宿主注意**：`fp4_e2m1` 在 vllm-hust 上**没有对应的 CacheDType
  字面量**，选了会被直接拒绝（加字面量要走宿主路线图，不是运行时
  hack，见 [integration.md](integration.md) §3.2）。

### 3.4 两类方案的本质区别

| | 自己管状态的（int8_dynamic / kivi_int4 / fp8_per_token_head） | 只定格式的（packed 四件套） |
|---|---|---|
| 插件提供什么 | 完整的 NPU attention 前向实现（存缓存、算注意力、管请求状态） | 只声明存储 dtype + scale 放哪 |
| 量化计算在哪 | 方案自己的代码（torch_npu / triton-ascend，用到才 import） | 宿主 attention 内核 |
| 怎么挂上去 | 注册后改 attention layer 的实现类（C8 式类手术） | 注册即完成 |
| 测试覆盖 | 语义层 CPU 全测；内核 910B2 逐位验证 | 格式语义类 CPU stub 全测 |

## 4. 横向对比总表

| 方案 | 压缩比 | 量化粒度 | scale 来源 | 有无状态 | NPU 内核 | vllm-ascend-hust | vllm-hust |
|---|---|---|---|---|---|---|---|
| `int8_dynamic` | 2x | 每通道 (head,dim) | 动态（首 prefill 算一次，之后不变） | 有 | torch_npu fused attention 在线 antiquant | scheme + 改实现类 | CUSTOM backend，字面量 `int8_per_token_head` |
| `kivi_int4` | 4x（历史区）+ fp16 残差 | 键按 token 组 / 值按 head 维组，非对称 | 动态（写历史区时算） | 有 | triton-ascend pack + torch gather | scheme + 改实现类 | CUSTOM backend，字面量 `int4_per_token_head` |
| `fp8_per_token_head` | 1.94x（head 128：256B→132B） | 每 (token, head) | 动态（每 token 写入时算 amax/448） | 有 | 写入内核已写好、**没上机验过**；读路径没写 | **选不了**（没接命令行，选了直接报错） | 宿主自带字面量（triton 后端） |
| `int4_packed` | 4x | per-token-head | 动态（backend 内） | 无 | backend 内核侧 | scheme（`VLLM_HUST_KV_INT4`） | CUSTOM backend，字面量 `int4_per_token_head` |
| `fp4_e2m1` | ~3.6x | 16 元素块 | 块内 fp8 指数 scale（随数据走） | 无 | backend 内核侧 | scheme（`VLLM_HUST_KV_FP4_E2M1`） | **不可用**（没有 dtype 字面量） |
| `fp8_e4m3` | 2x | per-tensor | 静态或动态 | 无 | backend 内核侧 | scheme（`VLLM_HUST_KV_FP8_E4M3`） | CUSTOM backend，字面量 `fp8_e4m3` |
| `nvfp4` | ~3.6x | 16 元素块 | 块内 fp8 scale（随数据走） | 无 | backend 内核侧 | scheme（`VLLM_HUST_KV_NVFP4`） | CUSTOM backend，字面量 `nvfp4` |

### 4.1 接下来做哪个：按性价比排序（2026-10-03）

先说三条判断标准（详细论证在
[kvquant-schemes-beyond-int8-int4.md](kvquant-schemes-beyond-int8-int4.md)
§7.3/§9.3）：注意力算子读不了低比特时，压缩**只省显存、不省带宽**；
per-token-head 这条路走 triton，不需要等厂商出新算子，是成本最低的
增量；旋转（Hadamard）必须做进内核里才有意义，做不进去反而更慢，
**做不进去就不做**。

| # | 方案 | 现在的状态（feat/int4） | 下一步 | 为什么排这个位置 |
|---|---|---|---|---|
| 0 | `int8_dynamic` / `kivi_int4` | 命令行能用；int4 还差端到端 serve | 910B2 端到端验证 + 真实模型精度 | 两个已经验证过的基线，后面的排序都相对它们 |
| 1 | `fp8_per_token_head` | dtype 解析、布局、CPU 参考数学、写入内核都有了（没上机、没接命令行） | 上机比对 + 测内核能编译哪些形状 → 写读路径 → 接命令行 | 宿主 triton 后端本来就声明了这个 dtype，一个厂商算子都不用等；scale 逐 token 更新，正好补 int8_dynamic "scale 用一次旧的"这个短板 |
| 2 | `int8_per_token_head` | 还没开始（int8 的分支已经写在写入内核里了） | fp8 验证完之后顺手开 | 同一个内核、同一种布局（每 token 每头 S=head+4），几乎不要钱 |
| 3 | TurboQuant（对准 `turboquant_4bit_nc`） | 宿主有完整实现和纯 triton 内核；NPU 这边没人做 | 去 vllm-ascend 的 #15198/#15821 两个 issue 下认领，别自己另起一套回不了上游的 | 4bit 往下最现实的一级（Hadamard 旋转 + 16 级查找表，和 int4 解包是同一种指令形态）；**只省显存**——GPU 上实测吞吐只有 BF16 的 66–80% |
| 4 | `kivi_int4` + 旋转融进内核 | 还没开始；宿主里有 `single_rht` 可以参考 | 学 SAW-INT4 的块对角 Hadamard，融进 pack 内核，Q 侧做配套旋转 | 显著压组内离群值，是往 3bit/2bit 推的必经之路；**融不进内核就不做** |
| 5 | GEAR 式误差修正 | 还没开始 | 先只做离群值的稀疏修正（不做低秩矩阵乘，读路径不多跑流量） | KIVI 写历史区时误差本来就摆在手上，是"下一个便宜的加法" |
| — | `nvfp4` / fp4 系、FP8 attention、MXFP8/HiF8 | 保持报错 | 等 950 代硬件 | 这代 910B2 没有 FP8 注意力能力（#8102）；NVFP4 是 Blackwell 专属 |
| — | 格码（QTIP 那类）、熵编码、加性 VQ | 不做 | — | 解码串行、依赖数据内容、要重写注意力内层循环，服务化不现实 |

表里的外部数字（TurboQuant 的压缩比/PPL、OSCAR 的数字等）在我们自己
跑出 benchmark 之前只当参考，不当结论（每个数字从哪来、核没核实过，
见调研文档）。

### 4.2 现在仓库里到底有什么（feat/int4 分支，2026-10-03）

判断一个方案做到了哪一步，看五样：`dtypes.py` 认不认这个 dtype
字符串；量化数学写了没有（CPU 上能跑的纯 torch 版）；内核写了没有；
在 910B2 上验过没有；`--kv-cache-dtype` 能不能真的选它。

| 方案 | dtype 解析 | 量化数学 | 内核 | 上机验证 | 命令行 | 测试 |
|---|---|---|---|---|---|---|
| `int8_dynamic` | ✅ `int8` | ✅ `methods/int8_dynamic/semantics.py` | torch_npu FIA 在线 antiquant（decode/chunked/prefill 三条前向分支，`attention_backend.py`） | 选实现那步在真宿主上验过（[validation-int4-20260920.md](validation-int4-20260920.md) §11）；前向分支代码是从老实现原样搬的，**没在引擎里端到端跑过** | ✅ | `test_int8_dynamic.py`（2）+ `test_plugin.py` 公共部分 |
| `kivi_int4` | ✅ `kivi_int4` | ✅ `methods/kivi_int4/{semantics,geometry,byte_cache}.py` | triton-ascend 打包内核（`ops/triton/kivi_pack.py`）+ 纯 torch dequant-gather（`ops/kivi_gather.py`） | ✅ 13 节设备记录（[validation-int4-20260920.md](validation-int4-20260920.md)，`6146ab5` 上每节原样重跑、数字逐位一致）；**端到端 serve 没跑过** | ✅ | `test_kivi_int4.py`（48 个，含 Qwen3.5-35B-A3B 形状的专项测试） |
| `fp8_per_token_head` | ✅ `fp8_per_token_head` | ✅ `methods/fp8_per_token_head/{semantics,byte_cache}.py` | 写入内核 `ops/triton/per_token_head_store.py`（照宿主的 CUDA 参考一行行搬的）；**读路径没写** | ❌ 没上过机（上机要回答两件事：triton-ascend 支不支持往 fp8 指针写、这个内核形状编不编得过） | ❌ 没接（选了会报 "no adapter wired"） | `test_fp8_per_token_head.py`（15 个：dtype 解析/布局/数学/内核参数检查） |
| `int4_packed` / `fp4_e2m1` / `fp8_e4m3` / `nvfp4` | ❌ 本树不认 | ❌ 本树没代码（在 dev 分支，见 §3.3） | — | — | ❌ | — |

表之外还有几块相关的代码：

- **选实现和注册**：`adapters/vllm_ascend_hust/backend.py`——按
  cache_dtype 选实现类、开了 context parallel 就报错、没量化的 dtype
  原样交还宿主。真宿主上选实现对过，wheel 装好走 vLLM 自己的插件
  加载也对过（validation 记录 §11/§12）。`bootstrap.py` 是
  `vllm.general_plugins` 的入口——**注册不等于启用**，还要命令行
  传 dtype 才生效。
- **内核的配套检查**：`ops/kivi_layout.py`（打包内核的形状/对齐检查，
  CPU 可测）；`ops/triton/kivi_gather_experimental.py`（在
  triton-ascend 3.5 上编译出错，留着但不用，validation 记录 §5）。
- **怎么复现**：`PYTHONPATH=src python -m pytest -q` → 108 passed
  （2026-10-03）；宿主要改的四个地方在
  `scripts/host_int4_patch.py`（可一键打/还原），改动是否还成立跑
  `scripts/check_int4_patch_parity.py`。
- **还没做完的事**（和 §4.1 的排序对得上）：int4 的端到端 serve、
  真实模型精度、fp8 的读路径和上机验证、ACL Graph（int4/int8 都没有
  捕获分支，先 `--enforce-eager`）。

## 5. 怎么选

> 今天真正能选的只有两个：`int8_dynamic` 和 `kivi_int4`。§4.1 表里
> #1 往后是计划，不是现状。

- **只想省一半显存、要最快跑通**：`int8_dynamic`。粒度细、不用打包、
  fused attention 原生支持。
- **长上下文、追求 4x**：`kivi_int4`。残差窗口保住近期 token 的精度，
  历史区 int4；代价是几何限制多、状态最复杂。
- **想在 vllm-ascend-hust 上试 scheme 级集成**：packed 四件套
  （`int4_packed` / `fp8_e4m3` / `nvfp4` / `fp4_e2m1`）——但注意它们
  在 dev 分支上。它们只定义存储格式，计算走宿主 backend，适合先把
  注册/分发链路打通。
- **精度排序**（经验法则，上线前务必用目标模型实测）：per-channel
  INT8 ≈ fp8_e4m3（带动态 scale）> KIVI INT4 > 纯 INT4 > fp4/nvfp4 系。
