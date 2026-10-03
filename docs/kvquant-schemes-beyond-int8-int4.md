# 除 INT8 / INT4 之外的 KV 量化方案调研

> 调研时间：2026-09-22。本文只谈**非纯整数 int8/int4** 的 KV cache 量化
> 方案（浮点与块微缩格式、码本/向量量化、显著性自适应混合精度、旋转
> 与分层位宽分配、误差补偿混合方案）。KIVI / KVQuant / ZipCache 等既有
> 综述见 [industry-survey.md](industry-survey.md)，本仓库六个方案的语义
> 差异见 [schemes.md](schemes.md)。
>
> 证据分三档，逐条标注：**本地实测**（在本机宿主 checkout 里读到的
> 代码与行号，可复现）、**已核验**（联网打开过原始页面/博客/文档）、
> **待核验**（检索摘要或凭知识整理，采用前需复核）。

## 1. 一页结论

1. **最高杠杆项是 TurboQuant，而且它已经在我们的宿主里**。
   `vllm-hust@ba82f2122` 的 `CacheDType` 已含
   `turboquant_k8v4 / turboquant_4bit_nc / turboquant_k3v4_nc /
   turboquant_3bit_nc` 四个字面量，并且是**完整实现**：独立 attention
   backend + 纯 triton 的 store/decode 内核 + Lloyd-Max 最优码本
   （本地实测，§4）。它同时具备"硬件没有 fp8 就用软件编码"的降级路径
   （§4.3），这正是 910B2 需要的形态。**上游 vllm-ascend 已经有两条同
   主题的 open 请求（#15198 / #15821，§4.4）**，我们要做的是认领与对齐，
   而不是另起一条无法回上游的实现。
2. **Ascend 上确实存在 FP8 KV 算子，但只在稀疏 MLA 路径上**（本地实测，
   修订本文早先的说法）：`vllm-ascend-hust@b0613602f` 的
   `device_op.py:1540-1546` 在 `kv.dtype ∈ {float8_e4m3fn, float8_e5m2}`
   时调用 `torch_npu.npu_kv_quant_sparse_flash_attention`，输入是
   `npu_lightning_indexer` 产出的 `topk_indices`——**稠密 GQA 用不到它**。
   上游还把 `FP8_ATTENTION` 标为 **A5 专属**能力（A2/A3/310P 的硬件能力
   集合里没有），910B2 属 A2 系。另一处本地实测的关键约束：宿主对非
   fp16/bf16 的 KV dtype 要求 `enable_c8_quant`，而它等价于
   `quant_description.kv_cache_type == "C8"`（`modelslim_config.py:931`）
   ——**即 checkpoint 必须预量化**。这既是障碍，也正是本仓库
   "动态 scale、免校准"的定位空间（§9）。
3. **GPU 侧 FP8 是生产默认**（已核验）：vLLM 官方口径 FP8 E4M3 可作为长
   上下文部署起点；社区工程指南提示 INT8 在长上下文检索类任务比 FP8 多
   损失 1.5–3 点；TRT-LLM 的 `kv_cache_config.dtype` 只接受
   `auto|fp8|fp8_ds_mla|nvfp4`（**INT8 KV 已退到 XQA 支持矩阵与遗留页
   面**）。
4. **NVFP4 KV 是 Blackwell 专属，短期与本项目无关**（已核验）：NVIDIA
   公布 NVFP4 KV 的收益（对比 FP8 内存 −50%、TTFT 至多 3x、命中率
   +20%）明确要求 Blackwell + TensorRT Model Optimizer；上游 vLLM 的
   `nvfp4` KV 只在 FlashInfer + SM100 trtllm-gen 内核上可用（本地实测：
   `flashinfer.py:696` 对不满足条件直接报错）。本仓库 `nvfp4` 契约保留
   fail-closed 即可，不值得排期做内核。
5. **旋转（Hadamard）不是某个方案的专利，而是可复用的前置件**（本地实
   测 + 已核验）：宿主的 `int4_per_token_head` 与 TurboQuant 都在用
   RHT/Hadamard 旋转。旋转把离群值"摊平"，因此它与**任意**编码网格
   （fp8 / int4 / 2bit / 码本）正交组合——这是把 kivi_int4 往低比特推时
   成本最低的一招。2026 年这条路最完整的工程样板是 **SAW-INT4**
   （arXiv 2604.19157，块对角 Hadamard + 分页 128-block 对齐）：精度
   接近 BF16、吞吐与普通 INT4 持平，代价是**必须自己发布融合旋转的内核
   fork**——不融合时旋转是带宽净损失（§7.3）。另需注意：除 SGLang fork
   之外，旋转类方案没有进过任何服务引擎主干（PolarQuant 走的是 HF
   Transformers，issue #45203 已合并）。
6. **宿主已具备分层混合精度的旋钮**（本地实测）：
   `vllm/config/cache.py:118` 的 `kv_cache_dtype_skip_layers` 接受层号或
   注意力类型名（如 `sliding_window`）。学术侧的"按层分配位宽"
   （KVTuner 一类，§6）在我们的宿主上**已有落地入口**，不需要改宿主。
7. **判据先给结论：容量收益 ≠ 带宽收益**（§7.3、§4.4）。在 Ascend 上
   没有低比特 fused attention 算子时，几乎所有方案都只省容量。但要小心
   反向的坑：**码本路线也不是带宽方案**——vLLM 官方 H100 实测
   TurboQuant 吞吐只有 BF16 的 **66–80%**（容量 3.7x，对比 FP8 的 2x），
   因为 decode 是带宽瓶颈、每个 key 字节都要在 QK 点积前展开；官方结论
   仍是"FP8 作默认"。所以在我们的栈上，"能同时省带宽"的已知路径只有
   ** fused attention 的 antiquant 通路**（现仅 int8 验证过），其余都按
   "容量方案"评估。选方案时优先按这条筛，而不是按论文 PPL。
8. **2bit 档的门槛不在算法而在接线**（§5.4）：稳健的低比特方案无一例外
   要"旋转/重排 + sink 或残差 fp16 兜底 + 非均匀码本"三件套，其中前两件
   在我们宿主里**已有实现**（`single_rht`、KIVI 残差窗口）。外部对标建议
   盯 **OSCAR**（2605.17757，有效 2.28bit，已进 SGLang/llama.cpp，vLLM
   集成阻塞在 #46774）。另外本轮核出**四条以讹传讹**（§5.3）：Q2-KV、
   MiKV（含被误引的 arXiv 2406.08019）、EAKE、TriState-KV **均不存在**。
9. **自家文档已修一处口径错误**：`schemes.md` 曾把 `fp4_e2m1`/`nvfp4`
   写成"~7.1 bit/元素、相对 fp16 ~4.5x"，按契约自身的打包维
   （head_size=128 → 72 字节/头/侧）实为 **4.5 bit/元素、3.56x**——两个
   数写反了。与上游口径核对见 §11。

## 2. 坐标系：一个 KV 量化方案要回答的五个问题

按这五个维度比较方案，比按"论文名字"比较有用：

| 维度 | 取值范围 | 决定什么 |
|---|---|---|
| **编码网格** | 均匀整数 / 非均匀标量码本（Lloyd-Max）/ 浮点（E4M3 等）/ 块微缩（16 或 32 元素共享 scale）/ 向量-格码 | 同样位宽下的误差分布；浮点对动态范围友好，码本对分布形状友好 |
| **预处理** | 无 / Hadamard 旋转 / 通道重排 / 截断动态范围 / 非对称零点 | 离群值从哪消失 |
| **粒度** | per-tensor / per-head / per-token-head / per-channel / 每 16 元素块 / 每 group 个 token | 精度与元数据开销（本仓库 KIVI 的 scale+min 就吃掉 ~10% 压缩比） |
| **误差保护** | 全精度残差窗口 / 低秩残差 / 稀疏修正 / dense-sparse sink 分离 / 无 | 2–3bit 能不能用 |
| **计算位置** | 内核内直接吃低比特（fused attention antiquant）/ 读出后反量化成稠密再算 | **决定收益是带宽还是仅容量**——NPU 上尤其关键 |

外加一个工程前提：**是否需要校准数据或离线求逆矩阵**（免调参 vs 需要
校准集 vs 需要训练）。

## 3. 浮点与块微缩格式

### 3.1 支持矩阵（已核验）

| 格式 | 位宽 | scale 粒度 | 谁在用 | 硬件门槛 |
|---|---|---|---|---|
| FP8 E4M3 | 8 | per-tensor（默认 scale=1.0）/ per-KV-head 静态 / 校准 | vLLM 默认推荐、SGLang、TRT-LLM、**vllm-ascend 仅 950DT/A5 教程** | Hopper+（FA3）/ Blackwell（FlashInfer）/ AMD；NPU 侧受 `FP8_ATTENTION` 能力位限制（§9.1） |
| FP8 E5M2 | 8 | per-tensor | vLLM `fp8_e5m2`、SGLang | 动态范围大、精度差，KV 少用 |
| per-token-head FP8/INT8/INT4 | 8/8/4 + fp32 scale | 每 token 每头一组 | vLLM 三个 `*_per_token_head`（**我们宿主已有字面量**，本地实测） | **仅 triton 后端**（本地实测 `triton_attn.py:277-287`）→ NPU 可移植性最好的一档 |
| MXFP8 / MXFP4 | 8 / 4 | 每 32 元素共享 1 个 UE8M0（2 的幂）块 scale | AMD MI350（待核验）、SGLang `mxfp8`（需 SM100+）、Ascend 950 宣称原生（**待核验，未见 KV 用途**） | 块 scale 编码是幂次，误差比 E4M3 块 scale 大（NVIDIA 用它解释 NVFP4 与 MXFP4 的 4–5 点差） |
| NVFP4 | 4 + 每 16 元素 fp8 scale | 16 元素块 + per-tensor 全局 | TRT-LLM、vLLM `nvfp4`（FlashInfer/SM100）、我们宿主已有字面量 | **Blackwell 专属**（SM100/103；SM120 缺 FP4 FMHA 内核） |
| HiF8 / HiF4 | 8 | — | torch_npu 注册了 `_HiFloat8Tensor` 与 `hifloat8` dtype（待核验其 KV 用途） | Ascend 950 代际宣称支持 HiF8 |
| GGUF（llama.cpp） | 8/4 | 块整数 | `-ctk/-ctv` 支持 Q8_0/Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL/BF16 | **无 FP8/FP4 KV**（已核验 arg 列表） |

### 3.2 关键数字（都已核验原始页面）

**vLLM FP8 KV（官方博客 2026-04-22）**
- Hopper FA3 曾出现长上下文两级累加精度问题：128k NIAH 从 91% 掉到
  13%，修复后 89%——**FP8 的坑在内核累加，不在量化本身**。
- decode 的每 token KV 开销 ≈ BF16 的 54%；AUC 恢复 Hopper 94–98%、
  B200/FlashInfer 93–96%。
- `head_dim=256` 时 prefill TTFT 比 BF16 差 ~1.6x；短上下文（<~7k）
  不建议开。
- per-head 静态 scale、按层 skip 均由 PR 落地（#30141/#30833/#33695）。

**NVIDIA NVFP4 KV（开发者博客 2025-12-08）**
- 对比 FP8 KV：内存 **−50%**、可服务上下文 **2x**、TTFT 至多 **3x**、
  前缀缓存命中率 **+20%**。
- Qwen3-480B-A35B：FP16→NVFP4 在 LiveCodeBench ≈58%→58%、
  MMLU-PRO 78.2→77.4、MBPP 80.8→79.9、Ruler-64K 95.6→94.6（<1% 损失）。
- Llama-3.3-70B MMLU：**FP8 82.5 / NVFP4 81.9 / MXFP4 77.8**。NVIDIA 把
  NVFP4 优于 MXFP4 的 ~5 点差距归因于**块长 16 + E4M3（非 2 的幂）块
  scale**。这条对比是我们判断"要不要在 NPU 上做微缩格式"的核心依据：
  **块 scale 用浮点而非幂次，值 4bit 才站得住**。

**SGLang（已核验文档）**
- `fp8_e4m3`/`fp8_e5m2` 是 **per-tensor 标量**（默认 1.0 会警告精度）；
  FP4 路径在线计算块 scale。
- FP4 E2M1 相对 FP8 容纳 ≈1.78x token，相对 BF16 ≈3.56x（0.5B 数据 +
  1/16B scale）——与本仓库 §11 的勘误一致。
- 精度：Qwen3-235B AIME25 0.7733(FP16)/0.7333(FP8)/**0.6000(FP4)**；
  DeepSeek-R1-0528 0.5067/0.4934/0.4000。**小模型掉得更狠**，
  4bit 浮点不是"免费的 4x"。

### 3.3 对本项目的含义

- FP8/NVFP4 系的收益前提是 **fused attention 内核内直接吃低比特 +
  硬件块缩放**。Ascend 侧 `npu_fused_infer_attention_score` 的 antiquant
  通路目前只验证到 int8（本仓库 int8_dynamic），FP8 需要新算子或
  triton-ascend 内核。
- 若只做"写前量化、读后反量化成稠密"，FP8 相对 int8 的差别主要是
  **动态范围/免校准**，压缩比一模一样（2x）——在 NPU 上属于低性价比
  重复建设。

## 4. 码本 / 向量量化族（含 TurboQuant 专章）

### 4.1 TurboQuant：宿主里已经跑着的非均匀码本方案（本地实测）

**它是什么**：旋转 + 标量码本的 KV 压缩，不是均匀整数格。论文口径
（TurboQuant，arXiv 2504.19874，已核验）：随机正交旋转 → 旋转后的向量
在球面上服从可解析的分布 → 对**方向**做 Lloyd-Max 最优标量量化
（PolarQuant 那一步）+ 对**范数**用 1-bit QJL 残差校正；**在线、免校
准**。我们宿主里的实现是它的工程变体（Hadamard 旋转 + K 侧码本 + V 侧
均匀量化），是否实现了 QJL 残差步未在代码中声明，移植前需按代码逐条
核对，不能照论文叙述当作实现说明。

- **K**：Hadamard 旋转后，用 **Lloyd-Max 最优标量码本**做 MSE 量化
  （`centroids.py:solve_lloyd_max`，对 N(0, 1/d) 分布解 Lloyd-Max 条件，
  质心数 = 2^bits）；`turboquant_k8v4` 的 K 侧则直接存 FP8。
- **V**：均匀量化（`value_quant_bits` 3–4bit）+ 打包。
- **norm correction（NC）**：把量化后的质心向量重新归一到单位范数，
  4bit 处约值 0.8% 困惑度（`config.py` 注释）。
- 质心/旋转矩阵跨层共享（`turboquant_attn.py:377`），无每层额外开销。

**宿主给出的口径数字**（`vllm/model_executor/layers/quantization/turboquant/config.py:71-75`）：

| 字面量 | K/V 位宽 | 压缩 | PPL 代价 |
|---|---|---|---|
| `turboquant_k8v4` | FP8 / 4bit | 2.6x | +1.17% |
| `turboquant_4bit_nc` | 4bit MSE / 4bit + NC | 3.8x | +2.71% |
| `turboquant_k3v4_nc` | 3bit MSE / 4bit + NC | ~3.5x | +10.63% |
| `turboquant_3bit_nc` | 3bit MSE / 3bit + NC | 4.9x | +20.59% |

> 注：这四个数字来自宿主代码注释（ICLR 2026 论文的转述），属**待核验**
> ——要引用到对外材料前需回到论文/自测复现。

**与 kivi_int4 的正面对比**（同为 4bit 档，本地代码口径）：

| | `kivi_int4`（本仓库） | `turboquant_4bit_nc` |
|---|---|---|
| 编码网格 | 均匀 int4，非对称 min-max | Hadamard 旋转 + Lloyd-Max 码本 |
| 离群策略 | 全精度残差窗口 + 分组 | 旋转摊平 + 码本贴合分布 |
| 元数据 | 每 group 一份 fp32 scale+min（约 10% 预算） | 每 token 每头 scale，质心表全局共享 |
| 实测压缩 | 3.56x（默认几何，`KiviByteCacheLayout`） | 3.8x（宿主注释） |
| 内核 | triton-ascend pack（逐位验证）+ torch gather | 纯 triton store + **自带 decode attention 内核** |

关键差别是最后两行：**TurboQuant 把 decode 注意力自己做成了 triton
内核**（从压缩缓存直接算分数，再解包 V），因此它的收益含带宽；
本仓库 KIVI 走 gather→反量化→稠密 TND，收益只在容量。

### 4.2 移植到 Ascend 的可行性（本地实测）

- 内核全部是 `@triton.jit`（`triton_turboquant_store.py`、
  `triton_turboquant_decode.py`），阶段二复用
  `triton_decode_attention._fwd_kernel_stage2`——与本仓库已经在跑的
  triton-ascend pack 内核同一技术栈。
- **对无原生 fp8 的硬件有降级分支**：`_use_fp8_e4b15()` 在 SM<8.9
  （Ampere/Ada）时改用软件编码的 "e4b15"（4 位指数 + bias 15）而非
  原生 e4m3。910B2 属同类情形，这套 emulated-fp8 写法可直接借鉴。
- 缓存布局是"每 token 每头一个 `[key_packed | value]` 合并 slot"
  （`turboquant_attn.py` 文档字符串），与本仓库"两张等大字节缓冲 +
  插件切 6 视图"的接法**不同**，移植时要处理 K/V 合并 slot 的页大小
  协商（对照 `HOST_CONTRACT.md` 第一协议）。
- 风险：`turboquant_attn.py` 用 CUDA graph 支持标记与 workspace，
  本仓库 INT4 目前明确 `--enforce-eager`；内核编译期资源限制
  （910B2 上 pack 内核已有 UB 溢出实测）在 decode 内核上要重新测。

### 4.3 码本 / 向量量化 / 格码全景（已核验）

| 方案（arXiv） | 结构 | 位宽 | 口径 | 代码与真实状态 |
|---|---|---|---|---|
| **TurboQuant** (2504.19874，ICLR'26) | 随机旋转 → 逐坐标 **Lloyd-Max** 标量量化（**不是学出来的码本**）+ 可选 norm correction；**QJL 1-bit 残差被有意删掉**（经 softmax 会放大方差） | 3–3.5 | 近最优失真率；容量 4x+ | **唯一进主干的案例**（§4.4） |
| **PolarQuant（KV）** (2502.02617，Google Research) | 直角→**极坐标**变换后对角度做标量量化，不显式处理范数 | 2–3 | 缓存压缩 >4.2x | 摘要页**无代码** |
| **PolarQuant（keys）** (2502.00527，NeurIPS'25) | 键的极坐标：角度量化 + 长度泛化评测 | 2–3 | 长上下文保持 | ericshwu/PolarQuant（46★，官方，最后提交 2026-04-02，**仅评测脚本**）。`gh search code` 在 vllm/sglang/flashinfer **零命中**——**未进任何框架**；社区三个 fork（含宣称"融合解码内核、4.1x"）**基准数字未核验** |
| **CQ / Coupled Quantization** (2405.03917，NeurIPS'24) | **2–4 个相关通道耦合**成一张小联合码表（"KV Cache is 1 Bit Per Channel"） | 1–2 | 1 bit/通道 | 摘要页无代码 |
| **CommVQ** (2506.18879，ICML'25) | 与 **RoPE 可交换**的加性多码本 VQ | ~2 | 缓存 −87.5% | UMass-Embodied-AGI/CommVQ（27★，**2025-09-02 起停滞**） |
| **RQ-KVQ** (2410.15704，PMLR v262) | **残差向量量化** + 跨层共享码本 | 1–2 | 体积 −75~90% | iankur/vqllm（12★，2024-10 后未动） |
| VQKV (2603.16435) | VQ + 加性式精修 | <2 | 压缩 82.8%、保留 98.6%（LLaMA3.1-8B） | 未列（**仅摘要级**） |
| **QTIP** (2406.11235) | "bitshift" **trellis 格码**，查表→无查表谱系 + incoherence processing | 2–4 | 权重侧 SOTA | Cornell-RelaxML/qtip（190★，HYB 内核是 **matvec，权重不是 KV**）。**未找到任何 KV 侧格码论文** |
| GSRQ 2607.01065 / D-Quant 2609.19880 / JoLT 2607.12550 / Hurwitz 格 2605.27646 / KV 变换编码 2511.01815 | gain-shape RQ、可漂移熵编码、Tucker+JL、格码、TC+熵编码 | <1–3 | — | **arXiv 号仅经检索确认，页面未打开（snippet-only）** |

**查无此文**（本轮负结果）：把 "AQL" 加性量化用于 KV、"KVT：旋转 + 双重
量化（ICML 2024）"、"ResiVector 用于 KV" 都检索不到；Gemini 的 E8P 码本
**没有公开的 KV/TPU 落地材料**（E8/E8P 格码出自 QuIP# 2402.04396，是
**权重**量化）。

### 4.4 TurboQuant 的上游真实状态（本仓库最需要知道的一段）

- **性能口径**（vLLM 官方博客 2026-05-11，H100）：吞吐为 BF16 的
  **66–80%**、延迟 +10~68%、容量至多 **3.7x**（对比 FP8 的 2x）；
  **官方结论仍是"FP8 作为默认"**。也就是说：码本路线在 GPU 上换到的是
  容量，**吞吐是净亏的**——因为 decode 是带宽瓶颈、每个 key 字节都要在
  QK 点积前展开。
- **CI 明确排除 NPU**：后端按平台门控（`cuda.py:165,178`、
  `rocm.py:483`、`xpu.py:151`），**无 NPU 分支**；
  `.buildkite/hardware_tests/ascend_npu.yaml:41,103` 显式排除
  turboquant 后端与目录。
- **但 Ascend 侧已有人在要**：vllm-ascend **#15198**（2026-08-28，
  A2 上的 TurboQuant 4bit 融合算子：KV **−2.19x**、TTFT **−1.41x**、
  吞吐 **+77%**、并发 1→2、精度无损失）与 **#15821**（2026-09-05，
  DeepSeek V4 MLA 1024→260 B/token、**3.94x**、并发 4→10、+31.8%）。
  **两条都是 open feature request、0 评论、未合并**，数字来自提案本身
  （未核验）。
- **对本仓库的直接含义**：我们若做 TurboQuant 移植，**不是**开一条无人
  要过的路，而是接在两个已存在的上游请求上——应优先在 #15198 下认领/对
  齐，而不是在私有分支里重复实现一份无法回上游的代码（见 §10.1）。
- **移植前必须先算清字节**：宿主 `turboquant_attn.py` 头注释写的是
  `[key_packed | value_fp16]`，而 `config.py` 的 preset 又写
  `value_quant_bits: 4`——**两者矛盾**，且注释里的压缩比（k8v4 2.6x、
  4bit_nc 3.8x、k3v4_nc ~3.5x、3bit_nc 4.9x）与"K+V 都按注释位宽算"并不
  自洽。要用代码里实际算出的 `slot_size` 反推每个 preset 的
  字节/token/头，再决定对标口径——**不要引用注释**。

### 4.5 为什么码本类方案在 NPU 上更难（工程判断）

- **可接受档**：16 级 Lloyd-Max 表 = 寄存器/SRAM 内的 LUT，**指令形态与
  我们 INT4 的反量化同构**；CQ 的 2–4 通道联合码表也一样是
  `index → LUT`。这两类是 INT4 之后最现实的两级台阶。
- **不可接受档**：加性/残差 VQ（CommVQ、RQ-KVQ、VQKV）要在解码时对每个
  key 做 m 次**随机**码本 gather + 向量求和，再叠一个与 RoPE 可交换的
  旋转——分页块内随机访存对 L2 不友好，在 Ascend Vector 单元上更糟，且
  旋转是只有按块批量才有机会进 Cube 的小 GEMM。代价是**重写注意力内
  层循环**，不是改 pack/unpack。
- **格码（trellis）**：Viterbi 式解码带路径状态、串行，SIMT/SIMD 适配极
  差；QTIP 自己的缓解（bitshift trellis、lookup-only 层）只在"一行权重
  摊薄"的 matvec 场景成立。第三方 2026 移植（Escham → llama.cpp）报告
  在寄存器压力下**输给 tensor core**。**KV 侧格码没有任何部署先例**。
- **熵/算术编码**（D-Quant、变换编码、CacheGen 一系）：解码串行且数据相
  关，破坏 CUDA graph 的形状恒定（vLLM 已被迫为 TQ 固定
  `tq_max_kv_splits_for_cuda_graph`）与块共享。**在线服务不可行**，只适
  合作为**离线/前缀存储的线格式**——正好落在 §10.7 的 KV transfer 协议
  上。
- **极坐标类**：写路径便宜（每 token 一次 atan2/范数），但解码要把长度
  重新乘进分数内核；社区"融合内核逼近 cuBLAS"的说法**未核验**，且没有
  框架代码。

## 5. 亚 4bit 与显著性自适应混合精度（已核验）

### 5.1 方案表

| 方案（arXiv） | 机制 | 位宽 | 论文口径 | 代码状态 |
|---|---|---|---|---|
| KIVI (2402.02750，ICML'24) | K per-channel / V per-token 非对称 + 全精度残差窗口 | 2 | 峰值内存 2.6x、batch 4x、吞吐 2.35–3.47x | jy-yuan/KIVI（MIT，最后活动 **2025-01-18**，已停更） |
| KVQuant (2401.18079，NeurIPS'24) | per-channel **pre-RoPE** K + 非均匀 **LlamaNormal** 码本 + 逐向量 dense-and-sparse，sink 保 fp16 | 3（下探 2） | PPL 退化 <0.1；A100-80GB 跑 1M；8 卡 10M；加速 ~1.7x | squeezeailab/kvquant（435★，**无内核、无 vLLM 路径**） |
| SKVQ (2405.06219，COLM'24) | 滑窗**通道重排** + 弱通道**截断** | 2bit K / **1.5bit V** | 解码 7x、80GB 上 1M 上下文 | cat538/SKVQ（25★，反量化内核自称"朴素实现"） |
| ZipCache (2405.14256，NeurIPS'24) | 用归一化注意力分数识别显著 token → 逐 token 混合精度 | 均值 2–3 | 压缩 4.98x、精度降 0.38%、解码快 56.9% | ThisisBillhe/ZipCache（33★，最后活动 2024-07-24） |
| QAQ (2403.04643) | 质量自适应：分离显著（离群）分量 + 分组 | 下探 ~2 | "10x 压缩、影响可忽略" | ClubieDong/KVCacheQuantization |
| LogQuant (2503.19950) | **对数域非均匀电平**（贴合 KV 重尾分布） | 2 | 吞吐 +25%、batch +60% | Concyclics/LogQuantKV（16★，2024-08 起 43 次提交） |
| RotateKV (2501.16383，IJCAI'25) | 离群感知成对旋转 + pre-RoPE 分组头旋转 + **sink 感知** | 2 | PPL 损失 <0.3、峰值内存 3.97x、加速 2.32x | 摘要页**无代码** |
| MiniKV (2411.18077) | 层间判别式**双 scale**，逐层选位宽 | 2 | KV 压缩 86%、恢复 98.5%+ 精度 | 摘要页未列 |
| OTT（原名 AccKV，2505.10938，ACL'25） | 追踪"引发离群"的 token 并保 fp16 | 2 | 内存 6.4x、吞吐 2.3x | yisunlp/OTT（11★，**无 LICENSE**） |
| AnTKV (2506.19505) | **锚点 token + 向量量化**：离线学质心、在线留 1% token | **1** | Mistral-7B 6.32 PPL；LLaMA3-8B 840K token/A100；3.5x 吞吐 | 未列 |
| AsymKV (2410.13212，COLING'25) | 逐层非对称残差/投影位宽分配 | **1** | 至多 75% 解码层可用 1bit | 未列 |
| PM-KVQ (2505.18610，ICLR'26) | 位置驱动的 16/4/2bit 渐进混合 + **partial-mixing 残差 VQ** | 2/4/16 | 推理精度 +8%、吞吐 2.73–5.18x | thu-nics/PM-KVQ（30★，无 LICENSE） |
| **OSCAR (2605.17757)** | **离线谱协方差感知旋转**，sink/recent 保 BF16（标题与数字已二次核验） | 2（有效 **2.28bit**） | 与 BF16 平均差 −0.02（Qwen3-32B）、内存 ~8x、吞吐 ~7x | FutureMLS-Lab/OSCAR（**559★，已接 SGLang 与 llama.cpp**）；vLLM 侧是 **open 的 WIP 提案 #46774**——标题自称 "A new 2-bit KV cache quantisation backend that cuts 5x memory than FP16 (Oscar-2)"，25 条评论，**未合并**（已用 GitHub API 核对） |
| **KVarN (2606.03458)** | Hadamard + **对偶尺度方差归一化**（针对长推理误差累积） | 2（均值 2.3） | MATH500 84.8%、AIME24 60.0%、行检索 85% | vLLM RFC #44578（**仅检索摘要，待核验**） |
| **KV-COBRA (2609.24298，2026-09-21)** | 位宽与**秩**联合分配的逐头共优化 | 0.5–4 bpd | 低 bpd 段精度损失最小 | 未知 |

### 5.2 甄别：哪些其实不是"纯量化"

引用这些方案前必须先过这一关，否则对标会失真：

- **要 attention score 的**：ZipCache（显著性指标来自注意力分数）——与我
  们走 fused attention 算子、拿不到分数的现实冲突，**代价被低估**。
- **带 fp16 逃生口的**：KVQuant / RotateKV / OSCAR 都要留 attention-sink
  或 recent-token 的 fp16 侧缓冲。本仓库 KIVI 的残差窗口就是这类逃生口，
  所以比较压缩比时**必须都按有效位宽算**（我们的 3.56x 就是这么来的）。
- **其实是驱逐/稀疏**：AnTKV（VQ + token 选择）、KVmix/A²ATS 家族、
  MixKVQ——显著性驱动的精度分配在效果上接近软驱逐。
- **其实含低秩**：KV-COBRA（bit 与 rank 共优化）、PM-KVQ 的
  partial-mixing（残差向量量化）。

### 5.3 四处"查无此文"（避免以讹传讹）

- **Q2-KV** 与口号式标题"2-bit KV quantization is all you need"：arXiv、
  GitHub、awesome 列表、HF Papers **均无此工作**。
- **MiKV**：`ti:MiKV` 零结果；而**在社区里流传的那个 arXiv 号
  2406.08019 是一篇无关的极端风险模拟论文**（已核验）。最接近的真实
  工作是 MixKVQ (2512.19206，ACL 2026，查询感知混合精度，**仅摘要级**) 与
  "No Token Left Behind" (2402.18096，重要度驱动混合精度)。
- **EAKE**：标题/摘要检索零命中。
- **TriState / 1.58bit KV**：不存在；TriState 与 ITQ3_S (2603.27914) 是
  **权重**三值量化，与 KV 无关。

### 5.4 对本仓库的含义

- **2bit 档的现实门槛不是算法，是接线**：所有稳健的 2bit 方案都要
  ①旋转或重排 ②sink/残差 fp16 兜底 ③非均匀码本。我们已有 ②，
  ①③ 分别对应 §6.1 的宿主 RHT 与 §4 的 Lloyd-Max 码本——**两块积木
  都已在我们宿主里存在**，缺的是把它们接进 KIVI 的 flush 步与内核。
- **OSCAR 应作为外部对标首选跟踪对象**：同为 2bit、有效 2.28bit、
  已进 SGLang/llama.cpp，且公开了 vLLM 集成的阻塞状态（#46774）。它
  的"离线协方差旋转"需要权重（§7.4），与本仓库定位冲突，但其
  **sink/recent BF16 的处理口径**可直接借鉴。
- **KIVI 官方仓库已停更（2025-01-18）**：把 kivi_int4 当作"论文复现"
  对外描述时，不要再暗示有活跃上游可依。

## 6. 旋转 / 变换与分层位宽分配

### 6.1 宿主里已有的旋转路线（本地实测）

`vllm-hust` 的 `int4_per_token_head` 就是一个**完整实现的 QuaRot 式
KV 方案**（虽是 int4，但手法与格式无关）：

- 写路径：`single_rht()`（随机化 Hadamard 变换）→ 打包 int4 → 每
  (token, head) 一份 scale，**4bit 零点藏在 scale 的低尾数位里**
  （`ops/int4_per_token_head.py:1-11`）。
- 读路径：对 Q 做同样的前向 RHT，用 split-dot 注意力内核直接读压缩
  缓存，输出再逆 RHT 并除 `head_size`（`:911-948`）——**内核内计算，
  不反量化成稠密**。
- Hadamard 用 Hadacore（arXiv 2412.08832）的 Sylvester 构造，无需
  存权重矩阵。

含义：**"旋转 + 低比特 + 内核内反量化"在我们宿主上已经是可跑的现实，
不只是论文**。把它换成 fp8 或 3bit 码本，改动集中在编码步而非接线。

### 6.2 分层位宽的落地入口（本地实测）

`vllm/config/cache.py:118` `kv_cache_dtype_skip_layers: list[str]`，
接受层号（`'0'`、`'2'`）或注意力类型名（`'sliding_window'`）。这正是
vLLM 博客里把 FP8 break-even 从 ~741k token 拉到 ~7.7k 的那个旋钮，
**我们宿主已具备**，本仓库插件只需在 `get_impl_cls` 分派处尊重它。

### 6.3 学术侧（已核验来源，注意三处常见误引）

| 方案 | KV 侧做法 | 位宽 | 报告数字 | 代码 |
|---|---|---|---|---|
| QuaRot (2404.00456) | 随机 Hadamard 作用于权重/K/V，KV 在 **post-RoPE** 旋转，免校准 | W4A4KV4 | Llama2-70B 零样本保留 ~99%，ΔPPL ≤0.47 | Meta 有实现 |
| SpinQuant (2405.16406) | **学习**旋转矩阵（Cayley 优化），KV 头输出用 R4 | KV4 | 相对 QuaRot 收窄 45.1% 差距（Llama-3-8B） | facebookresearch/SpinQuant（CC-BY-NC） |
| SAW-INT4 (2604.19157，2026-04-21) | **块对角** Hadamard（BDR）+ token 级 INT4，专为分页布局/128 block 对齐设计 | INT4 KV | Qwen3-4B GPQA 65.82%（BF16 66.67%），吞吐与普通 INT4 持平（~120 tok/s） | togethercomputer/saw-int4 + `sglang-fast-rotation` fork（用 fast-hadamard-transform） |
| RotateKV (2501.16383，IJCAI'25) | 离群感知的**成对旋转** + pre-RoPE 分组头旋转 + 通道重排 | 2bit KV | PPL 损失 <0.3，跨分布稳健 | ZunhaiSu/RotateKV：仅评测/校准脚本，**无内核、无 vLLM 集成** |
| KVLinC (2510.05373) | Hadamard 旋转 + 对残差做**线性修正项** | ≤4bit KV | 摘要页未给数字 | 未定 |
| TurboQuant (2504.19874) | 见 §4.1 | 3–3.5 bit/通道 | 无偏、"近最优失真" | 无官方版；社区 OmarHory/turboquant |

**三条必须记住的更正**（本轮核验时踩到的误引）：SAW-INT4 **不是**
2410.14033（那是无关论文），正确为 2604.19157；KVTuner **不是**
2504.15937（数论论文），正确为 2502.04420（ICML 2025）；"GKD/QAD for
KV = 2405.17754" 未找到对应文献，按**不存在**处理。

### 6.4 按层/按 token 分配位宽（已核验）

| 方案 | 分配粒度 | 位宽 | 数字 | 前提 |
|---|---|---|---|---|
| KVTuner (2502.04420，ICML'25) | 逐层混合精度，预算下进化搜索 | 2/3/4/5bit 混合（均值≈3.25） | 近无损，TGS +21.25% | 离线搜索 |
| KVmix (2506.08018) | 梯度逐层重要度 | K 2.19 / V 2.38 bit | 内存 4.9x、加速 5.3x | **需要梯度** |
| PM-KVQ (2505.18610，ICLR'26) | 渐进分块混合精度，prefill/decode 分道 | 2/3/4 混合 | 长 CoT 保持 | 离线 |
| WindowQuant (2605.02262) | 窗口相似度驱动 | 混合 | — | 离线搜索 |
| Don't-Waste-Bits (2604.04722，CVPR'26) | **逐 token** 学习策略 {2,4,8,FP16} | 自适应 | — | 端侧部署 |

宿主对应物只有 `kv_cache_dtype_skip_layers`（§6.2）——即"某层跳过量化"，
**不是**"每层不同 dtype 的矩阵"；上游也还没有后者（issue #49198 提案按
新近度做渐进精度，仍 open，已核验为**未实现**）。

## 7. 误差补偿与系统级方案（已核验）

### 7.1 三路/两路误差补偿

| 方案 | 机制 | 位宽 | 数字 | 代码 |
|---|---|---|---|---|
| GEAR (2403.05527) | 低比特量化 + **量化误差的低秩近似** + **离群的稀疏修正**，设计成可叠加在任意 KV 量化器之上 | 4bit（有 2bit 变体） | 内存 2.29x，近无损 | opengear-project/GEAR |
| Palu (2407.21118，ICLR'25) | 对 W_k/W_v 做 SVD，**缓存低秩投影后的中间态**，注意力时即时重建 | 可再叠 4bit | 缓存 −50%，加速至多 2.91x | shadowpa0327/palu |
| ThinK (2407.21018) | 查询驱动的 **K 通道剪枝**（只压 K、与量化正交） | K 2bit | 摘要级信息 | — （**仅检索摘要，待核验**） |

对本仓库最直接的是 **GEAR**：它不要求换编码网格，只要"量化时已知的误
差"可存下来——而 KIVI 在 flush 一个 group 时**本来就算出了** min/max
与量化值，误差是可得的。风险是读路径要额外一次低秩 matmul +
稀疏修正，在 NPU 上是净增流量（§7.2）。

### 7.2 系统级：内核设计与跨设备传输

| 方案 | 对 KV 的处理 | 为什么值得记 |
|---|---|---|
| QServe (2405.04532) | KV4 = SmoothAttention 重缩放 + **per-channel pre-RoPE 键量化** + 非对称 HPB-INT4，**打包对齐到 GEMM 无需反量化**，寄存器级打包 | 它证明"4bit KV 不掉速"的关键不在算法而在**打包对齐**——与本仓库 INT4 的 `int4 数据 + 每组 fp32 scale/min` 预算问题同源 |
| CacheGen (2310.07240，SIGCOMM'24) | 相邻 token KV 差值编码 + 算术编码成比特流，**流式网络传输** | 有效 ~0.5–1 B/元素，体积 3.5–4.3x、TTFT 3.2–3.7x 改善；这是唯一对应我们 `HOST_CONTRACT.md` **第四协议（KV transfer）** 的路线 |

### 7.3 关键判据：容量收益 vs 带宽收益（NPU 语境）

§9.3 的分级判断在这里落到具体方案上：

- **默认全是容量收益**。本仓库 KIVI 走
  `gather → 反量化 → 稠密 TND 注意力`，注意力内核仍按 fp16 流量读，
  HBM 一点没省。
- **旋转类在不融合时是带宽净损失**（多一次 Hadamard）。SAW-INT4 之所
  以做到"与普通 INT4 同吞吐"，是因为它们**发布了融合旋转的 SGLang
  fork**（`sglang-fast-rotation` + fast-hadamard-transform）；QServe 靠
  HPB-INT4 对齐让 GEMM 完全免反量化才打平。
- GEAR/Palu 在读取时**增加** rank-r matmul。
- **按层混合精度是唯一同时砍容量与带宽的一类**——但前提是注意力算子
  认每层 dtype。在 Ascend 上这意味着**每层 `fa_quant_type`**，而不是
  Python 侧做 cast（Python cast 等于回退成稠密）。
- CacheGen 减的是**设备之间搬运的字节**，不减设备内读取。

### 7.4 部署前提分诊（决定能否作为插件开关）

| 前提 | 方案 |
|---|---|
| 纯运行时、免权重、免校准 | QuaRot、SAW-INT4(BDR)、RotateKV、KVLinC、**TurboQuant** |
| 需要模型权重 | SpinQuant（优化旋转）、Palu（SVD W_k/W_v）、KVQuant 系 pre-RoPE |
| 需要权重 + 校准集 + 离线搜索 | KVTuner、KVmix（**要梯度**）、WindowQuant、CommVQ（学码本） |
| 运行时学习型策略 | Don't-Waste-Bits |

**结论**：只有第一行能以"安装即用"的形态进我们的插件；第二、三行与本
仓库"不改宿主、不依赖预量化 checkpoint"的定位冲突（README 明确不读
`fa_quant_type`、不要求预量化）。

## 8. 框架支持矩阵（本轮刷新）

### 8.1 我们自己的宿主（本地实测，可直接引用行号）

`vllm-hust@ba82f2122`，`vllm/config/cache.py:19-36` 的 `CacheDType`
共 16 个取值：

```text
auto float16 bfloat16
fp8 fp8_e4m3 fp8_e5m2 fp8_inc fp8_ds_mla
turboquant_k8v4 turboquant_4bit_nc turboquant_k3v4_nc turboquant_3bit_nc
int4_per_token_head int8_per_token_head fp8_per_token_head nvfp4
```

- 判定"是否量化 KV"的口径（`vllm/utils/torch_utils.py:75-85`）：
  `startswith("fp8")` 或 `endswith("per_token_head")` 或 `== "nvfp4"`。
  **注意**：`int4_per_token_head` 因 `endswith("per_token_head")` 命中，
  而我们的 `kivi_int4` 与 `int8` 都不在该口径里——这正是本仓库需要宿主
  加字面量的根因之一。
- 上游 NVFP4 布局助手 `nvfp4_kv_cache_full_dim(h) = h/2 + h/16`
  （`utils/torch_utils.py:414-416`）与本仓库 dev 分支
  `dtypes.nvfp4_packed_dim` **完全一致**（head=128 → 72）。契约口径无偏差。
- 上游把 NVFP4 一页拆成 `[K_data | K_scale | V_data | V_scale]` 视图
  （`nvfp4_kv_cache_split_views`），与本仓库"两张等大缓冲切 6 视图"是同
  一问题的两种解法，可互为参考实现。

`vllm-ascend-hust@b0613602f`：全仓 **零处** 处理
`nvfp4 / turboquant / *_per_token_head`（本地实测）。即上述 dtype 在
NPU 宿主上目前都不可达——除非常驻插件自己接管实现与分配。

### 8.2 外部框架（已核验，全部读到 default 分支源文件；SHA 见 §12）

**要点先行**：`CacheDType` 这个 `Literal` **不是**真正的开关——真正的门
是各 backend 的 `supported_kv_cache_dtypes`（基类默认只有
`auto/float16/bfloat16`，`vllm/v1/attention/backend.py:62`），并按
`docs/mkdocs/gen_files/generate_attention_backends.py` 生成到
`docs/design/attention_backends.md`。**判断"某个 dtype 能不能用"要查后
端表，不是查 CLI 字面量。**

**vLLM 上游 main@1ea7c63f4af7** 的逐项真值：

| dtype | 内核/实现 | 真实门槛 |
|---|---|---|
| `fp8` / `fp8_e4m3` | `csrc/libtorch_stable/cache_kernels.cu` | 最广：FLASH_ATTN、FLASHINFER、ROCM_ATTN、AITER、TRITON_ATTN 与各 MLA 后端 |
| `fp8_e5m2` | `csrc/attention/dtype_fp8.cuh`、`cache_kernels.cu:1107`、`csrc/cpu/` | FLASHINFER / ROCM / TRITON / CPU；**FLASH_ATTN 不支持**（`flash_attn.py:285`） |
| `fp8_inc` | **无树内内核**，全仓仅 3 处引用（config/torch_utils/一个测试） | 事实上的别名占位 |
| `fp8_ds_mla` | `cache_kernels.cu:936`，断言 `kv_lora_rank=512`、`pe_dim=64`、**656 B/token** | 几何锁死，FLASHMLA_SPARSE / DeepSeek-V4 系 |
| `nvfp4` / `nvfp4_4over6` | `csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu` | **仅 FLASHINFER**，且要求 device capability family 100 **且** trtllm-gen 前后向都到位（`flashinfer.py:496`；`:1189` 明写"fa2/fa3 do not support nvfp4"）。内核虽为 SM120 构建（`CMakeLists.txt:1008-1024`）但后端门控拦住 |
| `nvfp4_ds_mla` | `nvfp4_ds_mla_cache_kernels.cu`，**352 B/token**，需 `sm_100a/100f` 家族特定编译 | 主机侧非 SM100 直接拒 |
| `int8_/fp8_/int4_per_token_head` | `triton_attn.py:781` + `ops/int4_per_token_head.py` | **仅 TRITON_ATTN** ← 对本项目最有意义的一行 |
| `turboquant_*` 四预设 | `backends/turboquant_attn.py` + `ops/triton_turboquant_{store,decode}.py`、`ops/turboquant_soa/*`、`ops/flydsl_turboquant_decode.py` | 在 CUDA/ROCm/XPU 后端表内（`cuda.py:165,178`、`rocm.py:483`、`xpu.py:151`），**无 NPU**；会强制 FA2（`arg_utils.py:2568`）；AMD FlyDSL 解码仅 gfx950、`HEAD_SIZE=128`、GQA∈{6,8,16} |

- **`int8` 不在上游 `CacheDType` 里**——vllm-ascend 用 monkeypatch 拓宽
  字面量（`vllm_ascend/patch/platform/patch_kv_cache_dtype.py:52-71`，
  worker 侧另有镜像）。本仓库要宿主加字面量的处境与上游同源。
- **文档与代码脱节**：`docs/features/quantization/quantized_kvcache.md`
  对 `nvfp4` / `turboquant` / `*_per_token_head` **零提及**（全 docs 目录
  搜 `turboquant` 命中 0）；文档只写 FP8、per-head 需 Flash Attention +
  llm-compressor 校准、以及 `--kv-cache-dtype-skip-layers`。**看代码不看
  文档**是这一族目前的常态。

**SGLang main@c79510cc2a33**：CLI 允许
`auto, fp8_e5m2, fp8_e4m3, mxfp8, bf16, nvfp4, fp4_mx_block16, fp4_e2m1`
（`srt/arg_groups/fields/model.py:196-219`）。其中 **`fp4_e2m1` 已标记
deprecated，传了直接 raise**（`mem_cache/kv_cache_dtype.py`）；
nvfp4/fp4_mx_block16 需 `torch.float4_e2m1fn_x2`（CUDA 12.8+ / torch
2.8+）；`mxfp8` 需 SM100+（FA4 路径）；原生 nvfp4 prefill 需
`trtllm_mha` + page%4 + head_dim%64。文档自陈"**FP4 仍是
experimental**"、FP8 "只支持 per-tensor 标量 scale"，且 FP4 与
PD 分离 / 分层 KV / LMCache / 部分投机解码**不兼容**。

**TensorRT-LLM main@3b3cbc436f4c**：`kv_cache_config.dtype` 校验器只接
`auto | fp8 | fp8_ds_mla | nvfp4`（`llmapi/llm_args.py:4434`）。NVFP4 KV
**必须离线 ModelOpt 量化**（`--quant fp8 --kv_cache_quant nvfp4`），且开
NVFP4 KV 时权重/激活只能是 FP8；**没有 MXFP8 KV 选项**；INT8 KV 只剩
XQA 支持矩阵（`features/attention.md:191`）与遗留页面。新增项：NVFP4
**冷页压缩**（`_torch/kv_cache_compression/.../nvfp4_quantization.py`）
——只作用于 host/磁盘层，不在注意力内核里，思路值得注意。

## 9. Ascend NPU 可行性分级（本地实测 + 上游源码）

### 9.1 NPU 上今天真实存在的 KV 通路只有两条

1. **稠密 GQA → 只有 INT8**。`torch_npu.npu_fused_infer_attention_score`
   的 antiquant 参数组（`key/value_antiquant_scale`、
   `key/value_antiquant_mode=0`、`inner_precise=1`、`input_layout="BNSD"`、
   FRACTAL_NZ 5 维视图、`block_size % 32 == 0`）在**上游与本宿主**里都只
   服务 int8（上游 `attention_v1.py:1521-1542` `_forward_c8_decode`）。
   且上游 `AscendAttentionBackendImpl.__init__` 对非 fp16/bf16 的 KV 直接
   raise，除非 `quant_config.enable_c8_quant`——本宿主实测该开关等价于
   `quant_description.kv_cache_type == "C8"`
   （`vllm_ascend/quantization/modelslim_config.py:931`）：
   **静态、要预量化 checkpoint**。这正是本仓库 int8_dynamic
   （在线 amax、不读 checkpoint）占住的位置。
2. **稀疏 MLA → 有 FP8，但要 A5**。本宿主
   `vllm_ascend/device/device_op.py:1540-1546` 在
   `kv.dtype ∈ {float8_e4m3fn, float8_e5m2}` 时走
   `torch_npu.npu_kv_quant_sparse_flash_attention`（配
   `npu_lightning_indexer` 的 `topk_indices`，融合单张 KV）。
   上游把 `HardwareCapability.FP8_ATTENTION` **只给 A5**
   （`vllm_ascend/device/hardware_profile.py:80,332`；A2/A3/310P 集合里
   没有）——**910B2 属 A2 系，用不上**。教程侧印证：MiniMax-M3 在
   **950DT** 上用 `--kv-cache-dtype fp8`；GLM-5.x 用
   `--kv-cache-dtype int8` + `indexer_kv_dtype int8`，且文档把 Sparse SFA
   C8 标为 "**Experimental in v0.23.0**"。

### 9.2 存储 dtype 齐、消费者缺位

`torch_npu/csrc/custom_dtype/Init.cpp:138-148` 注册了
`int4`、`uint1`、**`hifloat8`**、`float8_e5m2`、`float8_e4m3fn`、
`float8_e8m0fnu`、`float6_e3m2`、`float6_e2m3`、**`float4_e2m1fn_x2`**、
`float4_e1m2fn_x2`。**存得下，但注意力侧没有消费者**：HiF8 / MXFP8 /
FP4 在公开材料里全是**权重/激活**量化
（`npu_dynamic_mx_quant(dst_type=torch.float8_e4m3fn)` 的三处调用点
`mla_v1.py:1866`、`sfa_v1.py:1368`、`norm_quant_fusion_pass.py:273` 都不
是 KV cache）。**唯一提到 HiF8/MXFP8 参与注意力相邻缓存的地方是 QLI
索引器（indexer-K），文档明确那不是 SFA 的 KV cache。**

本宿主实际调用的量化类算子清单（去重）：

```text
torch.ops.npu.npu_dynamic_quant                  # int8 动态
torch.ops.npu.npu_dynamic_mx_quant               # 块微缩量化，dst_type 可为 float8_e4m3fn
torch.ops.npu.npu_add_rms_norm_quant
torch.ops.npu.npu_add_rms_norm_dynamic_quant
torch.ops.npu.npu_rms_norm_dynamic_mx_quant
torch.ops.npu.npu_add_rms_norm_dynamic_mx_quant
```

### 9.3 分级结论

**今天就能做（不需要新供应商算子）**：

1. 动态 per-channel INT8（本仓库 int8_dynamic 的现状）。
2. **`fp8_per_token_head` / `int8_per_token_head` 的 triton 路线**——本地
   实测：我们宿主 `vllm-hust@ba82f2122` 的
   `vllm/v1/attention/backends/triton_attn.py:277-287` 的
   `supported_kv_cache_dtypes` 明确包含
   `int4_per_token_head / int8_per_token_head / fp8_per_token_head`，
   且全仓**只有 triton 后端**（`triton_attn.py` 与
   `triton_attn_diffkv.py`）引用 `per_token_head`。也就是说这条**不需要任
   何厂商算子**，只需把
   `triton_reshape_and_cache_flash_per_token_head_quant` 移植到
   triton-ascend。以本仓库已在 910B2 上跑通 triton-ascend pack 内核的现
   状，这是**成本最低的新 dtype**。
3. 把 `float8_e4m3fn` 当真实张量 dtype 用（`STR_DTYPE_TO_TORCH_DTYPE` 与
   `is_quantized_kv_cache` 天然互通）。
4. TurboQuant 式"Hadamard + 16 级 LUT"（纯 triton，可移植；但 WHT 矩阵
   乘与 split-LSE 收尾是新工作，且上游 #15198 已有人在要——§4.4）。

**必须新写内核才能做**：稠密 GQA 的 FP8 fused-antiquant 注意力（FIA 只
有 int8 antiquant 参数）；任何 NVFP4/FP4-E2M1 KV（dtype 注册了但 NPU 无
消费者）；HiF8 / MXFP8 KV（无公开算子、无参考实现）；亚字节 INT4 的
"内核内反量化"读路径（现状是 gather→反量化→稠密，即带宽受限）；
格码/熵编码类（§4.5，判定为**不可服务化**）。

### 9.4 已知的宿主侧拦路 issue（状态经 GitHub API 逐条核对）

- vllm-ascend **#8102** `[Bug]: kv-cache-dtype set fp8 and fp8_e5m2, it
  crash` — **open**，8 条评论，2026-04-09 提出。与 §9.1 的 A5 门槛一致：
  910B2 上 FP8 KV 不是"没做好"，是**能力位缺失**。
- **#13321** `[RFC]: Refactor KV cache management to align vLLM Ascend KV
  cache allocation and reshape with upstream vLLM` — **open**，2 条评论
  （2026-08-01）。**本仓库 `int4-host-integration.md` 记的"宿主 KV 分配/
  重排不一致"正是这一条**，应订阅并在其基础上提接口，而不是各自绕过。
- **#15503** `[Bug]: npu_fused_infer_attention_score silently ignores
  atten_mask past KV position 512 (bool [1,1,1,S], sparse_mode=0)` —
  **open**，0 评论（2026-09-01）。**做长上下文精度评测前必须先排除这条**，
  否则 512 之后的 token 全部失去 mask，任何"NIAH/检索"数字都不可信。
- **#15198** `[Feature]: The TurboQuant fusion operator NPU implementation
  for long-context inference.` 与 **#15821** `[Feature]: TurboQuant
  supports DeepSeek V4 Flash` — 均 **open、0 评论**（2026-08-28 /
  2026-09-05，标题已核对）。**内容里的性能数字来自提案本身，未核验**。

## 10. 选型建议（按投入产出排序）

1. **最便宜的新 dtype 是 `fp8_per_token_head`（走 triton，不要厂商算
   子）**（§9.3）：本地实测，宿主 `triton_attn.py:277-287` 已把三个
   `*_per_token_head` 列为 triton 后端支持的 KV dtype，而**只有** triton
   后端引用它们——意味着移植到 triton-ascend **不依赖 CANN/FIA 新增
   antiquant 参数**，而我们的 pack 内核已在 910B2 逐位跑通。它同时给
   FP8 的动态 per-token-head scale——正好补上本仓库
   `int8_dynamic` 的"scale 只在首个 prefill 算一次"这一短板（见
   [industry-survey.md](industry-survey.md) §5.2 第 5 条）。先做语义 +
   CPU 参考，再测内核形状包络。
2. **TurboQuant 以"对齐上游"的方式做**（§4、§4.4）：宿主已有完整实现与
   纯 triton 内核，接入形态与我们的 `get_impl_cls` 分派匹配。但先做两件
   事：①用代码里的 `slot_size` 反推每个 preset 的真实字节（§4.4 末条，
   注释自相矛盾）；②去 vllm-ascend **#15198 / #15821** 下认领或对齐，
   避免产出一份回不了上游的私有内核。预期收益按**容量**记，不按带宽
   （§1 结论 7）。
3. **把 RHT 旋转作为 kivi_int4 的正交增强，但必须连内核一起做**（§6.1、
   §7.3）：宿主 `int4_per_token_head` 已有 `single_rht` 参考实现，
   SAW-INT4 的**块对角**变体（BDR）更契合我们的分页几何（head 128 落在
   一个 block 内、不跨分页边界）。落点很具体——语义侧在
   `methods/kivi_int4/semantics.py:130-131`（键组 min/max）与
   `:149-150`（值组 min/max）**之前**插入旋转，内核侧同步改
   `ops/triton/kivi_pack.py`；注意力侧照抄宿主的配对做法（Q 做前向旋转、
   输出做逆旋转并除 `head_size`，`int4_per_token_head.py:911-948`），
   否则旋转不进分数计算就是白做。判定标准很硬：旋转若不能融进 pack
   内核，就不合并——那只会换来带宽净损失（§7.3）。
4. **MXFP8-KV 只作为 950 代际的储备项**（§9.2）：`npu_dynamic_mx_quant`
   的编码口径可借用，块 scale 应选 E4M3 而非 2 的幂（依据 NVFP4>MXFP4 的
   4–5 点差，§3.2）；但 A2 系上既无 FP8 注意力能力（§9.1）也无 MXFP8 消费
   者，**现在排期等于做契约层重复建设**。等 950DT/A5 到位后与
   `fp8_ds_mla`（656 B/token）、`nvfp4_ds_mla`（352 B/token）的口径一起看。
5. **精度兜底优先用"已有旋钮"而不是"离线搜索"**（§6.2、§6.4）：接上
   `kv_cache_dtype_skip_layers`（宿主已有，成本最低）；KVTuner/KVmix
   一类逐层位宽搜索要权重 + 校准集 +（KVmix 还要）梯度，与本仓库"不改
   宿主、不要求预量化 checkpoint"的定位冲突（§7.4），**不做**。
6. **GEAR 式误差补偿是下一个便宜的加法**（§7.1）：KIVI flush 时误差已
   可知，可先只做**稀疏离群修正**（不加低秩 matmul），避免读路径增流。
7. **nvfp4 / fp4_e2m1 保持契约 + fail-closed，不排期内核**（§3.2）：
   收益依赖 Blackwell 类硬件块缩放；NPU 侧对应能力在 950 代际
   （MXFP8/MXFP4 宣称支持，待核验），届时契约可直接受益。上游
   `nvfp4_kv_cache_full_dim` 与我们的契约已逐位一致，改内核前先复跑该
   对照。
8. **KV transfer 那条路线现在就能立项**（§7.2）：CacheGen 的差值 +
   熵编码口径正好对应 `HOST_CONTRACT.md` 缺失的第四协议，且与位宽选择
   正交——disagg serving 下量化 KV 布局协商是业界真实需求。
9. **建 benchmark 前不对外引用任何外部数字**（沿用
   [industry-survey.md](industry-survey.md) §6）：本轮所有 PPL/AUC/压缩
   比都来自论文或厂商口径；§4.1 的四个 preset 数字目前只有宿主注释的
   出处，属**待核验**。

## 11. 与自家文档的口径差异（本轮已修）

`schemes.md` §2 契约表把 `fp4_e2m1` / `nvfp4` 记为
"**~7.1 bit/元素**、相对 fp16 **~4.5x**"。按契约自身
（`dtypes.fp4_e2m1_packed_dim` / `nvfp4_packed_dim`，head_size=128 →
72 字节/头/侧）：

```text
每元素有效位宽 = 72 B * 8 / 128 = 4.5 bit/元素
相对 fp16 压缩 = 256 B / 72 B = 3.56x
```

即两个数字被写反了。旁证：SGLang 文档口径为"FP4 E2M1 ≈ 3.56x vs
BF16、1.78x vs FP8"（0.5B 数据 + 1/16B scale），与本仓库算法一致
（已核验）。本仓库 `kivi_int4` 的 3.56x 与之一致，也说明 packed 两行
是笔误而非另一种口径。

**已就地修正**（本轮）：`schemes.md` §1.1 叙述 + §2 契约表两行 + §4
横向对比表两行，统一为"4.5 bit/元素、~3.6x"。仍**保留未动**的是
`kivi_int4` 的"4x（历史区）"写法——它在字面上没错（历史区确实是 4
bit），但对外报预算时必须用 `KiviByteCacheLayout.compression_vs_fp16()`
的 **3.56x**（`HOST_CONTRACT.md` 已明令"不要按 4x 报预算"）。

## 12. 来源清单

**本地实测**（可复现，命令见 §附录 A）：

- `../vllm-hust@ba82f2122`：`vllm/config/cache.py:19-36,118`、
  `vllm/utils/torch_utils.py:75-85,414-416,471-`、
  `vllm/v1/attention/backends/registry.py:117`、
  `vllm/v1/attention/backends/turboquant_attn.py:1-40,73-84,377-387`、
  `vllm/v1/attention/backends/flashinfer.py:339,388-428,688-708`、
  `vllm/v1/attention/ops/int4_per_token_head.py:1-11,59,865-948`、
  `vllm/v1/attention/ops/triton_turboquant_{store,decode}.py`、
  `vllm/model_executor/layers/quantization/turboquant/{config,centroids}.py`、
  `vllm/v1/attention/backends/triton_attn.py:277-287`
  （`supported_kv_cache_dtypes` 含三个 `*_per_token_head`；全仓仅
  `triton_attn.py` / `triton_attn_diffkv.py` 引用 `per_token_head`）
- `../vllm-ascend-hust@b0613602f`：
  `vllm_ascend/device/device_op.py:1540-1546`（FP8 KV 走
  `npu_kv_quant_sparse_flash_attention`，同一函数用
  `npu_lightning_indexer` 取 `topk_indices`）、
  `vllm_ascend/quantization/modelslim_config.py:719,931`
  （`enable_c8_quant ⇔ kv_cache_type == "C8"`）、
  `vllm_ascend/compilation/passes/norm_quant_fusion_pass.py`
  （`npu_dynamic_mx_quant` / `dst_type=torch.float8_e4m3fn`）、
  全仓 `nvfp4|turboquant|per_token_head` 检索为空
- **上游源码（2026-09-22 当日 default 分支 SHA，逐文件读取）**：
  vllm `main@1ea7c63f4af7`、vllm-ascend `main@5591facb371b`、
  sglang `main@c79510cc2a33`、TensorRT-LLM `main@3b3cbc436f4c`、
  torch_npu `master`（pushed 2026-09-22）。关键定位：
  vLLM `v1/attention/backend.py:62`（`supported_kv_cache_dtypes` 真门）、
  `config/cache.py:41-62`、`utils/torch_utils.py:38-55,543`、
  `flashinfer.py:496,1189`、`triton_attn.py:781`、
  `arg_utils.py:2568`、`cuda.py:165,178`、`rocm.py:483`、`xpu.py:151`、
  `.buildkite/hardware_tests/ascend_npu.yaml:41,103`；
  vllm-ascend `attention/attention_v1.py:1521-1542`、
  `patch/platform/patch_kv_cache_dtype.py:52-71`、
  `device/hardware_profile.py:80,332`、`csrc/attention/kv_quant_sparse_flash_attention/`；
  sglang `srt/arg_groups/fields/model.py:196-219`、
  `srt/mem_cache/kv_cache_dtype.py`、`arg_groups/kv_cache_hook.py`；
  TRT-LLM `llmapi/llm_args.py:4434`、`features/attention.md:191`、
  `_torch/kv_cache_compression/.../nvfp4_quantization.py`；
  torch_npu `csrc/custom_dtype/Init.cpp:138-148`、
  `torch_npu/utils/hif8_tensor.py`
- 本仓库 dev 分支：`src/.../dtypes.py:58-82`（打包维公式）
- 本仓库 feat/int4：`methods/kivi_int4/semantics.py:130-131,149-150`
  （旋转插入点）、`HOST_CONTRACT.md`（3.56x 预算口径）

**已核验**（本轮联网打开过原始页面）：

- NVIDIA《Optimizing Inference for Long Context and Large Batch Sizes
  with NVFP4 KV Cache》(2025-12-08)：
  https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/
- vLLM《The State of FP8 KV-Cache and Attention Quantization in vLLM》
  (2026-04-22)：https://vllm.ai/blog/2026-04-22-fp8-kvcache
- vLLM Quantized KV Cache 文档：
  https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- SGLang Quantized KV Cache 文档：
  https://sgl-project.github.io/advanced_features/quantized_kv_cache.html
- TensorRT-LLM 量化文档：
  https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/quantization.md
- vLLM issue **#32220**（NVFP4 KV Cache Support，**closed**，11 评论）：
  https://github.com/vllm-project/vllm/issues/32220
- vLLM issue **#49198**（Recency-based progressive mixed-precision KV
  cache，**open**，0 评论）/ **#46774**（OSCAR-2 2-bit KV backend，
  **open WIP**，25 评论）
- vLLM 官方博客《TurboQuant …》口径（2026-05-11，H100：吞吐 66–80% of
  BF16、容量 3.7x，FP8 仍是默认）
- vllm-ascend issue **#8102 / #13321 / #15503 / #15198 / #15821**
  （§9.4 逐条列状态）
- HF Transformers **#45203**《Add PolarQuant quantization:
  Hadamard-rotated Lloyd-Max optimal weights + KV cache》—
  **closed（已合并）**，22 评论
- 上游 raw 源文件（本轮逐文件读取）：
  https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/cache.py ·
  .../docs/features/quantization/quantized_kvcache.md ·
  https://raw.githubusercontent.com/sgl-project/sglang/main/docs/docs/advanced_features/quantized_kv_cache.mdx ·
  https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/tutorials/models/MiniMax-M3.md ·
  https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/custom_dtype/Init.cpp
- 论文（均已打开 arXiv 摘要/正文页）：QuaRot 2404.00456 · SpinQuant
  2405.16406 · SAW-INT4 2604.19157 · RotateKV 2501.16383 · KVLinC
  2510.05373 · TurboQuant 2504.19874 · CommVQ 2506.18879 · KVTuner
  2502.04420 · KVmix 2506.08018 · PM-KVQ 2505.18610 · WindowQuant
  2605.02262 · Don't-Waste-Bits 2604.04722 · GEAR 2403.05527 · Palu
  2407.21118 · QServe 2405.04532 · CacheGen 2310.07240 · KIVI
  2402.02750 · KVQuant 2401.18079 · SKVQ 2405.06219 · ZipCache
  2405.14256 · QAQ 2403.04643 · LogQuant 2503.19950 · MiniKV
  2411.18077 · OTT/AccKV 2505.10938 · AnTKV 2506.19505 · Coupled
  Quantization 2405.03917 · AsymKV 2410.13212（COLING'25）· OSCAR
  2605.17757 · KV-COBRA 2609.24298 · "No Token Left Behind" 2402.18096
- 代码仓库：facebookresearch/SpinQuant · togethercomputer/saw-int4 ·
  ZunhaiSu/RotateKV（仅脚本）· UMass-Embodied-AGI/CommVQ ·
  cmd2001/KVTuner · opengear-project/GEAR · shadowpa0327/palu ·
  mit-han-lab/omniserve · UChi-JCL/CacheGen · jy-yuan/KIVI ·
  cat538/SKVQ · ThisisBillhe/ZipCache · Concyclics/LogQuantKV ·
  yisunlp/OTT · thu-nics/PM-KVQ · FutureMLS-Lab/OSCAR
- vLLM OSCAR 集成 issue #46774（open / needs rebase）·
  https://github.com/vllm-project/vllm/issues/46774
- OSCAR 项目页 https://oscar-quantize.github.io/ · AMD TurboQuant
  内核博客（FlyDSL，宣称达 BF16 吞吐 95%）
  https://rocm.blogs.amd.com/artificial-intelligence/turboquant-vllm-agentic/README.html
- llama.cpp KV cache 类型参数（`common/arg.cpp`）：GGUF 无 FP8/FP4 KV
- LMDeploy KV 量化文档：
  https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html

**待核验**（引用前需回源）：

- TurboQuant 四个 preset 的压缩比/PPL 数字（宿主代码注释转引
  "Zandieh et al., ICLR 2026"；论文正文表格未打开）。且宿主内部
  `turboquant_attn.py` 头注释与 `config.py` 对 V 的存储宽度**自相矛盾**
  （§4.4 末条）——引用前先按代码算 `slot_size`
- vllm-ascend **#15198 / #15821 提案正文里的性能数字**（KV −2.19x、
  TTFT −1.41x、吞吐 +77%、MLA 1024→260 B/token 等；标题与状态已核对，
  数字未复算）
- Hadacore arXiv 2412.08832（宿主代码注释引用，页面未打开）
- KVarN 2606.03458 与 vLLM RFC #44578（仅检索摘要）
- VQKV 2603.16435 · MixKVQ 2512.19206 · ThinK 2407.21018 细节 ·
  GSRQ 2607.01065 / D-Quant 2609.19880 / JoLT 2607.12550 / Hurwitz 格
  2605.27646 / KV 变换编码 2511.01815（**arXiv 号仅检索确认，页面未打
  开**）
- PolarQuant 三个社区 fork 的"融合内核 / 4.1x"宣称
- Ascend 950 代际 MXFP8/MXFP4/HiF8 的算力口径（仅有华为主题演讲与
  950 白皮书 PDF 链接，**未逐页核对是否覆盖 KV cache**）
- 厂商/论文给出的所有吞吐与精度数字（本轮只做口径对齐，未自测）

### 附录 A：本地实测命令

```bash
# 宿主 dtype 清单
sed -n '19,36p' ../vllm-hust/vllm/config/cache.py
# TurboQuant 实现与 preset 注释
sed -n '60,130p' ../vllm-hust/vllm/model_executor/layers/quantization/turboquant/config.py
# RHT int4 方案
sed -n '1,12p' ../vllm-hust/vllm/v1/attention/ops/int4_per_token_head.py
# Ascend 侧算子清单
cd ../vllm-ascend-hust && grep -rhoE "torch\.ops\.npu\.npu_[a-z0-9_]*(quant|mx)[a-z0-9_]*" vllm_ascend/ | sort -u
```

## 13. 本轮增补（2026-10-03）：状态复核与 §10.1 第一步落地

### 13.1 上游与文献状态复核（当日核对）

- vllm-ascend **#15198 / #15821**（TurboQuant NPU 融合算子 / DeepSeek V4
  Flash）：GitHub API 核对，仍 **open、0 评论**，无进展。
- vLLM **#46774**（OSCAR-2 2bit backend）：仍 **open、needs-rebase、
  25 评论**，最后更新 2026-08-21。
- vllm-ascend **#15503**（FIA 掩码在 KV 位置 512 之后静默失效）：仍
  **open、0 评论**——做任何长上下文精度评测前依旧必须先排除这条。
- 文献面：2026-09-22 → 10-03 检索未发现超出本文覆盖的新方案。最近的
  动向仍集中在 INT2/1-bit（输出感知旋转、MPO 分解、可交换 VQ 等，
  多为 snippet 级、待核验），与 §4.5/§5.4 的门槛判断一致；NVFP4 KV
  仍锁定 Blackwell 家族（SM120 讨论帖口径：FP8 "statistically
  lossless"、NVFP4 +0.01–0.04 nats，待核验），不改变 §3.3/§10.7。

### 13.2 §10.1 第一步已落地（本仓库 feat/int4）

按 §10.1 的顺序（先语义 + CPU 参考，再内核形状包络）：

- **契约层**：`KVQuantMode.FP8_PER_TOKEN_HEAD`（插件本地编号 10），
  `resolve_layout("fp8_per_token_head")`——uint8 存储、不打包、
  head_size % 4 校验（fp32 scale 对齐）。
- **布局**：`methods/fp8_per_token_head/byte_cache.py`——与 KIVI 同一
  "两张等大宿主缓冲、插件切视图"接法，每侧
  `[head_size B E4M3 数据 | 4 B fp32 scale]`，S = head_size + 4，与
  宿主 `real_page_size_bytes` 的 per_token_head 分支
  （数据 `2*block*kvh*head` + scale `2*block*kvh*4`）逐位一致；
  head 128 → 132 B/（token·head·侧），≈1.94x。
- **CPU 参考语义**：`methods/fp8_per_token_head/semantics.py`
  （scale = amax/448、clamp 后 cast、误差界与离群稳健性测试）——
  将来 triton-ascend store 内核的对拍基准。
- **fail-closed 边界**：`adapter_factories` 缺位、
  `bootstrap.REGISTERED_METHODS` 不含它——设备路径落地前
  `--kv-cache-dtype fp8_per_token_head` 不可选，`host_adapter` 抛
  "no adapter wired"。
- 后续（按投入产出排序）：①triton-ascend store 内核（对拍本语义）→
  ②形状包络测量（复用第 10 节的 910B2 方法）→ ③FIA 读通路
  （per-token-head scale 反量化进注意力）→ ④接线注册 + 真宿主分派核对。
- 同轮（2026-10-03）还完成了 kivi_int4 热路径去同步化：
  `ordered_slots` / 残差 store / 残差 gather 的逐 token
  host-device 同步全部向量化（真机 serving 每步原本付
  O(sum(seq_len)) 次同步），顺序语义由逐位置参考实现对拍钉死。

提交：`d432a92`（perf 去同步化）、`eb3a54f`（fp8_per_token_head 脚手架）。
