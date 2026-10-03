# 行业 KV 量化方案调研与本仓库进展对标

> 调研时间：2026-09-12（首版 2026-09-11，本版补齐了论文来源核验并
> 刷新进展盘点）。来源分两档：**已核验**（联网核对过官方文档/论文/
> issue，见文末来源清单）与少量细节凭知识整理（已逐条标注）。结论
> 如与最新版本不符，以官方文档为准。

## 1. 一页结论

1. **生产默认是 FP8 E4M3**：上游 vLLM 官方博客（2026-04）的结论是
   FP8 KV cache "已可作为多数长上下文部署的默认起点"（H100 上
   decode ITL 斜率降到 BF16 的 54%，精度保持 94–98%）；上游 vLLM
   **不支持 INT8 KV**（社区 issue #33480 仍在请求，2026 年的工程
   指南并提示 INT8 在长上下文检索类任务可能比 FP8 多损失 1.5–3
   个点），TRT-LLM 则 INT8/FP8/NVFP4 三种 KV 量化算法并列。
2. **Ascend NPU 侧的现役方案是 INT8（C8）**：vllm-ascend 的
   additional config 支持 `kv-cache-dtype: int8`（当时仅此一种），
   且社区 issue 显示 **C8 INT8 KV 在 v0.13.0rc1 中被移除**——这正是
   本仓库"外部可插拔方案库"定位的直接依据。本仓库的宿主注册链路
   已在真实 vllm-ascend-hust 宿主（910B2 容器）进程内实测通过。
3. **低比特（4bit/2bit）是研究活跃区**：KIVI（键 per-channel、值
   per-token 的非对称 + 全精度残差窗口）、KVQuant（per-channel、
   非均匀码本、pre-RoPE、attention-sink 稀疏分解）是两条被广泛引用
   的主线；ZipCache（显著 token 识别）、Palu（低秩投影）、SKVQ
   （滑窗重排 + 截断量化）、GEAR（量化+低秩+稀疏三路）从不同角度
   攻 2–4bit 的离群问题；本仓库 `kivi_int4` 是 KIVI 思想的 int4
   工程化变体。
4. **scale 粒度在变细，且 4bit 块微缩正在硬件落地**：per-tensor →
   per-head（vLLM FA3 静态 per-head scale）→ 16 元素块微缩（Blackwell
   MX/NVFP4）。**新动态**：NVIDIA 已发布 NVFP4 KV cache 的长上下文/
   大批量优化博客（对比 FP8 KV cache 至多 3x 延迟降低、命中率
   +20%），vLLM 社区出现内核级 NVFP4 KV 支持请求（issue #32220）且
   `--kv-cache-dtype nvfp4` 已有实跑报告（v0.24.0 / DGX Spark）——
   本仓库 `nvfp4` / `fp4_e2m1` 契约正站在这个前沿上，缺的是设备内核。
5. **本仓库的位差**：语义与内核数学已到"逐位验证"级，宿主**注册
   链路**已在真实 vllm-ascend-hust 宿主进程内实测通过（统一激活管线
   + 幂等再激活）；但**端到端 serving、精度/吞吐评测、packed 方法
   的设备内核**尚未落地——这是与业界生产实现（vLLM/TRT-LLM）的
   主要差距，详见 §5。

## 2. 推理框架的生产实现（已核验来源）

### 2.1 上游 vLLM — FP8 E4M3，事实上的行业默认

- 入口 `--kv-cache-dtype fp8`（`fp8_e4m3`）；文档明确 FP8 E4M3
  "通常只有极小精度损失"。
- 官方博客（2026-04《The State of FP8 KV-Cache and Attention
  Quantization in vLLM》）细节：
  - scale 粒度：**per-tensor 未校准（scale=1.0，默认）** 与
    **per-KV-head 静态**（FA3 支持每头一个 scale，为此扩展了
    `reshape_and_cache_flash` 内核）；可选 LLM-Compressor 校准。
    博客未涉及 per-block / 动态 scale。
  - 硬件：Hopper（FA3 后端，修复了长上下文两级累加精度问题——
    128k NIAH 曾从 91% 掉到 13%，修复后 89%）与 Blackwell
    （FlashInfer 后端）。
  - 混合注意力（gpt-oss 类 sliding-window 层）：`--kv-cache-dtype-
    skip-layers sliding_window` 跳过小窗口层，break-even 从 ~741k
    token 降到 ~7.7k。
  - 结论："ready to be the default starting point for many
    long-context vLLM deployments"；短上下文（<~7k）、head_dim=256
    且在意 TTFT、未校准精度 <95% 时建议留在 BF16。
- **不支持 INT8 KV**：issue #33480 在请求，理由是 A100/4090 等无
  FP8 硬件的存量卡用不上。

### 2.2 vllm-ascend — INT8（C8）是 NPU 侧现役路径，且刚经历移除

- 官方 additional config 文档：KV cache 量化目前**仅支持 int8**，
  需配合 `kv-cache-dtype: int8` 等配置。
- issue #5630：用户报告 **C8 INT8 KV 量化在 v0.12.0rc1 存在、
  v0.13.0rc1 被移除**，质疑回归。这说明：(a) Ascend 侧 fused
  inference 算子生态以 INT8 antiquant 为主（`npu_fused_infer_attention_score`
  的 antiquant 参数与我们的 int8_dynamic 路径一致）；(b) 在树实现
  存在被版本演进移除的风险——**外部可插拔方案库（本仓库）正是对
  这种风险的补位**。

### 2.3 TensorRT-LLM — INT8/FP8/NVFP4 三线并列；NVFP4 KV 开始落地

- `QuantAlgo` 提供 `INT8_KV_CACHE` / `FP8_KV_CACHE` /
  `NVFP4_KV_CACHE`；可对未内置 FP8 KV 的 checkpoint 手动开启。
- NVFP4（E2M1 + 块 scale）随 Blackwell 微缩（MX）格式支持注意力与
  KV cache 的 4bit 路径。
- **新动态（本版补充）**：NVIDIA 开发者博客《Optimizing Inference
  for Long Context and Large Batch Sizes with NVFP4 KV Cache》给出
  NVFP4 KV 的实测口径——对比 FP8 KV cache 至多 **3x 延迟降低、命中率
  +20%**；上游 vLLM 侧出现内核级 NVFP4 KV 支持请求（issue #32220），
  社区已有 `--kv-cache-dtype nvfp4` 的实跑报告（vLLM v0.24.0 /
  DGX Spark，消费级 RTX 50 系亦有反馈）。**4bit KV 正从"硬件能力"
  变成"生态默认选项"**。
- SqueezeBits 的对比评测：KV 量化的收益在 **decode 密集**负载更
  显著；同条件下 **FP8 KV 的吞吐与精度都优于 INT8**（GPU 语境）。

### 2.4 其它框架

- **SGLang**：`--kv-cache-dtype` 支持 FP8（文档覆盖 FP8/FP4），
  注意与 `--quantization fp8`（权重量化）分开配置；社区有 FP8 KV
  在多模态输入下的隐性退化报告。
- **llama.cpp**：`-ctk q8_0 -ctv q8_0`（2x）与 `-ctk q8_0 -ctv
  q4_0`（~4x）独立指定 K/V 缓存类型；V 量化要求 Flash Attention；
  q8_0 被视为近无损，q4_0 有感度风险。
- **LMDeploy**：提供 INT4/INT8 KV cache 量化选项（KV 量化粒度的
  per-head/per-channel 语境与上面一致）。

## 3. 学术代表方案

### 3.1 两条主线（已核验）

| 方案 | 核心思想 | 与本仓库的关系 |
|---|---|---|
| **KIVI**（arXiv 2402.02750，ICML 2024） | 键 **per-channel**（沿通道维分组，同组覆盖连续 token）、值 **per-token** 的非对称量化；2bit；免调参；**全精度滚动残差窗口**吸收近期 token 与溢出组；2.6x 峰值内存节省、4x batch、2.35–3.47x 吞吐 | `kivi_int4` 的直系来源：分组结构与残差窗口状态机一致，位宽为 int4（论文 2bit）；legacy 出处 ascend#116/0003-0013 |
| **KVQuant**（arXiv 2401.18079，NeurIPS 2024） | 四件套：**per-channel key 量化**（K 通道离群敏感）、**敏感度加权的非均匀码本**（LlamaNormal）、**pre-RoPE key 量化**、**dense-and-sparse 分解**保护 ~0.1% attention-sink 离群；~4bit 近无损，配合卸载冲 10M 上下文 | `kivi_int4` 的键分组同思路；非均匀码本 / pre-RoPE / sink 保护是潜在增强项（见 §6） |

### 3.2 其它常被引用的工作（本版已逐一核验来源）

- **ZipCache**（arXiv 2405.14256，NeurIPS 2024）：基于 token/通道
  维度差异构造**显著 token 识别**指标做自适应混合精度，配
  **canonic 乘法** GPU 内核加速识别阶段；高压缩比下近无损，吞吐
  至多 2.38x。代码：github.com/ThisisBillhe/ZipCache。
- **Palu**（arXiv 2407.21118，ICLR 2025）：把 K/V 的线性层分解为
  **低秩投影**，缓存压缩后的中间态、注意力时即时重建 K/V；配
  算子融合的定制 GPU 内核。50% 压缩率下 RTX 4090 可跑 32K 上下文，
  叠加 4bit 量化可更长。代码：github.com/shadowpa0327/palu。
- **SKVQ**（arXiv 2405.06219，COLM 2024）：**滑窗通道重排序**（提升
  组内通道相似度）+ **截断动态范围量化**（以组中点为中心按 α 收缩
  量化范围，饱和极端值），把 KV 压到 ~2bit。代码：github.com/cat538/SKVQ。
- **GEAR**（arXiv 2403.05527，Microsoft）：**低比特量化 + 低秩近似
  量化误差 + 稀疏矩阵修正离群**三路混合，即插即用叠加在任意 KV
  量化方案上，目标 2–4bit 近无损。代码：github.com/opengear-project/GEAR。
- **LogQuant**（ICLR 2025）：log 分布式 2bit 量化，KVQuant 系后续。

共同主题：**越低的位宽越要处理"离群/显著元素"**——要么给它们单独
更高精度（残差窗口、dense-sparse、低秩残差、稀疏修正），要么换
码本/截断范围/重排通道。这些思路对本仓库 kivi_int4 的精度增强路线
直接适用。

## 4. 技术趋势小结（对标坐标系）

1. **位宽 × 粒度矩阵**：8bit 档（int8/fp8）已生产化，粒度从
   per-tensor 走向 per-head/per-channel；4bit 档靠**块微缩 scale**
   （16 元素共享 fp8 scale）正在 Blackwell 上从硬件能力变成生态默认
   选项（NVIDIA NVFP4 KV 博客、vLLM `--kv-cache-dtype nvfp4` 实跑）；
   2bit 档仍在研究区，依赖残差/离群保护。
2. **非对称 K/V 处理是共识**：K 离群集中、V 平缓——键按通道/值按
   token（KIVI/KVQuant/本仓库 kivi_int4 一致），llama.cpp 干脆允许
   K/V 各选各的 dtype。
3. **计算位置决定成败**：FP8/NVFP4 之所以能做生产默认，是因为
   fused attention **内核内直接吃低比特输入 + 硬件块缩放**；而
   "写前量化、读后反量化回稠密"（本仓库 KIVI 的 dequant-gather
   路径）是内核生态不齐时的务实过渡，收益上让出了部分带宽优势。
4. **部署默认化 + 分层细化**：vLLM 把 FP8 KV 推为长上下文默认起点，
   同时给混合注意力提供按层 skip 的旋钮——"全局一刀切"正在让位
   给"按层/按头/按块"的细粒度控制。
5. **NPU 生态的特殊性**：Ascend fused inference 算子生态以 INT8
   antiquant 为现役路径（FP8 KV 未见于 vllm-ascend 文档），所以
   NPU 上的 2x 节省路径是 INT8（本仓库 int8_dynamic），与 GPU 侧
   FP8 默认形成对照。

## 5. 本仓库实现进展盘点（2026-09-12，0.2.0.dev0，已提交 eced7ac @ dev）

> 术语口径：上一轮 "solution → method" 命名重构已随 `eced7ac` 提交
> ——门面 `kv_methods`、模型类 `KvQuantMethod`/`MethodSpec`、包目录
> `methods/`、环境变量 `VLLM_HUST_KV_METHODS`；原 `int4` 方法更名为
> `int4_packed`（dtype 契约键仍是 `int4`）；triton 内核模块拆分为
> `ops/triton/kivi_pack.py`（在线路径）与
> `ops/triton/kivi_gather_experimental.py`（保留不路由）。
> 测试基线：95 passed / 1 skipped。

### 5.1 已完成且有验证证据

| 能力 | 证据 |
|---|---|
| 六方法统一 API（契约/注册/门面/统一激活管线）、bootstrap 默认 no-op | `pytest -q` 95 passed / 1 skipped（CPU 全测，无 NPU 依赖） |
| KIVI 打包数学 + 残差窗口簿记语义层 | `tests/test_kivi_int4.py`，同时是 NPU 内核的数值参考 |
| KIVI triton-ascend pack 内核（`ops/triton/kivi_pack.py`）+ torch gather | Ascend 910B2 上对照 CPU 参考**逐位通过**（`scripts/npu_smoke_kivi.py`） |
| int8_dynamic 三分支 NPU 前向（decode BNSD antiquant / chunked-prefill / prefill TND）、KIVI mixin 状态机 | 移植自 ascend#116 最终状态，端口保真度；semantics 对拍 |
| 双宿主适配器（ascend scheme 注册 + C8 类手术；vllm-hust CUSTOM 后端 + dtype 字面量协商） | stub 测试全绿；fail-closed 行为全覆盖 |
| **宿主注册链路**（`kv_methods.activate` → 宿主 `register_scheme`） | ✅ 真实 vllm-ascend-hust 宿主**进程内实测**（2026-09-11，910B2 容器 `container-86`：六方法注册可见、int8_dynamic/kivi_int4 激活成功、幂等再激活 OK） |
| **fa_quant_type 分发 + impl 类手术 + 浮点 serving 冒烟** | ✅ 2026-09-12 同容器：Qwen3-0.6B checkpoint 注入 `fa_quant_type`，真实 vLLM 引擎 28/28 层完成类手术，`LLM.generate` 生成成功；int8 存储路径阻塞于宿主 KV 分配/重排不一致（宿主联调工作项） |
| 打包/发布链路（wheel 校验、manifest、CI 3.10/3.12/3.14） | CI 配置 + `scripts/verify-wheel.sh` |

### 5.2 与业界生产实现的差距（按影响排序）

1. **端到端 serving 未完全打通**。业界的 FP8 KV 已是"默认起点"
   （vLLM 博客），TRT-LLM 全内核化。我们的注册链路与 fa_quant_type
   分发/impl 手术已在真实宿主引擎实测（28/28 层，浮点前向生成成功），
   int8 存储路径当前阻塞于宿主 `model_runner_v1` 对
   `int8_per_token_head` 的 KV 分配/重排不一致——这是一个**定位清晰
   的宿主联调工作项**，不是本库缺陷（见
   [how-to-run.md](how-to-run.md) §8.1）。
2. **没有精度/吞吐评测数字**。业界以 NIAH/AUC、ITL 斜率、
   break-even token 数说话（vLLM 博客、SqueezeBits 对比）；本仓库
   尚未建立对自家方法的精度回归 + 吞吐基线，这是补齐"可发布"前的
   最大缺口。
3. **packed 四方法（int4_packed/fp4_e2m1/fp8_e4m3/nvfp4）无自有设备内核**。
   语义类只定存储契约，量化计算指向"宿主 backend 内核侧"——
   该内核在宿主侧尚未就位；对照 TRT-LLM 的 NVFP4_KV_CACHE，这是
   4bit 路径能否生产化的关键。
4. **注意力内的融合低比特计算缺失**。KIVI 走 gather→反量化→稠密
   TND（务实过渡，§4.3）；内核内反量化/块缩放计算是行业主路线，
   依赖 triton-ascend 修复（融合 gather 误编译，见
   [npu-implementation.md](npu-implementation.md) §3.3）或宿主算子。
5. **scale 策略静态**：int8_dynamic 的 per-channel amax 只在首个
   prefill 算一次，无在线刷新/校准数据集路径；行业有 per-head 静态 +
   LLM-Compressor 校准的组合。
6. **无 KV transfer / 量化布局跨机协议**（HOST_CONTRACT 第四协议），
   而 disagg serving 下量化 KV 布局协商是业界真实需求。

### 5.3 尚未覆盖的行业方向（roadmap 候选）

- 2bit 档 + 离群保护（KIVI 2bit、KVQuant dense-sparse、ZipCache/
  GEAR 思路）——当前 kivi_int4 的残差窗口已是同类保护，可作扩展基座。
- pre-RoPE 键量化、非均匀码本（KVQuant 增益点）。
- 混合注意力按层 dtype skip（vLLM `--kv-cache-dtype-skip-layers`
  对应物）。
- FP8 KV 在 Ascend 上的可行性（依赖 torch_npu fp8 attention 算子
  成熟度，当前 vllm-ascend 文档仅列 int8）。

## 6. 结论与建议优先级

1. **把 int8_dynamic 在 vllm-ascend-hust 上端到端打通**——注册链路
   已实测（`eced7ac`），下一步是 fa_quant_type 分发 + impl 类手术 +
   一条真实 serving 冒烟（对位被移除的 C8，2x 节省是业界共识的默认
   起点，且算子生态现成）；
2. **建 benchmark 基建**：精度回归（NIAH/长上下文问答）+ 吞吐/ITL，
   评测口径对齐 vLLM 博客，让后续方案有可比数字；
3. **kivi_int4 增强**沿 KVQuant 方向（键分组已对齐；pre-RoPE、
   sink 保护是论文验证过的增益）；同时跟踪 triton-ascend 修复，
   重验融合 gather 以收敛到内核内计算；
4. **packed 方法**紧盯 NVFP4 KV 的生态落地节奏（NVIDIA 博客 +
   vLLM #32220）：宿主 backend 内核或 HOST_CONTRACT 四协议任一落地，
   nvfp4/fp4_e2m1 契约即可直接受益；在此之前保持契约层与
   fail-closed 语义不变。

## 7. 来源清单

已核验（2026-09-11/12 两轮调研在线核对）：

- vLLM 官方博客《The State of FP8 KV-Cache and Attention Quantization in vLLM》：https://vllm-project.github.io/2026/04/22/fp8-kvcache.html （镜像：https://vllm.ai/blog/2026-04-22-fp8-kvcache）
- vLLM Quantized KV Cache 文档：https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- vllm-ascend Additional Configuration（int8 KV）：https://docs.vllm.ai/projects/ascend/en/v0.9.2rc1/user_guide/configuration/additional_config.html
- vllm-ascend issue #5630（C8 INT8 KV 移除）：https://github.com/vllm-project/vllm-ascend/issues/5630
- vLLM issue #33480（INT8 KV 请求）：https://github.com/vllm-project/vllm/issues/33480
- vLLM issue #32220（NVFP4 KV Cache 内核支持请求）：https://github.com/vllm-project/vllm/issues/32220
- NVIDIA 博客《Optimizing Inference for Long Context and Large Batch Sizes with NVFP4 KV Cache》：https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/
- TensorRT-LLM 量化特性文档：https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html
- TensorRT-LLM 量化博客：https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/quantization-in-TRT-LLM.md
- SqueezeBits《vLLM vs TensorRT-LLM #8: KV Cache Quantization》：https://blog.squeezebits.com/vllm-vs-tensorrtllm-8-kv-cache-quantization-35079
- SGLang Quantized KV Cache 文档：https://sgl-project.github.io/advanced_features/quantized_kv_cache.html
- llama.cpp issue #21450（V 量化需 FlashAttention）：https://github.com/ggml-org/llama.cpp/issues/21450
- LMDeploy KV 量化文档：https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html
- KIVI 论文：https://arxiv.org/abs/2402.02750
- KVQuant 论文（NeurIPS 2024）：https://arxiv.org/abs/2401.18079 ；代码：https://github.com/squeezeailab/kvquant
- ZipCache（NeurIPS 2024）：https://arxiv.org/abs/2405.14256 ；代码：https://github.com/ThisisBillhe/ZipCache
- Palu（ICLR 2025）：https://arxiv.org/abs/2407.21118 ；代码：https://github.com/shadowpa0327/palu
- SKVQ（COLM 2024）：https://arxiv.org/abs/2405.06219 ；代码：https://github.com/cat538/SKVQ
- GEAR：https://arxiv.org/abs/2403.05527 ；代码：https://github.com/opengear-project/GEAR
- LogQuant（ICLR 2025）：https://iclr.cc/virtual/2025/33542

凭知识整理（未逐一在线核验，采用前请确认）：个别吞吐/加速比数字
（如 ZipCache 2.38x、Palu 压缩语境）转引自上述论文页与检索摘要。
