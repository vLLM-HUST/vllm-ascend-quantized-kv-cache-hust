# KV 量化：现在的进展和接下来值得做的

> 更新：2026-10-03，分支 feat/int4。这份是汇总入口：我们的进展看 §1，
> 学术界哪些方案真有效看 §2，两边对照后决定做什么看 §3–§4。
> 深挖细节：方案分页见 [schemes.md](schemes.md)，调研全文见
> [kvquant-schemes-beyond-int8-int4.md](kvquant-schemes-beyond-int8-int4.md)。

## 1. 我们现在的进展

| 方案 | 做到哪一步 | 差什么 |
|---|---|---|
| `int8_dynamic` | 代码全有，命令行可用 | 没在引擎里端到端跑过 |
| `kivi_int4` | 最扎实：910B2 上 13 节设备验证全过，Qwen3.5-35B-A3B 形状有专项测试；**端到端 serve 已跑通**（稠密模型 3.56x 兑现；Qwen3.5 也跑通，见 [serving-verification-20261003.md](serving-verification-20261003.md)） | decode 吞吐待融合读路径；Qwen3.5 的 GDN 层走纯 torch 兜底（模型适配，非量化问题） |
| `fp8_per_token_head` | dtype 解析、布局、CPU 参考数学、写入内核都有（本周完成） | 读路径没写；没上机；命令行没接 |
| 四个纯格式 | 只在 dev 分支 | — |

环境侧：打榜服务器（vllm-hust-cyj）的 bench 树已经打好宿主补丁、
装好插件，Qwen3.5-35B-A3B 就位——**端到端验证随时可以开跑**，
只等服务器排期。本轮还顺手做完了 KIVI 热路径去同步化
（每 decode 步原来要付几万次 host-device 同步，现在没有了）。

## 2. 学术界哪些方案真有效（2026-10 视角）

四类被验证过的路线，按工程落地程度排：

**① 浮点格式（8bit/4bit 浮点）——最被验证**
- FP8 E4M3：GPU 生产默认，精度几乎无损（任务恢复 94–98%），2x。
- NVFP4：4bit 浮点 + 每 16 元素 fp8 块 scale，Llama-70B MMLU 只掉
  0.6 分。**块 scale 用浮点而不是 2 的幂是关键**。锁 Blackwell 硬件。

**② 2bit 混合精度——学术最活跃**
- OSCAR：有效 2.28bit，和 BF16 平均只差 0.02，已进 SGLang/llama.cpp。
- KVQuant（NeurIPS'24）：3bit，PPL 退化 <0.1。
- RotateKV（IJCAI'25）：2bit，PPL 损失 <0.3。
- ZipCache / LogQuant：按 token 重要性分 bit，平均 2–3bit。
- **共同配方是三件套：旋转摊平离群值 + 留一小段全精度兜底 +
  非均匀编码。缺一件就不稳。**

**③ 码本量化（3–4bit 非均匀格）**
- TurboQuant（ICLR'26）：旋转 + Lloyd-Max 最优码本，3.5bit 近最优
  失真，免校准，已进 vLLM 官方。容量 3.7x，但吞吐只有 BF16 的
  66–80%——**码本路线省显存不省带宽**。
- PolarQuant / CQ：极坐标、联合码表，有效但没进框架。

**④ 误差补偿（叠在任意量化器上）**
- GEAR：低秩 + 稀疏修正补量化误差，4bit 近无损。
- Palu：SVD 低秩投影 + 4bit，压缩一半、加速至多 2.9x。

一条硬判据贯穿所有方案：**注意力内核读不了低比特时，一切压缩只省
显存、不省带宽。** 我们栈上的 fused attention 算子（FIA）只有 int8
antiquant 一条量化通路，所以短期内所有新方案都是容量方案。

## 3. 对照我们的栈：哪些接得进来

好消息是 2bit 三件套的积木我们**全都有**：

| 学术方案要什么 | 我们手里有什么 | 差距 |
|---|---|---|
| 全精度兜底（sink / 近期窗口） | KIVI 残差窗口，已验证 | 无 |
| 旋转（Hadamard） | 宿主 `single_rht` 参考实现（int4_per_token_head 在用） | 没接进 KIVI 的 flush 和内核 |
| 非均匀编码 | 宿主 TurboQuant 有完整 Lloyd-Max 实现 + 纯 triton 内核 | 没搬到 NPU；上游 #15198/#15821 在等 NPU 实现 |
| FP8 存储 | 本周写好的 fp8_per_token_head 写入内核 | 读路径 |
| GEAR 的"误差已知" | KIVI 写历史区时 min/max 本来就算了 | 稀疏修正逻辑没写 |

接不进来的：NVFP4 / FP8 attention（这代 910B2 没有 FP8 注意力能力，
#8102；等 950 代）；需要权重或校准集的离线方案（KVTuner、KVmix、
Palu 的 SVD——和我们"不读 checkpoint、免校准"的定位冲突）；
格码和熵编码（解码串行、依赖数据内容，服务化不现实）。

## 4. 接下来值得做的（按这个顺序）

**第 0 步：把两个能用的方案补完（基线，先做）**
✅ 端到端 serve 已跑通（fp16/int8/kivi_int4 三组，稠密模型与
Qwen3.5-35B 都通；数据见 [benchmark.md](benchmark.md)，过程见
[serving-verification-20261003.md](serving-verification-20261003.md)）。
剩两件：Qwen3.5 上的真实数字（卡在宿主树落后两拍，见该文档 §4）、
模型级精度评测。跑长上下文评测前先确认绕开 FIA 掩码 bug #15503。

**第 1 步：fp8_per_token_head 补完**
写读路径（注意力带 per-token-head scale 读缓存）→ 910B2 上机比对
（先回答 triton-ascend 支不支持往 fp8 指针写）→ 接命令行。
顺手开 `int8_per_token_head` 变体（同一内核，几乎免费）。

**第 2 步：KIVI + 旋转融合（通往 2bit 的路）**
学 SAW-INT4 的块对角 Hadamard，融进 pack 内核，Q 侧做配套旋转。
**验收标准很硬：旋转必须融进内核，融不进去就不合并**——不融合
等于白做多一次变换，带宽反而亏。

**第 3 步：TurboQuant 搬到 NPU（对齐上游）**
去 #15198/#15821 下认领，避免做出回不了上游的私有实现。
收益按容量记（3.7x@4bit 档），不按带宽。

**第 4 步：GEAR 式稀疏修正**
只做离群值稀疏修正，不做低秩矩阵乘（读路径不多跑流量）。
KIVI flush 时误差就在手上，这是最便宜的精度加法。

**顺手项**：接上宿主的 `kv_cache_dtype_skip_layers`（某些层不量化），
一行配置的精度保险。

## 5. 明确不做的

- NVFP4 / FP8 attention / MXFP8、HiF8：等 950 代硬件能力。
- 格码（QTIP 系）、熵编码：不可服务化。
- 需要权重 / 校准集 / 梯度的离线搜索方案：和插件定位冲突。
- 基于注意力分数的显著性方案（ZipCache 类）：我们走 fused
  attention 算子，拿不到分数。

## 6. 数字声明

本文和关联文档里的外部数字（精度、吞吐、压缩比）全部来自论文或
厂商口径，我们自己没复测过——对外引用前必须先跑自己的 benchmark
（第 0 步完成后就有了）。每个数字的出处和核实状态见调研文档 §12。
