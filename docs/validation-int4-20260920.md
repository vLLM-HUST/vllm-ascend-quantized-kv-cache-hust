# INT4 (KIVI) 910B2 验证记录 — 2026-09-20

设备验证容器：`vllm-hust-cyj-21rc-cloud-container-86`。代码为 `feat/int4`
分支的干净工作树（`/root/qkv-verify-f684231`，由 `git worktree add --detach`
创建，当前 detached 在 `6146ab5`，`git status --short` 为空）。设备固定在空闲卡
上运行：`ASCEND_RT_VISIBLE_DEVICES=4`。逐节的 revision 归属见下文说明。

设备套件（第 1~8 节）在 `0c61df8` 上完整跑过一遍并全部通过；`6a1dc4b` 只改插件
实现（宿主导入顺序），故第 11 节分派核对、第 12 节安装态核对与 87 个 CPU 测试在
`6a1dc4b` 重跑通过。第 9 节的多请求 chunked 批次在 `a9dd050` 的两种几何上跑通。

最终记录修订为 `6146ab5`（相比 `6a1dc4b` 只多了 CPU 测试与文档，`src` 逐字节
相同，`git diff 6a1dc4b..6146ab5 -- src` 为空）。在该 revision 的干净工作树上，
除第 5 节（实验性融合 gather，结论未变、仍待宿主修复误编译）之外，把本文档
**每一节的设备命令按原文参数重跑了一遍**，输出与记录逐位一致：第 1 节 90 个
CPU 测试、第 2 节打包/gather 对拍、第 3 节端到端冒烟、第 4 节两种几何与第 8 节
两种 GQA 的单请求注意力（chunked 分支 `max|diff|=0.000000`）、第 6 节批量解码、
第 8 节表格的四行量化误差与兜底差异（0.0638/0.9938、0.0849/0.9950、0.0680/0.9902、
0.1005/0.9905）、第 7 节两种几何的多步生成（玩具几何 67 步零违规；出厂几何 64 步、
131 次一层 tie flip 全部在容差内）、第 9 节一步 5 请求的 chunked 批次、第 10 节
13/15 可编译形状、第 11 节真宿主分派、第 12 节安装态（wheel 装到
`/tmp/kivi-probe-6146ab5` 后经 `load_general_plugins()` 加载）。这些探针都在进程
开头 `torch.manual_seed(0)`，所以复现的是**数字本身**，不只是 `RESULT: PASS`。

| 项目 | 值 |
|---|---|
| npu-smi | 26.0.rc1 |
| 设备 | 8 × Ascend 910B2（Health OK） |
| python / torch | 3.11.15 / 2.10.0+cpu + torch_npu |
| triton（triton-ascend） | 3.5.0 |
| 宿主 checkout | vllm-hust `f18cf803c5`、vllm-ascend-hust `17ed0571d` |

## 1. CPU 侧（同一台机器、同一份代码）

```text
PYTHONPATH=src python -m pytest -q        -> 90 passed
python scripts/check_int4_patch_parity.py -> PASS（补丁不变量全部在位）
python -m ruff check . / ruff format --check . -> All checks passed / 已格式化
```

87→90 的增量是 `test_host_published_numbers_rebuild_the_plugin_layout`
（3 个参数化）：它把 `docs/int4-host-integration.md` 第 1 节向宿主索取的三个数字
（形状里的 `S`、存储 dtype `torch.uint8`、`real_page_size_bytes`）变成可执行契约，
口径见该文档第 2 节。

## 2. 打包与 gather 内核（`scripts/npu_probe_kivi_key.py`）

```text
pack outputs finite: quant True scale True mn True
scale layout check: packed [0.09955404698848724, 0.10997648537158966]
                     ref    [0.09955404698848724, 0.10997648537158966]
mn    layout check: packed [-1.1991091966629028, -1.0654696226119995]
                     ref    [-1.1991091966629028, -1.0654696226119995]
manual vs fake_quant: max|diff| = 4.76837158203125e-07
kernel output: 0 NaNs of 4096 elements
```

结论：键侧 int4 打包内核写出的 word/scale/min 与 CPU 参考在 fp16 舍入内一致，
无 NaN。

## 3. 端到端写入 + gather（`scripts/npu_smoke_kivi.py`）

```text
== pack key cache == / == pack value cache == / == dequant gather ==
value max|diff|=0.000000 -> value path: EXACT match vs fake_quant reference
key vs fake_quant: max|diff|=0.000000
key vs manual-unpack: max|diff|=0.000000
gather kernel MATCHES the packed cache
RESULT: PASS
```

结论：路由路径（triton 打包 + 纯 torch dequant-gather）在 910B2 上逐位复现
语义参考。

## 4. 注意力通路（`scripts/npu_probe_kivi_attention.py`）

同一脚本跑三种元数据形态：纯 prefill、纯 decode、**ChunkedPrefill**（1 条
decode 行 + 1 条全新增 prompt 行，走合并分支）。玩具几何（head 64 / kv 2 /
group 32 / block 32 / residual 32）与**出厂默认几何**（head 128 / kv 8 /
group 128 / block 128 / residual 128）各一遍，均 `RESULT: PASS`：

```text
### geometry A (small)
causal mask: (2048, 2048) torch.int8 from vllm_ascend.AttentionMaskBuilder
prefill: 40 tokens, history bytes written=374362, finite=True
decode: gathered (41, 2, 64), max|diff| vs direct FIA = 0.000000
decode: gathered keys vs int4 reference max|diff| = 0.000000
geometry: head=64 kv_heads=2 group=32 block=32 residual=32 tokens=40
chunked: rows=33 (1 decode + 32 prompt), decode max|diff|=0.000000, prefill max|diff|=0.000000
RESULT: PASS

### geometry B (shipped 128/8/128/128/128)
causal mask: (2048, 2048) torch.int8 from vllm_ascend.AttentionMaskBuilder
prefill: 136 tokens, history bytes written=9256229, finite=True
decode: gathered (137, 8, 128), max|diff| vs direct FIA = 0.000000
decode: gathered keys vs int4 reference max|diff| = 0.000000
geometry: head=128 kv_heads=8 group=128 block=128 residual=128 tokens=136
chunked: rows=129 (1 decode + 128 prompt), decode max|diff|=0.000000, prefill max|diff|=0.000000
RESULT: PASS
```

chunked 的两半分别对照：decode 行 → 直接对插件自己 `_gather_dequant_kivi_paged_cache`
的结果调 FIA；prompt 行 → 直接对该批 K/V 调 FIA（`sparse_mode=3` + 宿主掩码）。
参考实现都建立在同一份 gather 结果上，所以差异只可能来自插件的参数拼装
（layout / seq 长度 / 切片 / 输出写回位置），实测为 0。

变异验证（在设备工作树上临时改代码，跑完还原）：合并分支的输出写回
`attention_backend.py:1338` 由 `output[num_decode:num_tokens]` 改成
`output[:n_prefill]`——纯 prefill 时 `num_decode == 0`，两种写法等价，所以旧
的三种形态里只有新增的 chunked 步能抓到它：

```text
chunked: rows=129 (1 decode + 128 prompt), decode max|diff|=4.042267, prefill max|diff|=3.683594
RESULT: FAIL ['chunked decode rows deviate from the operator (4.042266845703125)',
              'chunked prefill rows deviate from the operator (3.68359375)']
```

还原后 `RESULT: PASS`，工作树 `git status --short` 再次为空。

顺带测得的宿主契约：因果 prefill 必须带 `AttentionMaskBuilder` 给的
`int8 [2048, 2048]` split-fuse 掩码 —— 传 `T×T` 加性掩码或不传掩码时 aclnn
直接拒绝（`error code 561002`，`When attnMask is not provided, sparseMode must
be 0` / `maskDim 2 shall be 2048`）。插件按 `attn_metadata.attn_mask` 原样透传，
与 INT8 路径同一口径。

## 5. 实验性融合 gather（`scripts/npu_probe_kivi_dim.py`）

```text
[head=32 tile=16]  unstored=0 wrong-stored=4095 total=4096
[head=32 tile=32]  unstored=0 wrong-stored=4095 total=4096
[head=128 tile=16] unstored=1730 wrong-stored=14654 total=16384
[head=128 tile=32] unstored=0 wrong-stored=16381 total=16384
```

`ops/triton/kivi_gather_experimental.py` 在 triton-ascend 3.5 上仍然误编译（读出
垃圾值，且哪些格子是垃圾每次跑都不同——同一次 db9517c 复跑与 b8083f3 的记录就
差了几格），继续**不路由**；该脚本留作日后重验。

## 6. 批量解码：ragged 历史 + 多 block（`scripts/npu_probe_kivi_batched.py`，本轮新增）

前面的注意力检查每步最多只有一个 decode 请求（chunked 那步是 1 decode + 1
prompt），而 serving 是多个请求一起 decode：历史长度互不相干、各自跨若干 cache
block，批的形状只通过 `actual_seq_lengths_kv` 的**前缀和**告诉 aclnn。CPU 侧测试
对这一点只能"记录参数"（`torch_npu` 是桩），所以多请求批量形态此前从未在硬件上
跑过。该探针让 3~4 个请求分别 prefill 出 1/2/3/4 个已 flush 窗口的历史，再一次性
decode：

```text
geometry: head=128 kv_heads=8 group=128 block=128 residual=128 seqs=4
          histories=[136, 264, 392, 520] blocks=14
batched decode: gathered (1316, 8, 128), kv lens [137, 402, 795, 1316],
                max|diff| vs direct FIA = 0.000000
  req-0: history 137 over 2 blocks, keys vs int4 reference max|diff| = 0.000000
  req-1: history 265 over 3 blocks, keys vs int4 reference max|diff| = 0.000000
  req-2: history 393 over 4 blocks, keys vs int4 reference max|diff| = 0.000000
  req-3: history 521 over 5 blocks, keys vs int4 reference max|diff| = 0.000000
int4 accuracy: worst |diff|/K-V rms = 0.068025, worst cosine = 0.990201
RESULT: PASS
```

玩具几何（head 64 / kv 2 / group 32 / block 32）同样 PASS：
`worst |diff|/K-V rms = 0.063794, worst cosine = 0.993776`。

三个变异确认这一步不是空跑，且**只有批量探针能抓到**（同一变异下单请求探针
`RESULT: PASS`）：

| 变异 | 批量探针结果 |
|---|---|
| `actual_seq_lengths_kv` 不取前缀和（直接用每请求长度） | aclnn 直接拒绝：`error code is 561002` |
| gather 里所有请求都读第 0 行残差窗口 | `FAIL ['req-1 gathered keys deviate (5.484375)', 'req-2 ... (5.4453125)', 'req-3 ... (5.75)', 'int4 attention correlates with fp16 only 0.9638']` |
| 每请求切片不偏移（`dense_k[0:req_len]`） | `FAIL ['req-1 ... (7.609375)', ..., 'int4 attention deviates from fp16 by 0.5063 of K/V rms', 'int4 attention correlates with fp16 only 0.3383']` |

顺带得到第一个**设备侧量化误差**口径：同一份注意力在 int4 历史 vs 全精度 fp16
缓存下，偏差 ≤ 0.068 倍 K/V rms、余弦 ≥ 0.990（随机 K/V、无模型权重；真实
激活上的分布仍待模型级评测）。归一化用 K/V rms 而不是注意力输出幅值——随机
key 下 softmax 接近均匀，输出本身按 `1/sqrt(N)` 缩小，用输出做分母会把任何
量化方案都衬得很难看（首轮就因此误报 0.48）。

## 7. 多步生成：逐步 flush 调度在真机上（`scripts/npu_probe_kivi_generate.py`，本轮新增）

前面每一步都是"写一次、读一次"。serving 是循环：token 一个一个进来，键按整窗口
溢出才落 int4、值按最老槽位逐个 eviction，而且这些动作发生在**真 triton 打包内核**
之下。该探针先给 2 个请求各 prefill 一个整窗 prompt，再连续 decode，**每一步**
都把 gather 出来的 K/V 与"按写入路径规则重算的参考"逐元素对比：

```text
geometry: head=64  kv_heads=2 group=32  block=32  residual=32  prompt=32  steps=67
prefill: history=[32, 32] key diff=0.000000 value diff=0.000000
step 67/67: history=99 residual rows=3/32 key diff=0.000000 value diff=0.000000
retirement: req-0 released; survivor req-1 still exact at history 100 (keys 0.000000, values 0.000000)
final step: req-1 history=[100] gathered (100, 2, 64), max|diff| vs direct FIA = 0.000000
generation: 67 steps ... one-level tie flips tolerated: 0
RESULT: PASS

geometry: head=128 kv_heads=8 group=128 block=128 residual=128 prompt=128 steps=64
prefill: history=[128, 128] key diff=0.000000 value diff=0.391602
generation: 64 steps ... one-level tie flips tolerated: 131
RESULT: PASS
```

出厂几何上那 0.3916 不是调度错，而是**量化格点上的四舍五入平局**：实测某元素
（row 24 / head 7 / dim 41，值 0.542969，组 min=-2.394531、step=0.391667）的归一
化值在实数上正好是 7.5，triton 的 fp32 中间结果落在 7.49999973，于是内核存了
7 档、`semantics.quantize_group` 的 `floor(x+0.5)` 给 8 档。该事实已写进
`quantize_group` 的 docstring。

这一步的判据因此**不是**幅度容差：先量测试用 2×半格的容差，结果它宽到连"整段
prompt 没被量化（保持全精度）"都能放过（把 bulk 规则改成 incremental 规则的变异
在玩具几何只报 0.21 的差、被判通过）。改成"逐元素要么与参考同档，要么在**精确
平局**处相差一档（两档到原值等距）"后：

| 变异 | 结果 |
|---|---|
| bulk prefill 改成等溢出才 flush（少 flush 一整窗） | `FAIL ['prefill: req-0 keys off-grid x3453', 'prefill: req-0 values off-grid x3470', ...]`，同一变异下 CPU 侧 `test_bulk_prefill_flushes_a_full_window_that_incremental_writes_keep_exact` 也失败 |
| 增量窗口溢出判定 `>=` 改 `>`（晚一个 token 才 flush） | 插件自己的对齐守卫先炸：`RuntimeError: KIVI key flush requires contiguous aligned token groups.`（错位的窗口不再是整组对齐） |

两条都是**只有多步运行才会出现**的状态：单步探针（第 4/6 节）对它们完全无感。

同一探针还补了两件事：

**误差随上下文长度的趋势**（同一批 K/V 的 fp16 注意力做对照，按 K/V rms 归一）：

```text
玩具几何:  48 -> 0.0443, 64 -> 0.0351, 80 -> 0.0432, 96 -> 0.0324, 99 -> 0.0333
出厂几何: 144 -> 0.0756, 160 -> 0.0658, 176 -> 0.0529, 192 -> 0.0533
```

也就是说在这段长度内误差**不随上下文累积**（甚至略降，因为 softmax 把历史摊薄）。

**请求重挂载**：serving 里 retire 一个请求后，它的 cache block 和残差行会立刻给新
请求用。探针因此用**被回收请求自己的 block** 重新准入一个 `residual + group/2`
的新请求（窗口不满，必须拿到行），再逐项对拍：

```text
re-admission: req-new (history 192) holds residual row 0, retired req-0 holds None;
              violations k=0 v=0, tie flips=1, exact tails 0.000000/0.000000
```

把"释放 finished 行"短路掉，这一步立刻炸：
`RuntimeError: KIVI residual rows are exhausted. max_num_seqs=2, req_id=req-new.`
但把 `_sync_kivi_residual_windows` 里的"清理不再属于本请求的槽位"短路掉，**设备
上看不出来**——释放路径已经把整行擦干净了，这一步只是第二道防线。检查发现这条
剪枝路径此前 CPU/设备两侧都没有任何测试覆盖，于是补了
`test_sync_prunes_residual_slots_the_request_no_longer_owns`（把请求回滚两个
token，断言残差窗口只剩两个槽位）；该测试是唯一能杀掉这个变异的检查。

## 8. 分组查询注意力（GQA）与 torch 兜底路径（本轮新增）

两个注意力探针此前把 query 也按 `num_kv_heads` 造张量，也就是只跑过 MHA；
真实模型都是 GQA（每个 kv 头带多个 q 头）。加 `KIVI_PROBE_GQA=n` 后，
`KIVI_PROBE_GQA=7` 给出 Llama 口径的 14Q/2KV（玩具几何）与 56Q/8KV（出厂
几何），四种组合（单请求/批量 × 玩具/出厂）全部 `RESULT: PASS`。单请求那两条：

```bash
KIVI_PROBE_GQA=7 python scripts/npu_probe_kivi_attention.py
KIVI_PROBE_HEAD=128 KIVI_PROBE_KV_HEADS=8 KIVI_PROBE_GROUP=128 \
KIVI_PROBE_BLOCK=128 KIVI_PROBE_RESIDUAL=128 KIVI_PROBE_GQA=7 \
  python scripts/npu_probe_kivi_attention.py
```

这条覆盖不是象征性的——把 decode 分支里的 `num_key_value_heads=self.num_kv_heads`
改成 `self.num_heads`（等价于告诉算子"每个 kv 头只服务一个 q 头"）：

```text
-- MHA (GQA=1), batched:   RESULT: PASS      ← 旧探针完全看不见
-- GQA=7, batched:  RESULT: FAIL ['batched decode output has NaN/Inf',
     'int4 attention deviates from fp16 by 27.5055 of K/V rms',
     'int4 attention correlates with fp16 only 0.0228']
-- GQA=7, 单请求探针: RESULT: FAIL ['decode attention deviates from the
     operator (0.95721435546875)', 'chunked decode rows deviate from the
     operator (1004.4728393554688)']
```

同一轮把**纯 torch 兜底注意力**（宿主给出未知 attn_state 时走的逐请求
softmax 分支）也放到设备上跑：它自己用 `_repeat_kv` 扩 q 头、自己拼因果掩码，
和 aclnn 是两套独立实现，四组配置下两者相差不超过 aclnn 输出 rms 的 0.008（逐行
数字见本节末尾表格最后一列）。把 `_repeat_kv` 改成永不扩展后，GQA 那步直接崩在
`The size of tensor a (14) must match the size of tensor b (2)`，MHA 依旧
无感——再次说明只有 GQA 形状能验到这条路径的要点。

量化误差口径随头布局的变化（同一批量探针，`int4` 历史 vs 全精度 fp16）。探针内部
`torch.manual_seed(0)`，所以这四行是**可复现的确定值**而不是抽样；每条的完整命令：

```bash
KIVI_PROBE_SEQS=3 python scripts/npu_probe_kivi_batched.py                        # 玩具 MHA
KIVI_PROBE_SEQS=3 KIVI_PROBE_GQA=7 python scripts/npu_probe_kivi_batched.py       # 玩具 GQA
KIVI_PROBE_HEAD=128 KIVI_PROBE_KV_HEADS=8 KIVI_PROBE_GROUP=128 \
KIVI_PROBE_BLOCK=128 KIVI_PROBE_RESIDUAL=128 KIVI_PROBE_SEQS=4 \
  python scripts/npu_probe_kivi_batched.py                                        # 出厂 MHA
KIVI_PROBE_HEAD=128 KIVI_PROBE_KV_HEADS=8 KIVI_PROBE_GROUP=128 \
KIVI_PROBE_BLOCK=128 KIVI_PROBE_RESIDUAL=128 KIVI_PROBE_SEQS=4 KIVI_PROBE_GQA=7 \
  python scripts/npu_probe_kivi_batched.py                                        # 出厂 GQA
```

| 配置 | worst \|diff\|/K-V rms | worst cosine | torch 兜底 vs aclnn |
|---|---|---|---|
| 玩具几何 MHA | 0.0638 | 0.9938 | 0.0014 |
| 玩具几何 GQA=7 | 0.0849 | 0.9950 | 0.0051 |
| 出厂几何 MHA | 0.0680 | 0.9902 | 0.0072 |
| 出厂几何 GQA=7 | 0.1005 | 0.9905 | 0.0048 |

末列是纯 torch 兜底注意力与 aclnn 结果的差，按 aclnn 输出 rms 归一——四行都在
`0.008` 以下，即两套独立实现在硬件上互证。

## 9. 一条 chunked step 里的多请求批次（`scripts/npu_probe_kivi_chunked_batch.py`，本轮新增）

chunked 分支是全插件索引算术最密的地方：一个 batch 里先排 decode 行、再排若干
prompt 行，每个 prompt 请求的 q 长度要从**累积的** `actual_seq_lengths_q` 里减掉
decode 行数才能还原出来，融合结果再写回一个切片。此前真机上跑过的 chunked 步
全是"1 decode + 1 prompt"，在这种形状下这套算术退化成了 no-op，错了也看不出来。

现在一步带 2 个 decode（历史 41/73，出厂几何 137/265）+ 3 个长度互不相干的
prompt（40/72/104，出厂 136/264/392）：

```text
chunked batch: rows=218 (2 decode + [40, 72, 104]), qlen=[1, 2, 42, 114, 218], seq=[41, 73, 40, 72, 104]
  dec-0: decode row over 41 tokens, max|diff| = 0.000000
  pre-0: prompt row of 40 tokens, max|diff| = 0.000000
  pre-2: prompt row of 104 tokens, max|diff| = 0.000000
  dec-0: cache 41 tokens, quantised 32/32, violations k=0 v=0
  pre-2: cache 104 tokens, quantised 96/96, violations k=0 v=0
chunked batch: 5 requests in one step, one-level tie flips: 0
RESULT: PASS                     # 出厂几何同样 PASS（tie flips: 3）
```

变异对照（两个变异下旧的单请求 chunked 步都 `RESULT: PASS`）：

| 变异 | 多请求批次探针 |
|---|---|
| prompt 长度循环只取第一个请求 | aclnn 直接拒绝批：`FusedInferAttentionScore do tiling failed, ret is -1.` |
| 写路径忘了减 decode 行数 | 插件自己的对齐守卫先炸：`RuntimeError: KIVI key flush requires contiguous aligned token groups.`（第一个 prompt 的窗口长度变成 41，不再是整组） |

顺带把"逐档比较（含平局一档）"的实现抽成 `scripts/kivi_probe_reference.py`，
生成探针与本探针共用，不再各写一份规则。

## 10. 几何包络：block_size ≠ residual_length 与内核编不过的形状（本轮新增）

前面所有设备跑法都是 `block_size == residual_length`——那是窗口/块算术唯一不可能
分歧的组合。把两值错开后（`block=32/residual=64`、`block=64/residual=32`、出厂
`block=128/residual=256`），单请求注意力、批量 ragged、多步生成三种探针**全部
`RESULT: PASS`**；顺带暴露注意力探针自己把 block table 硬编码成两列，已在
`6426f37` 改成按几何推导。

更重要的是发现**通过 `validate_kivi_geometry` 的形状不等于跑得起来的形状**。
`scripts/npu_probe_kivi_geometry.py` 直接调真 triton 打包内核扫了 15 个合法形状：

```text
 head  group  block  blk*head  result
   64     32    512     32768  ok
  128     32    512     65536  ok
  128     64    256     32768  ok
  128    128    128     16384  ok          ← 出厂默认
  128    128    256     32768  FAIL ub needs 201KiB > 192KiB
  256    128    128     32768  ok
  256    128    256     65536  FAIL ub needs 201KiB > 192KiB
   64     64    512     32768  ok
  256     64    256     65536  ok
RESULT: 13/15 shapes compile
```

两个失败都是 bishengir 的 UB（片上统一缓冲）溢出，报错位置
`ops/triton/kivi_pack.py:135`：`ub overflow, requires 1655040 bits while
1572864 bits available`。它们的共同点是 **group_size=128 且 block_size=256**；
而 `group=64, block=512`、`head=256, block=256` 都能过——所以我先前提的候选规则
`group_size * block_size <= 16384` 被 `(64, 64, 512)` 直接证伪，这里只留测量结果、
不写公式。结论对使用的影响：**出厂默认 128/128/128 在可编译包络内**，但宿主若把
`block_size` 配成 256 且沿用 128 的量化组，内核会在运行期编译失败。

## 11. 真宿主分派核对（`scripts/probe_host_dispatch.py`，本轮脚本化）

此前这一节是一次性手工脚本的产物，改成本仓脚本后立刻暴露出一个被它掩盖的
宿主漂移：那份手工代码自己 `enable_cp = lambda: False`，而该 revision 的宿主
根本没有 `enable_cp`。去掉这个桩、按 `f684231`（修复前）重跑，直接崩在插件的
CP 探测上：

```text
host CP helpers: enable_dcp, enable_pcp
  File ".../adapters/vllm_ascend_hust/backend.py", line 104, in get_impl_cls
ImportError: cannot import name 'enable_cp' from 'vllm_ascend.attention.utils'
             (/root/vllm/vllm-ascend-hust/vllm_ascend/attention/utils.py)
```

即：在这台容器的宿主上，任何量化 dtype 的分派都会先 ImportError（INT8 同样
中招，它是 `60cd123` 一起带进来的）。修复见 `fb046ec`：`enable_cp` 优先、否则
用 `enable_dcp()`/`enable_pcp()` 的并集、两者都没有时 fail-closed。修复后在同一
份干净工作树上跑（桩只剩 vllm config 上下文，CP 标志用宿主真实函数 +
`decode_context_parallel_size=2` 造）：

```text
host CP helpers: enable_dcp, enable_pcp
host before: AscendAttentionBackendImpl
auto       -> AscendAttentionBackendImpl     ok
int8       -> AscendInt8KvAttentionImpl      ok
kivi_int4  -> AscendKiviInt4KvAttentionImpl  ok
fp8        -> AscendAttentionBackendImpl (delegated)
fp8_e4m3   -> AscendAttentionBackendImpl (delegated)
float16    -> AscendAttentionBackendImpl (delegated)
CP guard: Ascend KV cache kivi_int4 does not support context parallel yet.
RESULT: PASS
```

结论：插件的 INT4 实现类能直接组合在该宿主 `AscendAttentionBackendImpl` 之上
（mixin 在 MRO 里、宿主 impl 在 MRO 前三位、未量化 dtype 原样委托、真实 CP 配置
fail-closed、重复安装不改选），`int8`/`kivi_int4` 两个 mixin 都在真宿主上验过。

顺带测出两条宿主环境事实，接合时会遇到：

- `vllm_ascend.attention.attention_v1` 不能先于 `vllm_ascend.ops` 被导入，否则
  触发该 revision 的循环导入（`ImportError: cannot import name 'DeviceOperator'
  from partially initialized module 'vllm_ascend.device.device_op'`）。真实
  serve 里平台插件先加载，因此插件的惰性导入没问题；但独立脚本必须
  `import vllm_ascend.ops` 先。
- `enable_dcp` 带 `@lru_cache(maxsize=1)`（`vllm_ascend/attention/utils.py:238`），
  第一次调用的结果会被永久缓存；脚本要换 CP 配置必须 `cache_clear()`。真实
  serve 里 config 固定，所以不影响。

## 12. 安装态核对（`scripts/probe_installed_plugin.py`，本轮新增）

前面所有检查都在源码树上跑（`PYTHONPATH=src`）。但 vLLM 真正使用这个插件的
方式是：**安装的发行包 + `vllm.general_plugins` entry point**。把 wheel 装到
源码树之外、只通过 vLLM 自己的加载器跑一遍，当场暴露一个会拦住启动的缺陷：

```text
entry point: vllm_ascend_quantized_kv_cache.bootstrap:register_plugins
  File "/tmp/kivi-probe/.../adapters/vllm_ascend_hust/register.py", line 43, in register
    host_backend = install_kv_impl_dispatch()
  File "/root/vllm/vllm-ascend-hust/vllm_ascend/ops/fused_moe/moe_mlp.py", line 22, in <module>
ImportError: cannot import name 'DeviceOperator' from partially initialized module
             'vllm_ascend.device.device_op' (most likely due to a circular import)
```

原因：插件的注册在"宿主 attention 栈还没被碰过"的进程里第一次 import
`vllm_ascend.attention.attention_v1`，而该 revision 的这个模块会经
`device_op` 反向 import `vllm_ascend.ops`，只有自己先被 import 过才不炸。
第 9 节的手工/脚本检查都先 `import vllm_ascend.ops`，所以把这个坑遮住了——
又是"探针替被测对象做了准备工作"的典型（见 `PROVENANCE.md` 之外的教训）。
**INT8 也一样中招**（`60cd123` 起就是这个写法），所以这个修复对已发布的
INT8 路径同样是必需的。

修好后（注册改走 `_host_attention_v1()`：先 `import vllm_ascend.ops`，再用
`importlib.import_module` 取模块，并由 CPU 测试盯住这个顺序）：

```text
entry point: vllm_ascend_quantized_kv_cache.bootstrap:register_plugins  (dist vllm-ascend-quantized-kv-cache)
[vllm-ascend-quantized-kv] registered quantized KV attention backends; enable one of them with --kv-cache-dtype int8_dynamic, kivi_int4
[vllm-hust-knorm] runtime patches registered; activate with VLLM_KNORM_ENABLED=1 ...
package loaded from: /tmp/kivi-probe/vllm_ascend_quantized_kv_cache/__init__.py
auto       -> AscendAttentionBackendImpl     ok
int8       -> AscendInt8KvAttentionImpl      ok
kivi_int4  -> AscendKiviInt4KvAttentionImpl  ok
RESULT: PASS
```

顺带确认两件事：插件与这台机器上另一个 vLLM 插件（`vllm-hust-knorm`）在同一次
`load_general_plugins()` 里和平共存；注册的包确实来自安装目录而不是仓库
`src`（脚本会检查并拒绝源码树导入）。复现：

```bash
python -m build --wheel
python -m pip install --target /tmp/kivi-probe --no-deps dist/*.whl
PYTHONPATH=/tmp/kivi-probe python scripts/probe_installed_plugin.py
```

## 13. 仍未完成

- **端到端 `vllm serve --kv-cache-dtype kivi_int4`**：该容器宿主
  （vllm-hust `f18cf803c5`）的 `CacheDType` 是 pydantic 校验过的
  `Literal`（`vllm/config/cache.py:39`，18 个取值），既不含 `kivi_int4`
  也不含 `int8`，所以 CLI 字面量在构造 `CacheConfig` 时就被拒：

  ```text
  pydantic_core._pydantic_core.ValidationError: 1 validation error for CacheConfig
  cache_dtype
    Input should be 'auto', 'float16', ..., 'int4_per_token_head',
    'int8_per_token_head', 'fp8_per_token_head', 'nvfp4' or 'nvfp4_4over6'
    [type=literal_error, input_value='int8', input_type=str]
  ```

  分派本身已在第 11 节用真宿主类验通（脚本里用 `object.__setattr__` 绕过该
  Literal），剩下的就是 `docs/int4-host-integration.md` 那四处宿主改动。
  另外，插件 README/HOST_CONTRACT 锁定的基线 `8a6655cf62` 在这两个宿主
  checkout 的历史里都不存在（`git cat-file -t` 均报 not a valid object），
  所以接合时要以目标基线复核行号。
- **模型级精度**：设备侧目前只有第 6 节那种随机 K/V 下的量化误差口径
  （≤0.068 倍 K/V rms、余弦 ≥0.990）；真实权重下的输出质量、perplexity
  对比仍要在端到端跑通后测。
- 多卡 / context parallel 未覆盖（插件侧一律 fail-closed）。
