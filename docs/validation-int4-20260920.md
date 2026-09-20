# INT4 (KIVI) 910B2 验证记录 — 2026-09-20

设备验证容器：`vllm-hust-cyj-21rc-cloud-container-86`。代码为 `feat/int4`
分支 `db9517c` 的干净工作树（`/root/qkv-verify-f684231`，由
`git worktree add --detach` 创建，`git status --short` 为空）。设备固定在
空闲卡上运行：`ASCEND_RT_VISIBLE_DEVICES=4`。

| 项目 | 值 |
|---|---|
| npu-smi | 26.0.rc1 |
| 设备 | 8 × Ascend 910B2（Health OK） |
| python / torch | 3.11.15 / 2.10.0+cpu + torch_npu |
| triton（triton-ascend） | 3.5.0 |
| 宿主 checkout | vllm-hust `f18cf803c5`、vllm-ascend-hust `17ed0571d` |

## 1. CPU 侧（同一台机器、同一份代码）

```text
PYTHONPATH=src python -m pytest -q        -> 84 passed
python scripts/check_int4_patch_parity.py -> PASS（补丁不变量全部在位）
python -m ruff check . / ruff format --check . -> All checks passed / 已格式化
```

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

## 7. 真宿主分派核对（`scripts/probe_host_dispatch.py`，本轮脚本化）

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

## 8. 仍未完成

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

  分派本身已在第 7 节用真宿主类验通（脚本里用 `object.__setattr__` 绕过该
  Literal），剩下的就是 `docs/int4-host-integration.md` 那四处宿主改动。
  另外，插件 README/HOST_CONTRACT 锁定的基线 `8a6655cf62` 在这两个宿主
  checkout 的历史里都不存在（`git cat-file -t` 均报 not a valid object），
  所以接合时要以目标基线复核行号。
- **模型级精度**：设备侧目前只有第 6 节那种随机 K/V 下的量化误差口径
  （≤0.068 倍 K/V rms、余弦 ≥0.990）；真实权重下的输出质量、perplexity
  对比仍要在端到端跑通后测。
- 多卡 / context parallel 未覆盖（插件侧一律 fail-closed）。
