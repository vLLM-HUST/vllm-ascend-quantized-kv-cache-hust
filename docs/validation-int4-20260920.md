# INT4 (KIVI) 910B2 验证记录 — 2026-09-20

设备验证容器：`vllm-hust-cyj-21rc-cloud-container-86`。代码为该容器上
`feat/int4` 分支 `fc1d497` 的干净克隆（`/root/qkv-int4-test`）。

| 项目 | 值 |
|---|---|
| npu-smi | 26.0.rc1 |
| 设备 | 8 × Ascend 910B2（Health OK） |
| python / torch | 3.11.15 / 2.10.0+cpu + torch_npu |
| triton（triton-ascend） | 3.5.0 |
| 宿主 checkout | vllm-hust `f18cf803c5`、vllm-ascend-hust `17ed0571d` |

## 1. CPU 侧（同一台机器、同一份代码）

```text
PYTHONPATH=src python -m pytest -q        -> 82 passed
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

## 4. 注意力通路（`scripts/npu_probe_kivi_attention.py`，本轮新增）

与小尺寸几何（head 64 / kv 2 / group 32 / block 32 / residual 32）和**出厂
默认几何**（head 128 / kv 8 / group 128 / block 128 / residual 128）各跑一次，
两者 `rc=0 RESULT: PASS`：

```text
causal mask: (2048, 2048) torch.int8 from vllm_ascend.AttentionMaskBuilder
prefill: 136 tokens, history bytes written=9256229, finite=True
decode: gathered (137, 8, 128), max|diff| vs direct FIA = 0.000000
decode: gathered keys vs int4 reference max|diff| = 0.000000
```

参考实现是"对同一份 gather 结果直接调用 `npu_fused_infer_attention_score`"，
所以差异只可能来自插件的参数拼装（layout / seq 长度 / 切片 / 输出写回），实测
为 0。

顺带测得的宿主契约：因果 prefill 必须带 `AttentionMaskBuilder` 给的
`int8 [2048, 2048]` split-fuse 掩码 —— 传 `T×T` 加性掩码或不传掩码时 aclnn
直接拒绝（`error code 561002`，`When attnMask is not provided, sparseMode must
be 0` / `maskDim 2 shall be 2048`）。插件按 `attn_metadata.attn_mask` 原样透传，
与 INT8 路径同一口径。

## 5. 实验性融合 gather（`scripts/npu_probe_kivi_dim.py`）

```text
[head=32 tile=16]  unstored=0 wrong-stored=4096 total=4096
[head=32 tile=32]  unstored=0 wrong-stored=4095 total=4096
[head=128 tile=16] unstored=0 wrong-stored=16384 total=16384
[head=128 tile=32] unstored=0 wrong-stored=16381 total=16384
```

`ops/triton/kivi_gather_experimental.py` 在 triton-ascend 3.5 上仍然误编译
（读出垃圾值，tile=32 时输出干脆是 0），继续**不路由**；该脚本留作日后重验。

## 6. 仍未完成

- **端到端 `vllm serve --kv-cache-dtype kivi_int4`**：该容器上的宿主
  `CacheDType` 既无 `kivi_int4` 也无 `int8`（且插件锁定的宿主基线
  `8a6655cf62` 不在该 checkout 历史里），所以整链路必须等
  `docs/int4-host-integration.md` 列出的宿主改动落地后复跑。
- 精度（模型输出质量）与多卡 / context parallel 未覆盖。
