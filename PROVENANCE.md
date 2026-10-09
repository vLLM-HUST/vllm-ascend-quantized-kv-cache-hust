# Provenance

当前发行保留两条量化 KV 路径：动态 INT8 与 KIVI INT4。

| 当前代码 | 来源 | 保留内容 |
|---|---|---|
| `methods/int8_dynamic/attention_backend.py` | vLLM-Ascend-HUST `f4f49832` 中的 `AscendInt8AttentionBackendImpl` | 动态 per-channel scale、对称 INT8 量化、decode/prefill/chunked-prefill、pooling 和 ACL Graph 路径 |
| `methods/int8_dynamic/semantics.py` | legacy vllm-ascend PR #116 patch 0001 | 可脱离 NPU 测试的量化语义 |
| `methods/kivi_int4/semantics.py` | legacy vllm-ascend PR #116 patch 0003-0009 | KIVI 分组量化数学、int32 打包/解包、残差窗口地址簿与对齐判据 |
| `methods/kivi_int4/attention_backend.py` | legacy vllm-ascend PR #116 patch 0003-0009（宿主 KIVI 分支最终状态） | 分页缓存绑定（两张字节缓冲或 6 元组）、残差环形窗口与整组 flush 状态机、gather+反量化与 TND/稠密 FIA 分派 |
| `ops/triton/kivi_pack.py` | legacy vllm-ascend PR #116 patch 0003-0009 | triton-ascend int4 打包内核（910B2 逐位验证过的实现） |
| `ops/kivi_gather.py` / `ops/kivi_layout.py` | 本仓库移植（对照 legacy 打包布局） | 纯 torch 的 dequant-gather、缓存布局与送内核前的 slot 闸门（已从 triton 模块移入，CPU 可测） |
| `methods/kivi_int4/byte_cache.py` | 本仓库新增（非 legacy） | 把宿主按层给的两张字节缓冲切成内核期望的 6 个视图；区域预算与压缩比测算 |
| `adapters/vllm_ascend_hust/` | 本仓库插件化适配 | 宿主 `get_impl_cls` 的量化 dtype 分派（`int8` / `kivi_int4`）与非量化 dtype 委托 |

历史补丁原文继续保存在 `provenance/legacy-patches/`。除上述两条路径外的
legacy 方案（packed INT4、FP4、FP8、NVFP4）不属于本分支的发行代码；
`ops/triton/kivi_gather_experimental.py` 是已验证误编译的融合 gather 内核，
随包保留但**不被路由**（在线路径走 `ops/kivi_gather` 的纯 torch 实现），
triton-ascend 修复后用 `scripts/npu_probe_kivi_dim.py` 重验。

## INT4 移植对账（2026-09-19 逐条测量）

- 补丁范围订正：KIVI 实现内容只在 `ascend-pr-116/0003-0009`。0010/0011/0012
  改的是 torch 版本、spec decode 与 sampler 测试，0014 只删 profiling 产物；
  0013 的 subject 写着 “Update kivi int4 triton ascend integration”，但 diff
  实为 `platform.py` / `ops/__init__.py` 的导入懒化，不触碰 KIVI 代码。
- 算子契约对账（用 `scripts/check_int4_patch_parity.py` 可重跑）：按补丁顺序
  重放 add/remove 事件得到 `tl.` 行的最终状态，64 行中 50 行在移植后的内核里
  逐字命中；其余 14 行是跨行语句片段与被后续提交替换掉的实验性 gather 索引
  （该内核不被路由）。六类承载性不变量全部落地：键槽位对齐判据、
  `group_size % 8` 打包门、8 lane 打进 1 个 int32（`lane * 4` 位移、
  `& 0xF` 解包）、值的 head 维分组、kv 长度的 cumsum、残差窗口整除性。
- legacy 自身有一处不自洽：patch 里同时出现内核的 `tl.maximum((mx-mn)/15, 1e-6)`
  与 torch 侧的 `(mx-mn).clamp(min=1e-6)/15`（下限位置不同，scale 相差 15 倍），
  且一侧用 `torch.round`（half-to-even）而内核用 `floor(x + 0.5)`（half-up）。
  本仓库以**内核为准**：`semantics.group_scale` / `quantize_group` 复刻内核算术，
  由 `test_reference_arithmetic_matches_pack_kernel_contract` 钉住；真机复验若
  出现逐位差异，先查这一契约是否被动过。
- 2026-09-20 真机确实出现了逐位差异，但**不是**契约被改动：出厂几何下 128 行里有
  1 行的某个元素，其归一化值在实数上正好是 7.5（半格点），内核的 fp32 中间结果是
  7.49999973 因而存了 7 档，`floor(x + 0.5)` 参考给 8 档。复刻算式不足以保证平局
  处的逐位一致——两侧浮点顺序不同就会差一档。`scripts/npu_probe_kivi_generate.py`
  因此按"同档，或在精确平局处相邻一档（两档到原值等距）"判定一致，而不是用幅度
  容差（幅度容差宽到能放过"整段没量化"的错误，已实测）。

`attention_backend.py` 保留了上游 Huawei Technologies 版权声明，并继续
使用 Apache-2.0。本仓库根目录的 `LICENSE` 包含完整许可证文本。
