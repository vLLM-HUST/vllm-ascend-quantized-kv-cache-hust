# 验收与证据矩阵（quantized KV cache）

对标离线量化生态的验收纪律（见
[Ascend-LLM-quant 的 acceptance matrix](https://github.com/jxd1111/Ascend-LLM-quant/blob/main/docs/acceptance-matrix.md)）：
**schema 正确 ≠ 硬件验证过**。每个能力声明必须能追溯到对应推广门级的
证据；发布说明只允许声明已达门级的能力。

## 推广门（promotion gates）

| 门 | 定义 | 证据形态 |
|---|---|---|
| `schema_only` | 描述/清单/哈希绑定静态成立 | `vllm-hust-kv-inject --check` 通过 |
| `cpu_semantics` | 语义数学在 CPU 张量上与参考实现一致 | `pytest tests/test_int8_dynamic.py` 等 |
| `host_registration` | scheme 真实注册进宿主注册表（幂等、fail-closed） | 真宿主进程 `kv_methods.activate` 日志 |
| `npu_kernel_bitexact` | 设备内核与 CPU 参考逐位对拍通过 | `scripts/npu_smoke_kivi.py` → `RESULT: PASS` |
| `npu_e2e` | 真实引擎加载 → 分发 → 前向 → 生成 → 关停全链通过 | 带日期的验证记录（docs/validation-*.md） |
| `matched_benchmark` | 与 BF16 同协议对照的 PPL / 吞吐 / TTFT / HBM | schema 化 evidence 记录 + 原始日志 |

**当前总口径（0.2.0.dev0）**：语义层全方法 `cpu_semantics`；
`int8_dynamic` 与 `kivi_int4` 达成 `host_registration`；
`kivi_int4` 另有 `npu_kernel_bitexact`；`int8_dynamic` 的分发与前向在
真实引擎通过（见 `docs/validation-int8-20260912.md`）但 **int8 存储
路径阻塞于宿主侧分配/重排不一致，未达 `npu_e2e`**；无任何方法达到
`matched_benchmark`。

## 方法状态矩阵

| 方法 | schema_only | cpu_semantics | host_registration | npu_kernel_bitexact | npu_e2e | matched_benchmark |
|---|---|---|---|---|---|---|
| `int8_dynamic` | ✅ | ✅ | ✅ (910B2, 2026-09-11) | n/a（torch_npu 融合算子路径） | ⚠️ 分发/类手术/浮点前向 ✅；int8 存储路径根因已定位、修复已备（宿主补丁 + 插件侧守卫 `VLLM_HUST_KV_ALLOC_GUARD`，见 `provenance/host-fixes/README.md`），**待 NPU 复验** | ❌ |
| `kivi_int4` | ✅ | ✅ | ✅ (910B2, 2026-09-11) | ✅ pack+gather 逐位 (910B2) | ❌ | ❌ |
| `int4_packed` | ✅ | ✅ | ✅ | n/a（backend 侧） | ❌ | ❌ |
| `fp4_e2m1` | ✅ | ✅ | ✅（vllm-ascend 侧） | n/a | ❌ | ❌ |
| `fp8_e4m3` | ✅ | ✅ | ✅ | n/a | ❌ | ❌ |
| `nvfp4` | ✅ | ✅ | ✅ | n/a | ❌ | ❌ |

vllm-hust CUSTOM backend 路线：注册与类路径钩子就绪（脚手架），
整链 `npu_e2e` 未验证——不计入上表。

## 负向门（negative gates，必须保持 fail-closed）

每一项都有对应测试钉住，改动激活/注入路径时逐条自检：

| 负向场景 | 期望行为 | 测试 |
|---|---|---|
| 未知方法名 / 注册键 | ValueError，列出已知方法 | `test_checkpoint_inject.py::test_resolve_method_token_fail_closed` |
| 违反几何约束（KIVI group_size 等） | ValueError | `test_kivi_int4.py` |
| checkpoint 带非 ascend 量化配置 | 拒绝注入（weight-quantized） | `test_inject_refuses_weight_quantized_checkpoint` |
| 外部 ModelSlim 描述 | 拒绝覆盖（`--force` 才放行） | `test_inject_refuses_foreign_ascend_description` |
| 注入后 config.json 被改 | `--check` 哈希失配退出 2 | `test_check_detects_post_injection_drift` |
| 缺失/陈旧注入清单 | `--check` 失败 | `test_check_detects_missing_manifest`、`test_restore_removes_manifest` |
| 环境变量点名方法但无宿主栈 | RuntimeError | `test_missing_host_fails_closed` |
| 未知宿主 / 不支持组合 | ValueError | `test_activation.py`、`test_adapters.py` |
| 非 NPU 环境调设备路径 | RuntimeError（`require_npu`） | `test_int8_dynamic.py` 等 |
| dtype 字面量未注册 | `resolve_layout` ValueError | `test_dtypes.py` |
| 宿主注册键真冲突（同名不同类） | 注册抛错，不静默顶替 | `test_adapters.py`（幂等 no-op 仅限等价注册） |

## 证据记录纪律

- 每次真机验证落一份**带日期的验证记录**（`docs/validation-<method>-<yyyymmdd>.md`）：
  冻结输入（模型、宿主 commit/版本、容器）、逐项结果、原始日志位置、
  剩余缺口。历史记录不回写。
- 证据记录落成 JSON 并通过
  `vllm-hust-kv-evidence validate --file <record>`（schema 见
  `contracts/kv-evidence-v1.schema.json`，示例
  `evidence/example-int8-v1.json`）；未知字段拒绝，越级声明
  （`matched_benchmark` 无 BF16 基线）拒绝。
- 声明门级时引用记录，不允许"应该能跑"的口径。
- `matched_benchmark` 额外要求：BF16 与量化 profile 同协议（同数据、
  同 tokenizer、同长度上限、同 warm-up/聚合），记录空闲/负载/峰值 HBM
  与原始日志路径。
