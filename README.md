# Quantized KV Cache for vLLM-HUST

Pluggable quantized KV-cache **method** library for the vLLM-HUST stack. The
package mines the provenance-preserved legacy implementations (INT8 dynamic
per-channel, KIVI INT4, and the int4 / fp4_e2m1 / fp8_e4m3 / nvfp4 packed
handlers) into **independent, self-registering method modules** behind one
unified API, with host adapters for both **vllm-hust** and
**vllm-ascend-hust**.

This project is distinct from offline model quantization and the Adaptive
Quantized KV observer project. Technical ownership: @hustcui,
@SuccinctPaul. See [MAINTAINERS.md](MAINTAINERS.md) and
[PROVENANCE.md](PROVENANCE.md).

**Device scope: the kernel layer (triton-ascend / torch_npu) executes on
Ascend NPU only. The pure semantics layer runs anywhere; every device path
fails closed with an explicit error off-NPU.**

## Install

```bash
pip install vllm-ascend-quantized-kv-cache
```

Zero runtime dependencies. Installing never changes any vLLM behaviour.

## Unified API

```python
from vllm_ascend_quantized_kv_cache import kv_methods

kv_methods.list()                          # every registered method
kv_methods.list(host="vllm_ascend_hust")  # filtered by host

method = kv_methods.get("kivi_int4", head_size=128, block_size=128)
method.descriptor        # name, quant mode, provenance, support matrix
method.resolve_layout()  # -> KVCacheLayout (dtype / packed dim / storage)
method.semantics         # pure math: scales, pack/unpack, window bookkeeping

# Plug into a host (explicit; host imports are lazy)
ad = method.host_adapter("vllm_ascend_hust")
info = ad.register()  # registers the scheme into the host registry
```

Unknown names, invalid configuration, and unsupported host/method pairs
raise `ValueError` (fail-closed, mirroring the layout contract).

## Solutions

| method | dtype | source patch | semantics | Ascend NPU kernels | vllm-ascend-hust | vllm-hust |
|---|---|---|---|---|---|---|
| `int8_dynamic` | `int8_per_token_head` | ascend#116/0001 | pure | torch_npu fused attention | scheme + impl surgery | CUSTOM backend (NPU only) |
| `kivi_int4` | `kivi_int4` | ascend#116/0003-0013 | pure | triton-ascend pack / gather | scheme + impl surgery | CUSTOM backend (NPU only) |
| `int4_packed` | `int4` | ascend#160 | pure | backend-side | scheme | CUSTOM backend |
| `fp4_e2m1` | `fp4_e2m1` | ascend#160 | pure | backend-side | scheme | no dtype literal yet |
| `fp8_e4m3` | `fp8_e4m3` | ascend#160 | pure | backend-side | scheme | CUSTOM backend |
| `nvfp4` | `nvfp4` | ascend#160 | pure | backend-side | scheme | CUSTOM backend |

Maturity notes: the **semantics layer** of every method is tested on CPU.
Device execution is validated only to port fidelity; end-to-end serving
needs an Ascend NPU environment. The host-registration link
(`kv_methods.activate` → host `register_scheme`) has been verified
in-process on a real vllm-ascend-hust 910B2 container (2026-09-11, six
methods registered, idempotent re-activation OK); the full engine path
remains a host-integration roadmap item.

## Documentation

| doc | contents |
|---|---|
| [docs/index.md](docs/index.md) | 文档导航 + 30 秒了解本项目 |
| [docs/schemes.md](docs/schemes.md) | 模块作用与含义；六个量化方案的语义、布局、差异与选型 |
| [docs/how-to-run.md](docs/how-to-run.md) | **How to run**：安装、CPU 测试、Python API、NPU 冒烟、宿主 serving、故障排查 |
| [docs/acceptance-matrix.md](docs/acceptance-matrix.md) | 验收与证据矩阵：推广门、方法状态、负向门 |
| [docs/validation-int8-20260912.md](docs/validation-int8-20260912.md) | int8 真机验证记录（container-86） |
| [docs/gap-analysis-vs-ascend-llm-quant.md](docs/gap-analysis-vs-ascend-llm-quant.md) | 对照 Ascend-LLM-quant 的差距分析与下一步 |
| [docs/release-checklist.md](docs/release-checklist.md) | 发布清单 |
| [docs/adr/](docs/adr/) | 架构决策记录 |
| [docs/integration.md](docs/integration.md) | 怎么集成进 vllm-hust / vllm-ascend-hust；Extension Manager 路线 |
| [docs/layers.md](docs/layers.md) | **调用层次与宿主可见性**：vllm-hust / vllm-ascend-hust 各自能调什么、两条激活链路 |
| [docs/npu-implementation.md](docs/npu-implementation.md) | NPU 实现要点：内核路由、fail-closed、C8 类手术、残差窗口、已知问题 |
| [docs/development.md](docs/development.md) | 开发指南：分层纪律、新增方案、测试策略、CI |
| [docs/architecture.md](docs/architecture.md) | 分层架构设计（英文） |
| [docs/packaging-and-release.md](docs/packaging-and-release.md) | 打包与发布流程（英文） |
| [HOST_CONTRACT.md](HOST_CONTRACT.md) | 宿主协议提案与中间态挂载路线 |

## Using from a host (pluggable activation)

Installation is not activation. Two mechanisms exist:

**1. Native vLLM bootstrap (works today, explicit per process):**

```bash
# vllm-ascend-hust or vllm-hust serving process
VLLM_HUST_KV_METHODS=int8_dynamic,kivi_int4 vllm serve MODEL ...
```

The `vllm.general_plugins` hook stays a no-op unless this variable names
methods. Each named method registers into the host that is importable
in that process (unknown names / missing hosts fail closed). On
vllm-hust, start the engine with `--attention-backend CUSTOM` and the
dtype literal negotiated by the adapter (e.g. `int4_per_token_head` for
`kivi_int4`).

**2. Extension Manager (blocked by design):** the bundle
`org.vllm-hust.quantized-kv-cache` ships a Manifest 0.2 descriptor and is
`import_only` — the manager can inspect but must refuse enablement until
the [HOST_CONTRACT.md](HOST_CONTRACT.md) protocols land in the host. See
`docs/architecture.md` for the flip conditions.

**Operator tools** (stdlib-only, no heavy imports):

```bash
vllm-hust-kv-doctor                                   # environment/readiness diagnosis
vllm-hust-kv-inject <model_dir> --method int8_dynamic # inject dispatch config (auto-backup)
vllm-hust-kv-inject <model_dir> --check               # pre-serve contract check (SHA-256 bound)
vllm-hust-kv-inject <model_dir> --restore             # roll back
vllm-hust-kv-evidence validate --file <record.json>   # evidence record validation
python scripts/verify_host_sources.py \
    --vllm-ascend-src <host-checkout>                 # static host-surface verification
```

## Extension framework

```bash
python -m pip install "vllm-hust-ext @ git+https://github.com/vLLM-HUST/extension-manager.git@main"
python -m pip install -e ".[test]"
vllm-hust-ext extension inspect org.vllm-hust.quantized-kv-cache
pytest -q
```

## Packaging and release

Follows the vLLM-HUST packaging and release guide (bidkv reference):
single version source (`_version.py`), manifest inside the wheel, wheel
content verification, isolated smoke install, and tag-triggered PyPI
publishing. See [docs/packaging-and-release.md](docs/packaging-and-release.md).
