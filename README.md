# Quantized KV Cache for vLLM-HUST

Pluggable quantized KV-cache solution library for the vLLM-HUST stack. The
package mines the provenance-preserved legacy implementations (INT8 dynamic
per-channel, KIVI INT4, and the int4 / fp4_e2m1 / fp8_e4m3 / nvfp4 packed
handlers) into **independent, self-registering solution modules** behind one
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
from vllm_ascend_quantized_kv_cache import kv_solutions

kv_solutions.list()                          # every registered solution
kv_solutions.list(host="vllm_ascend_hust")   # filtered by host support

sol = kv_solutions.get("kivi_int4", head_size=128, block_size=128)
sol.descriptor        # name, quant mode, provenance, support matrix
sol.resolve_layout()  # -> KVCacheLayout (dtype / packed dim / storage)
sol.semantics         # pure math: scales, pack/unpack, window bookkeeping

# Plug into a host (explicit; host imports are lazy)
ad = sol.host_adapter("vllm_ascend_hust")
info = ad.register()  # registers the scheme into the host registry
```

Unknown names, invalid configuration, and unsupported host/solution pairs
raise `ValueError` (fail-closed, mirroring the layout contract).

## Solutions

| solution | dtype | source patch | semantics | Ascend NPU kernels | vllm-ascend-hust | vllm-hust |
|---|---|---|---|---|---|---|
| `int8_dynamic` | `int8_per_token_head` | ascend#116/0001 | pure | torch_npu fused attention | scheme + impl surgery | CUSTOM backend (NPU only) |
| `kivi_int4` | `kivi_int4` | ascend#116/0003-0013 | pure | triton-ascend pack / gather | scheme + impl surgery | CUSTOM backend (NPU only) |
| `int4` | `int4` | ascend#160 | pure | backend-side | scheme | CUSTOM backend |
| `fp4_e2m1` | `fp4_e2m1` | ascend#160 | pure | backend-side | scheme | no dtype literal yet |
| `fp8_e4m3` | `fp8_e4m3` | ascend#160 | pure | backend-side | scheme | CUSTOM backend |
| `nvfp4` | `nvfp4` | ascend#160 | pure | backend-side | scheme | CUSTOM backend |

Maturity notes: the **semantics layer** of every solution is tested on CPU.
Device execution is validated only to port fidelity; end-to-end serving
needs an Ascend NPU environment. The vllm-hust backend adapter is
interface-ready scaffolding — registration and the class-path hook work,
while the full engine path is a host-integration roadmap item.

## Using from a host (pluggable activation)

Installation is not activation. Two mechanisms exist:

**1. Native vLLM bootstrap (works today, explicit per process):**

```bash
# vllm-ascend-hust or vllm-hust serving process
VLLM_HUST_QUANT_KV_SOLUTIONS=int8_dynamic,kivi_int4 vllm serve MODEL ...
```

The `vllm.general_plugins` hook stays a no-op unless this variable names
solutions. Each named solution registers into the host that is importable
in that process (unknown names / missing hosts fail closed). On
vllm-hust, start the engine with `--attention-backend CUSTOM` and the
dtype literal negotiated by the adapter (e.g. `int4_per_token_head` for
`kivi_int4`).

**2. Extension Manager (blocked by design):** the bundle
`org.vllm-hust.quantized-kv-cache` ships a Manifest 0.2 descriptor and is
`import_only` — the manager can inspect but must refuse enablement until
the [HOST_CONTRACT.md](HOST_CONTRACT.md) protocols land in the host. See
`docs/architecture.md` for the flip conditions.

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
