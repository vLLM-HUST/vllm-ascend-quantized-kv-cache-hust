# Architecture

The package is a **solution library + host adapter layer**, not a monolithic
plugin. Five layers, strictly dependency-ordered: importing the package
pulls in none of torch / vllm / triton; every layer loads its heavy
dependencies lazily at use time.

```
layer 0  contracts      dtypes.py (KVQuantMode, KVCacheLayout, resolve_layout)
layer 1  core           core/spec.py SolutionSpec/Config/KvSolution
                        core/registry.py fail-closed registry
                        core/runtime.py NPU / import capability probes
layer 2  solutions      solutions/int8_dynamic/  (ascend#116/0001)
                        solutions/kivi_int4/     (ascend#116/0003-0013)
                        solutions/packed/        (ascend#160: int4, fp4_e2m1,
                                                  fp8_e4m3, nvfp4)
layer 3  ops            ops/int8_ops.py            torch helpers
                        ops/triton/kivi_cache.py   triton-ascend kernels
                                                   (Ascend NPU only)
layer 4  adapters       adapters/vllm_ascend_hust/  scheme registry + impl surgery
                        adapters/vllm_hust/         CUSTOM backend registration
layer 5  bootstrap      bootstrap.py  vllm.general_plugins hook (default no-op)
```

## Module rules

1. **Metadata-only registration.** A solution's `__init__.py` constructs a
   `SolutionSpec` (name, dtype, provenance, support matrix, config
   validator) and registers it. The spec references heavy modules through
   *loaders*, never imports them.
2. **Lazy semantics.** `KvSolution.semantics` invokes the spec's
   `semantics_loader(config)` the first time it is touched. Semantics
   modules may import torch (they are CPU-testable) but are never imported
   by registration.
3. **Device code behind loaders.** `ops/` modules import triton /
   torch_npu. Only the mixin forward paths and kernel launch hooks reach
   them; everything is imported inside the using function so a missing NPU
   stack produces one precise error message, not an import cascade.
4. **Adapters bind, never import at module scope.** An adapter receives the
   host base classes as arguments (bind style, `build_impl_cls(name,
   base_cls)`) or imports them inside `register()`. The package itself has
   zero vllm/vllm_ascend imports.

## Unified interface

`kv_solutions` (package root) is the single entry point:

- `get(name, **config) -> KvSolution` — fail-closed on unknown names,
  invalid geometry, unsupported hosts.
- `KvSolution.resolve_layout()` — reuses the layer-0 contract.
- `KvSolution.semantics` — pure math (scale computation, pack/unpack,
  window bookkeeping), CPU-testable.
- `KvSolution.host_adapter(host)` — per-host integration object exposing
  `register()` plus the bind-style class factories.

## How a solution plugs into each host

Both mechanisms require **zero host-repository changes**.

### vllm-ascend-hust

1. `AscendHustAdapter.register()` imports the host's
   `@register_scheme` registry and registers a generated scheme class under
   a namespaced quant_type key (`VLLM_HUST_KV_*`; duplicate keys raise on
   the host, so namespacing avoids collisions with in-tree schemes).
2. Attention layers dispatch schemes from the checkpoint `fa_quant_type`
   key (ModelSlim path).
3. For the stateful solutions the generated scheme's `create_weights`
   performs the in-tree C8 precedent — `layer.impl.__class__ = <our impl>`
   — then initialises the mixin state explicitly (a class swap does not
   re-run `__init__`). Our impl classes are `type()` combinations of the
   solution mixin over the host `AscendAttentionBackendImpl`.

### vllm-hust

1. `VllmHustAdapter.register()` imports the host attention registry and
   maps `AttentionBackendEnum.CUSTOM` to this package's backend class path.
   The host stores string paths and imports them lazily; our
   `adapters/vllm_hust/backend.py` builds the actual class in a module
   `__getattr__` at that moment.
2. `CacheDType` on the host is a closed Literal enforced at three layers
   (pydantic config, backend selector, torch-dtype lookup). The adapter
   therefore *negotiates* an existing literal per solution
   (`map_cache_dtype`) and fails closed when none fits (`fp4_e2m1`
   today) — adding a literal is a host-roadmap item, not a runtime hack.
3. Device execution still routes to the same Ascend NPU kernels;
   `get_impl_cls` refuses to run off-NPU.

## Bootstrap and activation discipline

`bootstrap.register_plugins` (entry point `vllm.general_plugins`) returns
immediately unless `VLLM_HUST_QUANT_KV_SOLUTIONS` names solutions. This
keeps the Extension-Manager invariant — *installation only makes the
bundle discoverable* — while allowing explicit per-process activation
today. The static manifest stays `status: import_only`; flipping it to
`active` requires the HOST_CONTRACT protocols (`vllm.kv-cache.dtype-registry.v1`,
`vllm.kv-cache.layout.v1`, `vllm.attention.quantized-kv.v1`,
`vllm.kv-transfer.quantized-layout.v1`) to exist in the host, plus
compatibility evidence.

## Adding a new solution

1. `solutions/<name>/__init__.py` — build a `SolutionSpec` with a config
   validator and a semantics loader; call `register_solution`.
2. `solutions/<name>/semantics.py` — pure torch math (CPU tests).
3. Device work: extend `ops/` and/or write an attention mixin whose state
   init follows the `_init_*_state` convention; add the mixin to
   `adapters/vllm_ascend_hust/attention.py::_MIXINS`.
4. Wire adapter factories in the spec, extend the support matrix, and map
   a `CacheDType` literal in `adapters/vllm_hust/register.py` if the
   vllm-hust host is targeted.
5. Record the source patch in `provenance` and in `PROVENANCE.md`.

## Testing strategy without an NPU

- Semantics and registry run on CPU tensors.
- Host adapters are tested against stub base classes that mirror the
  researched host surfaces; kernel launch hooks are stubbed so the
  validation logic is exercised without triton.
- `tests/test_facade.py` asserts the import-hygiene invariant in a clean
  subprocess (no torch / vllm / triton / vllm_ascend in `sys.modules`).
- Kernel numerics and end-to-end serving need an Ascend NPU CI stage.
