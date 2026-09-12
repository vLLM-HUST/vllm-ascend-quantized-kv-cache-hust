# Architecture

The package is a **method library + host adapter layer**, not a monolithic
plugin. Five layers, strictly dependency-ordered: importing the package
pulls in none of torch / vllm / triton; every layer loads its heavy
dependencies lazily at use time.

```
layer 0  contracts      dtypes.py (KVQuantMode, KVCacheLayout, resolve_layout)
layer 1  methods        methods/base.py     MethodSpec/Config/KvQuantMethod
                        methods/registry.py fail-closed registry
        core            core/hosts.py host names; core/runtime.py NPU probes
layer 2  methods/impl    methods/int8_dynamic/   (ascend#116/0001)
                         methods/kivi_int4/      (ascend#116/0003-0013)
                         methods/int4_packed.py, fp4_e2m1.py,
                           fp8_e4m3.py, nvfp4.py  (ascend#160)
                         methods/packed_base.py  shared format semantics
layer 3  ops            ops/int8_ops.py            torch helpers
                        ops/kivi_layout.py         shared validators
                        ops/kivi_gather.py         torch gather (routed)
                        ops/triton/kivi_pack.py    triton-ascend pack (routed)
                        ops/triton/kivi_gather_experimental.py   not routed
layer 4  adapters       adapters/vllm_ascend_hust/  scheme registry + impl surgery
                        adapters/vllm_hust/         CUSTOM backend registration
layer 5  bootstrap      bootstrap.py  vllm.general_plugins hook (default no-op)
```

## Module rules

1. **Metadata-only registration.** A method's `__init__.py` constructs a
   `MethodSpec` (name, dtype, provenance, support matrix, config
   validator) and registers it. The spec references heavy modules through
   *loaders*, never imports them.
2. **Lazy semantics.** `KvQuantMethod.semantics` invokes the spec's
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

`kv_methods` (package root) is the single entry point:

- `get(name, **config) -> KvQuantMethod` — fail-closed on unknown names,
  invalid geometry, unsupported hosts.
- `KvQuantMethod.resolve_layout()` — reuses the layer-0 contract.
- `KvQuantMethod.semantics` — pure math (scale computation, pack/unpack,
  window bookkeeping), CPU-testable.
- `KvQuantMethod.host_adapter(host)` — per-host integration object exposing
  `register()` plus the bind-style class factories.

## How a method plugs into each host

Both mechanisms require **zero host-repository changes**.

### vllm-ascend-hust

1. `AscendHustAdapter.register()` imports the host's
   `@register_scheme` registry and registers a generated scheme class under
   a namespaced quant_type key (`VLLM_HUST_KV_*`; duplicate keys raise on
   the host, so namespacing avoids collisions with in-tree schemes).
2. Attention layers dispatch schemes from the checkpoint `fa_quant_type`
   key (ModelSlim path).
3. For the stateful methods the generated scheme's `create_weights`
   performs the in-tree C8 precedent — `layer.impl.__class__ = <our impl>`
   — then initialises the mixin state explicitly (a class swap does not
   re-run `__init__`). Our impl classes are `type()` combinations of the
   method mixin over the host `AscendAttentionBackendImpl`.

### vllm-hust

1. `VllmHustAdapter.register()` imports the host attention registry and
   maps `AttentionBackendEnum.CUSTOM` to this package's backend class path.
   The host stores string paths and imports them lazily; our
   `adapters/vllm_hust/backend.py` builds the actual class in a module
   `__getattr__` at that moment.
2. `CacheDType` on the host is a closed Literal enforced at three layers
   (pydantic config, backend selector, torch-dtype lookup). The adapter
   therefore *negotiates* an existing literal per method
   (`map_cache_dtype`) and fails closed when none fits (`fp4_e2m1`
   today) — adding a literal is a host-roadmap item, not a runtime hack.
3. Device execution still routes to the same Ascend NPU kernels;
   `get_impl_cls` refuses to run off-NPU.

## Bootstrap and activation discipline

There is exactly **one activation pipeline**: `core/activation.py::
activate(name, host=None, **config)` → `methods.registry.get_method`
→ `method.host_adapter(host)` → `adapter.register()`. Two thin callers
sit on top of it:

- `kv_methods.activate(name, host=..., **config)` — the programmatic
  facade entry (explicit host, or auto-detect via `core.hosts.
  detect_host`, which prefers `vllm_ascend` then `vllm`);
- `bootstrap.register_plugins` — the `vllm.general_plugins` entry-point
  hook, which only parses `VLLM_HUST_KV_METHODS`, detects the host, and
  delegates each name to the same pipeline.

`bootstrap.register_plugins` returns immediately unless the environment
names methods. This keeps the Extension-Manager invariant —
*installation only makes the bundle discoverable* — while allowing
explicit per-process activation today. The static manifest stays
`status: import_only`; flipping it to `active` requires the
HOST_CONTRACT protocols (`vllm.kv-cache.dtype-registry.v1`,
`vllm.kv-cache.layout.v1`, `vllm.attention.quantized-kv.v1`,
`vllm.kv-transfer.quantized-layout.v1`) to exist in the host, plus
compatibility evidence.

Adapter registration semantics: adapters are guarded by **host-stack
importability**, not by platform choice. `vllm_ascend_hust.register()`
is idempotent — the host registry raises on duplicate keys, and the
adapter accepts the duplicate only when the existing class has the same
generated class name (an equivalent registration of the same method
binding), marking `already_registered=True`; a same-key/different-name
scheme is a real conflict and fails closed. In dual-stack environments
(both `vllm` and `vllm_ascend` importable, e.g. NPU dev containers) both
adapters can be driven explicitly; their registries are independent.

## Adding a new method

1. `methods/<name>/__init__.py` — build a `MethodSpec` with a config
   validator and a semantics loader; call `register_method`.
2. `methods/<name>/semantics.py` — pure torch math (CPU tests).
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
