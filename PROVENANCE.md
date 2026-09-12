# Provenance

Source archives:

- [vLLM-HUST core legacy](https://github.com/intellistream/vllm-hust-legacy-20260831)
- [vLLM Ascend HUST legacy](https://github.com/intellistream/vllm-ascend-hust-legacy-20260831)

Primary history:

- [Core #118 MERGED](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/118)
- [Core #161 MERGED](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/161)
- [Core #181 CLOSED_UNMERGED](https://github.com/intellistream/vllm-hust-legacy-20260831/pull/181)
- [Ascend #116 CLOSED_UNMERGED](https://github.com/intellistream/vllm-ascend-hust-legacy-20260831/pull/116)
- [Ascend #160 CLOSED_UNMERGED](https://github.com/intellistream/vllm-ascend-hust-legacy-20260831/pull/160)
- [Ascend #207 CLOSED_UNMERGED](https://github.com/intellistream/vllm-ascend-hust-legacy-20260831/pull/207)

Original commit patches for the scoped PRs are preserved under `provenance/legacy-patches/`. Mixed core PR #161 is intentionally excluded from blind archival and must be mined file-by-file.

Closed does not mean merged, and open does not mean accepted. These references are migration evidence, not a release receipt. Exact commits, files, authors, licenses, tests, constraints, and benchmark receipts must be recorded before implementation code is accepted.

## Patch → module mining map

What the current method library extracted, and from where. Kernels and
math are preserved from the final patched state; host plumbing (metadata
builders, worker allocation, platform gating) was deliberately **not**
ported — that stays host-roadmap work.

| method module | source patches | extracted | intentionally left behind |
|---|---|---|---|
| `methods/int8_dynamic/` | ascend#116 `0001` | `_calc_int8_scales` / `_quantize_kv_to_int8` / `_dequant_paged_kv_to_dense` math and the decode / chunked-prefill / prefill INT8 branches | debug `print` traces, `do_kv_cache_update` inline edits, C8-mirror naming |
| `methods/kivi_int4/` | ascend#116 `0003`–`0009`, `0013` (final state) | residual ring-window state machine (`_store/_flush_kivi_*`, `_get_kivi_residual_*`), write/flush validation, dense-attention fallback, FIA TND dispatch, chunked-prefill all-new guard | the dict-based residual cache from 0004-0006 (superseded in-repo by the per-request ring buffer), `req_ids` metadata-builder plumbing, spec-decode hooks |
| `ops/triton/kivi_cache.py` | ascend#116 `0007` + `0008` + `0009` + `0013` | pack kernels verbatim at the final patched state (verified bit-exact on 910B2); only module imports adapted (vllm triton shim → direct triton fallback; local vectorcore probe) | nothing else for the pack path; profiling artifacts from 0009's `result/` were never part of the port |
| `ops/kivi_gather.py` + `ops/kivi_layout.py` | ascend#116 `0003`/`0005` approach + final-state validators | **NPU-verified deviation**: the upstream fused triton dequant-gather kernels miscompile on triton-ascend 3.5 (silent garbage loads and dropped stores, varying between runs, at both 16×16 and 32×32 tiles — found on 910B2, reproduced by `scripts/npu_probe_kivi_dim.py`). `kivi_dequant_gather_cache` routes to a plain torch-op gather (the upstream 0003/0005-era implementation); the triton gather kernels are kept unreferenced for future revalidation. Additional NPU fix: aclnnRightShift rejects broadcasting, so int4 unpack uses eight scalar-shift extractions. Pack + gather both verified bit-exact against the CPU reference on 910B2 | running the unvalidated upstream fused gather path at runtime |
| `methods/packed/` | ascend#160 `0001` (+ `0004`/`0005`/`0007` gating context) | four handler classes (`create_weights` / `process_weights_after_loading` / fail-closed `apply`) and the `kv_cache_utils` dispatch table, with registry keys namespaced to `VLLM_HUST_KV_*` | the host `model_runner_v1.py` call site and in-tree `@register_scheme` decoration (now done by the adapter at registration time) |
| `dtypes.py` (layer 0) | core#181 `0001`/`0003` (previously extracted) | unchanged | — |

Not mined, by scope decision: ascend#207 (weight quantization, not KV
cache), core#118/#161 (host-side `CacheDType` plumbing), and ascend#116
`0010`–`0012` (CI/upstream-sync noise).
