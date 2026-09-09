# Quantized KV cache host contract proposal

The extracted dtype and packed-layout resolver is device-independent. Activation
on vLLM Ascend requires:

1. `vllm.kv-cache.dtype-registry.v1`: register a namespaced dtype descriptor
   without extending core string literals in-place;
2. `vllm.kv-cache.layout.v1`: negotiate storage dtype, packed dimension, scales,
   block size, and alignment before allocation;
3. `vllm.attention.quantized-kv.v1`: query a device-provider capability and bind
   matching quantize/dequantize/attention kernels;
4. `vllm.kv-transfer.quantized-layout.v1`: include the complete versioned layout
   in connector handshakes and reject mismatches.

Unknown layouts, missing kernels, incompatible head sizes, connector mismatches,
and unsupported graph modes must fail closed. A uint8 allocation alone is not
evidence that a dtype is supported. Triton, CATLASS, and Ascend kernels remain
separate device-provider components with hardware-specific tests.

## Interim native integration path (before the contract lands)

The solution library is usable today through each host's existing extension
surfaces, without any host-repository change and without flipping the
manifest away from `import_only`:

- **vllm-ascend-hust:** external registration into the host's
  `@register_scheme` quant registry (new namespaced quant_type keys) plus
  the in-tree C8-style impl substitution, bootstrapped per process through
  the `vllm.general_plugins` entry point.
- **vllm-hust:** `AttentionBackendEnum.CUSTOM` class-path registration in
  the host attention registry, reusing an existing `CacheDType` literal
  negotiated by the adapter.

This interim path is *fail-closed by construction* (unknown names, missing
hosts, missing kernels, and non-NPU devices all raise), but it is not the
manager-mediated activation: enabling intent lives in the operator's
environment (`VLLM_HUST_QUANT_KV_SOLUTIONS`), not in the Extension Manager
catalog. Flip the manifest `implementation.status` to `active` only when
the four protocols above exist in the host, adapter-mediated activation
replaces the environment opt-in, and compatibility evidence (minimum and
latest host versions) is recorded per the packaging guide's version
verification table.
