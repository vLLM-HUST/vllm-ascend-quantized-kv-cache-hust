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
