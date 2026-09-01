"""Quantized KV dtype/layout contracts and inert runtime metadata."""

from .dtypes import (
    KVCacheLayout,
    KVQuantMode,
    fp4_e2m1_packed_dim,
    get_kv_quant_mode,
    int4_packed_dim,
    is_quantized_kv_cache,
    nvfp4_packed_dim,
    resolve_layout,
)


class VllmAscendQuantizedKvCacheContractProposal:
    """Metadata-only proposal; this class performs no runtime activation."""


__all__ = [
    "KVCacheLayout",
    "KVQuantMode",
    "VllmAscendQuantizedKvCacheContractProposal",
    "fp4_e2m1_packed_dim",
    "get_kv_quant_mode",
    "int4_packed_dim",
    "is_quantized_kv_cache",
    "nvfp4_packed_dim",
    "resolve_layout",
]
