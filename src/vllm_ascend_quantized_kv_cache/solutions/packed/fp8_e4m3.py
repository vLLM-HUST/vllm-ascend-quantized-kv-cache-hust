# SPDX-License-Identifier: Apache-2.0
"""FP8 E4M3 KV-cache handler (per-tensor scales)."""

from __future__ import annotations

from .base import PackedKvScheme


class FP8E4M3PackedKvScheme(PackedKvScheme):
    """FP8 E4M3 KV cache quantization for dense-attention models.

    Uses per-tensor scaling with FP8 E4M3 storage. Supports both static
    (checkpoint-loaded) and dynamic (computed) scales.
    """

    scheme_key = "VLLM_HUST_KV_FP8_E4M3"
    cache_dtype = "fp8_e4m3"
    storage_torch_dtype_name = "float8_e4m3fn"
    uses_scales = True


__all__ = ["FP8E4M3PackedKvScheme"]
