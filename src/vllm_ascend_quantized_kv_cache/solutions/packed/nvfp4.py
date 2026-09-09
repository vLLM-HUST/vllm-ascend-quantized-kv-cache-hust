# SPDX-License-Identifier: Apache-2.0
"""NVFP4 packed KV-cache handler (fp4 data + fp8 block scales)."""

from __future__ import annotations

from .base import PackedKvScheme


class NVFP4PackedKvScheme(PackedKvScheme):
    """NVFP4 KV cache quantization for dense-attention models.

    Uses packed fp4 data plus fp8 block scales: each block of 16 fp4
    elements shares a single fp8 scale factor.
    """

    scheme_key = "VLLM_HUST_KV_NVFP4"
    cache_dtype = "nvfp4"
    storage_torch_dtype_name = "uint8"
    uses_scales = False


__all__ = ["NVFP4PackedKvScheme"]
