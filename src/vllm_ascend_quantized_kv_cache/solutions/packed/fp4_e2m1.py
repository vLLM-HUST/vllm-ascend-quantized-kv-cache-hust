# SPDX-License-Identifier: Apache-2.0
"""FP4 E2M1 (microscaling) packed KV-cache handler."""

from __future__ import annotations

from .base import PackedKvScheme


class FP4E2M1PackedKvScheme(PackedKvScheme):
    """FP4 E2M1 KV cache quantization for dense-attention models.

    Uses block microscaling (MXFP4): tensors are divided into blocks of 16
    elements, each sharing a single fp8 exponential scaling factor. Scales
    travel with the packed data, so the layer carries none.
    """

    scheme_key = "VLLM_HUST_KV_FP4_E2M1"
    cache_dtype = "fp4_e2m1"
    storage_torch_dtype_name = "uint8"
    uses_scales = False


__all__ = ["FP4E2M1PackedKvScheme"]
