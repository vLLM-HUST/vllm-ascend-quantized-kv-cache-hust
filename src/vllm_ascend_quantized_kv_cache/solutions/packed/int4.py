# SPDX-License-Identifier: Apache-2.0
"""INT4 packed KV-cache handler (per-token-head, 2x int4 per byte)."""

from __future__ import annotations

from .base import PackedKvScheme


class Int4PackedKvScheme(PackedKvScheme):
    """INT4 KV cache quantization for dense-attention models.

    Uses per-token-head dynamic quantization with symmetric scale. Each
    token-head pair computes its own scale, and the INT4 data is packed as
    2x int4 per byte.
    """

    scheme_key = "VLLM_HUST_KV_INT4"
    cache_dtype = "int4"
    storage_torch_dtype_name = "uint8"
    uses_scales = True


__all__ = ["Int4PackedKvScheme"]
