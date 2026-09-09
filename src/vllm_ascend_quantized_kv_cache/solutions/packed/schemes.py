# SPDX-License-Identifier: Apache-2.0
"""Dispatch utilities for the packed KV-cache handlers.

Ported from legacy ascend PR #160 commit 0001
(``vllm_ascend/quantization/kv_cache_utils.py``), with the registry keys
namespaced under ``VLLM_HUST_KV_*`` so they cannot collide with the
in-tree ``@register_scheme`` entries on the host.
"""

from __future__ import annotations

from typing import Any

from ...core.runtime import import_torch
from .base import PackedKvScheme
from .fp4_e2m1 import FP4E2M1PackedKvScheme
from .fp8_e4m3 import FP8E4M3PackedKvScheme
from .int4 import Int4PackedKvScheme
from .nvfp4 import NVFP4PackedKvScheme

#: ``--kv-cache-dtype`` string -> handler class (legacy parity mapping).
CACHE_DTYPE_TO_SCHEME: dict[str, type[PackedKvScheme]] = {
    "int4": Int4PackedKvScheme,
    "nvfp4": NVFP4PackedKvScheme,
    "fp8_e4m3": FP8E4M3PackedKvScheme,
    "fp4_e2m1": FP4E2M1PackedKvScheme,
}

#: Non-quantized dtype strings that must never dispatch to a handler.
_NON_QUANTIZED_DTYPES = frozenset({"", "auto", "float16", "bfloat16"})


def get_packed_scheme(cache_dtype: str) -> PackedKvScheme | None:
    """Return a handler instance for *cache_dtype*, or ``None``.

    Unknown or non-quantized dtype strings return ``None`` (legacy parity:
    the caller decides whether that is a no-op or a fail-closed error).
    """
    scheme_cls = CACHE_DTYPE_TO_SCHEME.get(cache_dtype)
    if scheme_cls is None:
        return None
    return scheme_cls()


def setup_kv_cache_quant(layer: Any, cache_dtype: str) -> PackedKvScheme | None:
    """Attach handler weights to *layer* for the given ``--kv-cache-dtype``.

    Returns the handler that was applied, or ``None`` when *cache_dtype* is
    not a quantized dtype known to this library (the call is then a no-op).
    """
    if not cache_dtype or cache_dtype in _NON_QUANTIZED_DTYPES:
        return None

    scheme = get_packed_scheme(cache_dtype)
    if scheme is None:
        return None

    # create_weights imports torch lazily.
    scheme.create_weights(layer)
    return scheme


def storage_torch_dtype(cache_dtype: str) -> Any:
    """Resolve the torch storage dtype for *cache_dtype* (torch required)."""
    torch = import_torch("storage_torch_dtype")
    scheme = get_packed_scheme(cache_dtype)
    if scheme is None:
        raise ValueError(f"unknown packed KV cache dtype: {cache_dtype!r}")
    return getattr(torch, scheme.storage_torch_dtype_name)


__all__ = [
    "CACHE_DTYPE_TO_SCHEME",
    "get_packed_scheme",
    "setup_kv_cache_quant",
    "storage_torch_dtype",
]
