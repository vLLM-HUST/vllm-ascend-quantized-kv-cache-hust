# SPDX-License-Identifier: Apache-2.0
"""Pure math for the dynamic per-channel INT8 KV-cache solution.

Mined from legacy ascend PR #116 commit 0001
(``AscendAttentionBackendImpl._calc_int8_scales`` and
``_quantize_kv_to_int8``). Everything here runs on CPU tensors too, so the
numerics are unit-testable without an NPU.

Semantics: on the first prefill the amax is taken over the token dimension
(``dim=0``), yielding one scale per ``(kv_head, head_dim)`` channel pair
(keepdim, shape ``[1, H, D]``). ``inv_scale = 127 / amax``; the offset stays
zero (symmetric). Quantization is ``clamp(round(x * inv_scale + offset))``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...core.runtime import import_torch


@dataclass
class Int8DynamicScales:
    """Online per-channel scales for one attention layer.

    ``*_inv_scale`` / ``*_offset`` share the amax shape ``[1, H, D]``;
    ``*_aq_*`` are the same values reshaped to the BNSD layout the NPU
    fused-inference operator expects (``[1, H, 1, D]``).
    """

    k_inv_scale: Any
    k_offset: Any
    v_inv_scale: Any
    v_offset: Any
    k_aq_scale: Any
    k_aq_offset: Any
    v_aq_scale: Any
    v_aq_offset: Any


class Int8DynamicSemantics:
    def __init__(self, config: Any) -> None:
        self.config = config

    def calc_scales(self, key: Any, value: Any) -> Int8DynamicScales:
        """Compute the online scales exactly as the legacy implementation."""
        torch = import_torch("Int8DynamicSemantics.calc_scales")
        k_max = key.abs().amax(dim=0, keepdim=True).clamp(min=1e-12)
        v_max = value.abs().amax(dim=0, keepdim=True).clamp(min=1e-12)
        k_inv_scale = 127.0 / k_max
        k_offset = torch.zeros_like(k_inv_scale)
        v_inv_scale = 127.0 / v_max
        v_offset = torch.zeros_like(v_inv_scale)
        bnsd = (1, self.config.num_kv_heads, 1, self.config.head_size)
        return Int8DynamicScales(
            k_inv_scale=k_inv_scale,
            k_offset=k_offset,
            v_inv_scale=v_inv_scale,
            v_offset=v_offset,
            k_aq_scale=(1.0 / k_inv_scale).view(bnsd).contiguous(),
            k_aq_offset=k_offset.view(bnsd).contiguous(),
            v_aq_scale=(1.0 / v_inv_scale).view(bnsd).contiguous(),
            v_aq_offset=v_offset.view(bnsd).contiguous(),
        )

    @staticmethod
    def quantize(x: Any, inv_scale: Any, offset: Any) -> Any:
        """Quantize K/V from float to INT8 with the per-channel scales."""
        torch = import_torch("Int8DynamicSemantics.quantize")
        return torch.clamp(torch.round(x * inv_scale + offset), -128, 127).to(
            torch.int8
        )

    @staticmethod
    def dequantize(x: Any, inv_scale: Any, offset: Any, target_dtype: Any) -> Any:
        """Inverse of :meth:`quantize`: ``(x - offset) * (1 / inv_scale)``."""
        return (x.to(target_dtype) - offset) * (1.0 / inv_scale)

    def describe(self) -> dict[str, Any]:
        return {
            "scheme": "int8_dynamic_per_channel",
            "scale_shape": [1, self.config.num_kv_heads, self.config.head_size],
            "scale_axis": "token-amax (dim=0), computed on first prefill",
            "symmetric": True,
        }


__all__ = ["Int8DynamicScales", "Int8DynamicSemantics"]
