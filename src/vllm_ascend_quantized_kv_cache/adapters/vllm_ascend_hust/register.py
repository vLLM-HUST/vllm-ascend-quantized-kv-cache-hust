# SPDX-License-Identifier: Apache-2.0
"""Registration entry point for the vllm-ascend-hust host.

Zero-host-change attach path: the external package calls the host's
``@register_scheme`` registry with a *new* quant_type key (duplicate keys
raise on the host), and the scheme reaches attention layers through the
checkpoint ``fa_quant_type`` key or the general-plugins bootstrap.
"""

from __future__ import annotations

from typing import Any

from ..base import HostAdapter
from .attention import supported_impl_solutions
from .scheme import build_scheme_cls, packed_scheme_for


class AscendHustAdapter(HostAdapter):
    host = "vllm_ascend_hust"
    host_module = "vllm_ascend"

    def register(
        self, *, quant_type: str | None = None, layer_type: str = "attention"
    ) -> dict[str, Any]:
        """Register the solution's scheme into the host registry.

        Idempotent: re-registering an identical solution name is a no-op.
        """
        self.require_host()
        try:
            from vllm_ascend.quantization.methods.registry import register_scheme
        except ImportError as exc:
            raise RuntimeError(
                "vllm_ascend.quantization.methods.registry is unavailable; "
                "this adapter targets vllm-ascend-hust"
            ) from exc

        solution_name = self.solution.name
        try:
            from vllm_ascend.quantization.methods.base import (
                AscendAttentionScheme,
            )
        except ImportError as exc:
            raise RuntimeError(
                "vllm_ascend AscendAttentionScheme base class is unavailable"
            ) from exc

        scheme_cls = build_scheme_cls(solution_name, AscendAttentionScheme)
        key = quant_type or self.default_quant_type(solution_name)
        try:
            register_scheme(key, layer_type)(scheme_cls)
        except ValueError:
            # Duplicate key: accept only when it maps to an identical class.
            from vllm_ascend.quantization.methods.registry import get_scheme_class

            existing = get_scheme_class(key, layer_type)
            if existing is not scheme_cls:
                raise
        return {
            "host": self.host,
            "solution": solution_name,
            "quant_type": key,
            "layer_type": layer_type,
            "scheme_cls": scheme_cls,
        }

    @staticmethod
    def default_quant_type(solution_name: str) -> str:
        packed = packed_scheme_for(solution_name)
        if packed is not None:
            return packed.scheme_key
        if solution_name in supported_impl_solutions():
            return f"VLLM_HUST_KV_{solution_name.upper()}"
        raise ValueError(f"unknown solution for the Ascend adapter: {solution_name!r}")


__all__ = ["AscendHustAdapter"]
