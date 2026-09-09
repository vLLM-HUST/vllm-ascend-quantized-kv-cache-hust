# SPDX-License-Identifier: Apache-2.0
"""Quant-scheme class construction for the vllm-ascend-hust host.

The host dispatches attention-layer quant schemes from the checkpoint's
``fa_quant_type`` via ``@register_scheme``; a scheme's ``create_weights``
may then perform the C8-style impl surgery. This module generates scheme
classes for both families:

* packed solutions (int4/fp4_e2m1/fp8_e4m3/nvfp4): the handler decides the
  storage dtype and carries scales; ``apply`` stays a RuntimeError.
* stateful solutions (int8_dynamic/kivi_int4): ``create_weights`` swaps the
  layer's attention impl to the mixin-backed subclass.
"""

from __future__ import annotations

from typing import Any

from ...solutions.packed.base import PackedKvScheme
from ...solutions.packed.schemes import CACHE_DTYPE_TO_SCHEME


def packed_scheme_for(solution_name: str) -> type[PackedKvScheme] | None:
    return CACHE_DTYPE_TO_SCHEME.get(solution_name)


def build_scheme_cls(solution_name: str, base_scheme_cls: type) -> type:
    """Generate a scheme class over the host ``AscendAttentionScheme``."""
    packed_cls = packed_scheme_for(solution_name)

    def __init__(self: Any, quant_description: Any = None, prefix: Any = None) -> None:
        base_scheme_cls.__init__(self, quant_description, prefix)
        self.quant_description = quant_description or {}
        self.prefix = prefix or ""
        self._solution_name = solution_name

    namespace: dict[str, Any] = {"__init__": __init__}

    if packed_cls is not None:
        # Metadata from the packed handler, behaviour inherited from it.
        namespace.update(
            scheme_key=packed_cls.scheme_key,
            cache_dtype=packed_cls.cache_dtype,
            storage_torch_dtype_name=packed_cls.storage_torch_dtype_name,
            uses_scales=packed_cls.uses_scales,
        )

        def create_weights(self: Any, layer: Any) -> None:
            base_scheme_cls.create_weights(self, layer)
            packed_cls().create_weights(layer)

        def process_weights_after_loading(self: Any, layer: Any) -> None:
            packed_cls().process_weights_after_loading(layer)

        def apply(self: Any, *args: Any) -> Any:
            return packed_cls().apply(*args)

        namespace.update(
            create_weights=create_weights,
            process_weights_after_loading=process_weights_after_loading,
            apply=apply,
        )
        class_name = f"VllmHustPackedKvScheme_{solution_name}"
    else:
        # Stateful solution: create_weights performs the impl surgery.
        namespace.update(scheme_key=f"VLLM_HUST_KV_{solution_name.upper()}")

        def create_weights(self: Any, layer: Any) -> None:
            base_scheme_cls.create_weights(self, layer)
            from ...core.runtime import module_available

            if not module_available("vllm_ascend"):
                raise RuntimeError(
                    "impl surgery requires vllm_ascend "
                    "(vllm-ascend-hust) in this environment"
                )
            from vllm_ascend.attention.attention_v1 import (
                AscendAttentionBackendImpl,
            )

            from .attention import apply_impl_surgery, build_impl_cls

            impl_cls = build_impl_cls(solution_name, AscendAttentionBackendImpl)
            if not apply_impl_surgery(layer, impl_cls):
                raise RuntimeError(
                    f"layer {type(layer).__name__} has no 'impl' attribute; "
                    f"cannot activate the {solution_name} solution"
                )

        def process_weights_after_loading(self: Any, layer: Any) -> None:
            base_scheme_cls.process_weights_after_loading(self, layer)

        def apply(self: Any, *args: Any) -> Any:
            err_msg = (
                f"[vllm-hust/{solution_name}] apply should not be called; "
                "quantized KV compute happens in the attention backend impl."
            )
            raise RuntimeError(err_msg)

        namespace.update(
            create_weights=create_weights,
            process_weights_after_loading=process_weights_after_loading,
            apply=apply,
        )
        class_name = f"VllmHustKvScheme_{solution_name}"

    namespace["__doc__"] = (
        f"Generated quant scheme binding the {solution_name} solution to "
        f"{base_scheme_cls.__name__}."
    )
    return type(class_name, (base_scheme_cls,), namespace)


__all__ = ["build_scheme_cls", "packed_scheme_for"]
