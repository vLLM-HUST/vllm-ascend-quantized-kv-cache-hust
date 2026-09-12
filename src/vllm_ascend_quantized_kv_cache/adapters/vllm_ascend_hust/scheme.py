# SPDX-License-Identifier: Apache-2.0
"""vllm-ascend-hust 宿主的 quant scheme 类构造。

宿主通过 checkpoint 的 ``fa_quant_type`` 键经 @register_scheme 分发
attention 层的量化 scheme；scheme 的 create_weights 随后可以执行
C8 式 impl 手术。本模块为两类方法生成 scheme 类：

* packed 系（int4/fp4_e2m1/fp8_e4m3/nvfp4）：handler 决定存储 dtype、
  携带 scale；apply 保持 RuntimeError（量化在 backend 内核）。
* 有状态系（int8_dynamic/kivi_int4）：create_weights 把 layer 的
  attention impl 换成 mixin 支撑的子类。
"""

from __future__ import annotations

from typing import Any

from ...methods.packed_base import METHOD_NAME_TO_SEMANTICS, PackedFormatSemantics


def packed_semantics_for(method_name: str) -> type[PackedFormatSemantics] | None:
    return METHOD_NAME_TO_SEMANTICS.get(method_name)


def build_scheme_cls(method_name: str, base_scheme_cls: type) -> type:
    """在宿主 AscendAttentionScheme 之上生成方法专属的 scheme 类。"""
    packed_cls = packed_semantics_for(method_name)

    def __init__(self: Any, quant_description: Any = None, prefix: Any = None) -> None:
        base_scheme_cls.__init__(self, quant_description, prefix)
        self.quant_description = quant_description or {}
        self.prefix = prefix or ""
        self._method_name = method_name

    namespace: dict[str, Any] = {"__init__": __init__}

    if packed_cls is not None:
        # packed 系：元数据来自 handler，行为也委托给 handler 实例。
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
        class_name = f"VllmHustPackedSemantics_{method_name}"
    else:
        # 有状态系：create_weights 执行 impl 类手术（C8 先例）。
        namespace.update(scheme_key=f"VLLM_HUST_KV_{method_name.upper()}")

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

            impl_cls = build_impl_cls(method_name, AscendAttentionBackendImpl)
            # 手术 + 术后状态初始化（换类不会自动跑 __init__）
            if not apply_impl_surgery(layer, impl_cls):
                raise RuntimeError(
                    f"layer {type(layer).__name__} has no 'impl' attribute; "
                    f"cannot activate the {method_name} method"
                )

        def process_weights_after_loading(self: Any, layer: Any) -> None:
            base_scheme_cls.process_weights_after_loading(self, layer)

        def apply(self: Any, *args: Any) -> Any:
            err_msg = (
                f"[vllm-hust/{method_name}] apply should not be called; "
                "quantized KV compute happens in the attention backend impl."
            )
            raise RuntimeError(err_msg)

        namespace.update(
            create_weights=create_weights,
            process_weights_after_loading=process_weights_after_loading,
            apply=apply,
        )
        class_name = f"VllmHustKvScheme_{method_name}"

    namespace["__doc__"] = (
        f"Generated quant scheme binding the {method_name} method to "
        f"{base_scheme_cls.__name__}."
    )
    return type(class_name, (base_scheme_cls,), namespace)


__all__ = ["build_scheme_cls", "packed_semantics_for"]
