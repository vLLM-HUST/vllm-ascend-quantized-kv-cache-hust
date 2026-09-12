# SPDX-License-Identifier: Apache-2.0
"""vllm-ascend-hust 宿主的 Impl 类构造。

动态生成 "方法 mixin + 宿主 AscendAttentionBackendImpl 基类" 的子类，
遵循宿主在树 C8 先例（kv_c8.py：在 quant scheme 的 create_weights 里
做 layer.impl.__class__ = AscendC8AttentionBackendImpl 的类手术）。
"""

from __future__ import annotations

from typing import Any

from ...methods.int8_dynamic.attention_mixin import Int8DynamicAttentionMixin
from ...methods.kivi_int4.attention_mixin import KiviInt4AttentionMixin

# 方案名 -> (mixin 类, 状态初始化方法名)。
# mixin 提供设备路径方法；状态初始化方法在构造/类手术后建立全部属性。
_MIXINS: dict[str, tuple[type, str]] = {
    "int8_dynamic": (Int8DynamicAttentionMixin, "_init_int8_dynamic_state"),
    "kivi_int4": (KiviInt4AttentionMixin, "_init_kivi_state"),
}

# Positional slot of kv_cache_dtype in the AscendAttentionBackendImpl
# signature: (num_heads, head_size, scale, num_kv_heads, alibi_slopes,
# sliding_window, kv_cache_dtype, ...).
_KV_DTYPE_ARG_INDEX = 6


def supported_impl_methods() -> tuple[str, ...]:
    return tuple(sorted(_MIXINS))


def build_impl_cls(method_name: str, base_impl_cls: type) -> type:
    """把方法 mixin 与宿主 impl 基类组合成子类。"""
    try:
        mixin_cls, init_method = _MIXINS[method_name]
    except KeyError:
        raise ValueError(
            f"no Ascend impl mixin wired for method {method_name!r}; "
            f"supported: {', '.join(supported_impl_methods())}"
        ) from None

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        # 先跑宿主基类构造（建立 num_heads/head_size/vllm_config 等属性），
        # 再初始化方案状态；kv_cache_dtype 支持关键字或第 7 个位置参数。
        base_impl_cls.__init__(self, *args, **kwargs)
        kv_cache_dtype = kwargs.get("kv_cache_dtype")
        if kv_cache_dtype is None and len(args) > _KV_DTYPE_ARG_INDEX:
            kv_cache_dtype = args[_KV_DTYPE_ARG_INDEX]
        if kv_cache_dtype is None:
            kv_cache_dtype = getattr(self, "kv_cache_dtype", None)
        getattr(self, init_method)(kv_cache_dtype, getattr(self, "vllm_config", None))

    namespace = {
        "__init__": __init__,
        # apply_impl_surgery reads this to initialise mixin state after a
        # class swap (a swap does not re-run __init__).
        "_state_init_method": init_method,
        "__doc__": (
            f"{method_name} quantized-KV attention impl "
            f"(mixin {mixin_cls.__name__} over "
            f"{base_impl_cls.__name__})."
        ),
    }
    return type(
        f"{mixin_cls.__name__}On{base_impl_cls.__name__}",
        (mixin_cls, base_impl_cls),
        namespace,
    )


def apply_impl_surgery(layer: Any, impl_cls: type) -> bool:
    """C8 式逐 layer 的 impl 类替换（成功返回 True）。

    关键细节：换 __class__ 不会重跑 __init__，所以交换后要显式调用
    mixin 的状态初始化方法——否则 enable_kivi / 残差窗口等属性缺失，
    forward 时才会炸。
    """
    if not hasattr(layer, "impl"):
        return False
    impl = layer.impl
    impl.__class__ = impl_cls
    init_method = getattr(impl_cls, "_state_init_method", None)
    if init_method is not None:
        getattr(impl, init_method)(
            getattr(impl, "kv_cache_dtype", None),
            getattr(impl, "vllm_config", None),
        )
    return True


__all__ = ["apply_impl_surgery", "build_impl_cls", "supported_impl_methods"]
