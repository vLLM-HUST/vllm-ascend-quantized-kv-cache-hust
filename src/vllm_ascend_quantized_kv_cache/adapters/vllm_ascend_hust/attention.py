# SPDX-License-Identifier: Apache-2.0
"""Impl-class construction for the vllm-ascend-hust host.

Generates ``AscendAttentionBackendImpl`` subclasses combining our solution
mixins with the host base class, following the in-tree C8 precedent
(``kv_c8.py``: ``layer.impl.__class__ = AscendC8AttentionBackendImpl``
inside a quant scheme's ``create_weights``).
"""

from __future__ import annotations

from typing import Any

from ...solutions.int8_dynamic.attention_mixin import Int8DynamicAttentionMixin
from ...solutions.kivi_int4.attention_mixin import KiviInt4AttentionMixin

_MIXINS: dict[str, tuple[type, str]] = {
    "int8_dynamic": (Int8DynamicAttentionMixin, "_init_int8_dynamic_state"),
    "kivi_int4": (KiviInt4AttentionMixin, "_init_kivi_state"),
}

# Positional slot of kv_cache_dtype in the AscendAttentionBackendImpl
# signature: (num_heads, head_size, scale, num_kv_heads, alibi_slopes,
# sliding_window, kv_cache_dtype, ...).
_KV_DTYPE_ARG_INDEX = 6


def supported_impl_solutions() -> tuple[str, ...]:
    return tuple(sorted(_MIXINS))


def build_impl_cls(solution_name: str, base_impl_cls: type) -> type:
    """Combine the solution mixin with the host impl base class."""
    try:
        mixin_cls, init_method = _MIXINS[solution_name]
    except KeyError:
        raise ValueError(
            f"no Ascend impl mixin wired for solution {solution_name!r}; "
            f"supported: {', '.join(supported_impl_solutions())}"
        ) from None

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
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
            f"{solution_name} quantized-KV attention impl "
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
    """C8-style per-layer impl substitution (returns True when applied).

    After the swap the mixin state is initialised explicitly, mirroring
    what the generated ``__init__`` does at construction time.
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


__all__ = ["apply_impl_surgery", "build_impl_cls", "supported_impl_solutions"]
