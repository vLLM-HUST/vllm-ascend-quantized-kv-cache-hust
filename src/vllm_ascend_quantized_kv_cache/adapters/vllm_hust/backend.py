# SPDX-License-Identifier: Apache-2.0
"""Lazily-built attention backend for the vllm-hust host.

The host registry stores fully qualified class *paths* and imports them
lazily at backend selection. This module therefore exposes
``HustQuantizedKvAttentionBackend`` through a module-level ``__getattr__``:
the class (which must subclass the host ``AttentionBackend``) is only
constructed when a vllm process actually resolves the path, so the
package stays dependency-free everywhere else.

Status: interface-ready scaffolding. The class implements the registry
surface (name, supported dtypes, cache shapes) from the layout contract
and binds the solution mixins for device execution on Ascend NPU. The
metadata builder delegates to the host's generic builder; end-to-end
serving through this backend is validated by the host-integration roadmap
item, not by this release.
"""

from __future__ import annotations

from typing import Any

_SOLUTION_NAME = "int8_dynamic"


def _resolve_solution_name() -> str:
    """Pick the solution this backend serves (env-overridable)."""
    import os

    return os.environ.get("VLLM_HUST_QKV_BACKEND_SOLUTION", _SOLUTION_NAME)


def _build_backend() -> type:
    from vllm.v1.attention.backend import (
        AttentionBackend,
        AttentionImpl,
    )

    from ....core.runtime import npu_available
    from ....solutions.int8_dynamic.attention_mixin import (
        Int8DynamicAttentionMixin,
    )
    from ....solutions.kivi_int4.attention_mixin import KiviInt4AttentionMixin
    from .register import map_cache_dtype

    solution_name = _resolve_solution_name()
    cache_dtype_literal = map_cache_dtype(solution_name)
    mixin = {
        "int8_dynamic": Int8DynamicAttentionMixin,
        "kivi_int4": KiviInt4AttentionMixin,
    }[solution_name]

    class _Impl(mixin, AttentionImpl):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            AttentionImpl.__init__(self, *args, **kwargs)
            kv_cache_dtype = kwargs.get("kv_cache_dtype")
            if kv_cache_dtype is None and len(args) > 6:
                kv_cache_dtype = args[6]
            if solution_name == "int8_dynamic":
                self._init_int8_dynamic_state(kv_cache_dtype or cache_dtype_literal)
            else:
                self._init_kivi_state(
                    kv_cache_dtype or cache_dtype_literal,
                    getattr(self, "vllm_config", None),
                )

    class HustQuantizedKvAttentionBackend(AttentionBackend):  # noqa: N801
        _solution_name = solution_name
        _cache_dtype_literal = cache_dtype_literal

        @classmethod
        def get_name(cls) -> str:
            return f"VLLM_HUST_QUANTIZED_KV_{solution_name.upper()}"

        @classmethod
        def get_supported_kernel_block_sizes(cls) -> list[int]:
            return [128]

        @classmethod
        def supports_kv_cache_dtype(cls) -> bool:
            return True

        @classmethod
        def get_kv_cache_shape(
            cls,
            num_blocks: int,
            block_size: int,
            num_kv_heads: int,
            head_size: int,
            cache_dtype_str: str,
        ) -> tuple[int, ...]:
            return (num_blocks, block_size, num_kv_heads, head_size)

        @classmethod
        def get_impl_cls(cls) -> type:
            if not npu_available():
                raise RuntimeError(
                    f"the {cls._solution_name} backend executes Ascend NPU "
                    "kernels only; refusing to run on this device"
                )
            return _Impl

        @classmethod
        def get_builder_cls(cls) -> type:
            from vllm.v1.attention.backend import AttentionMetadataBuilder

            class _Builder(AttentionMetadataBuilder):
                pass

            return _Builder

    return HustQuantizedKvAttentionBackend


def __getattr__(name: str) -> Any:
    if name == "HustQuantizedKvAttentionBackend":
        return _build_backend()
    raise AttributeError(name)


# The backend class is exposed via module __getattr__ (lazy build), so it
# intentionally does not appear in __all__.
