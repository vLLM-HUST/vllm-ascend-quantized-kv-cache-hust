# SPDX-License-Identifier: Apache-2.0
"""vllm-hust 宿主的惰性构建 attention backend。

宿主注册表存的是"全限定类路径"，选定后端时才惰性 import。因此本模块
通过模块级 ``__getattr__`` 暴露 ``HustQuantizedKvAttentionBackend``：
这个类（必须是宿主 AttentionBackend 的子类）只在 vllm 进程真正解析
该路径的那一刻才被构建——包在其他所有环境保持零依赖。

成熟度：接口就绪的脚手架。类实现了注册表面（名字、支持的 dtype、
缓存形状，来自布局契约），并把方法 mixin 绑定到 Ascend NPU 设备执行。
metadata builder 委托宿主的通用 builder；经此后端的端到端 serving
属于宿主集成路线图项，本版本未验证。
"""

from __future__ import annotations

from typing import Any

_METHOD_NAME = "int8_dynamic"


def _resolve_method_name() -> str:
    """选择该 backend 服务的方法（环境变量可覆盖；默认 int8_dynamic）。"""
    import os

    return os.environ.get("VLLM_HUST_KV_BACKEND_METHOD", _METHOD_NAME)


def _build_backend() -> type:
    from vllm.v1.attention.backend import (
        AttentionBackend,
        AttentionImpl,
    )

    from ....core.runtime import npu_available
    from ....methods.int8_dynamic.attention_mixin import (
        Int8DynamicAttentionMixin,
    )
    from ....methods.kivi_int4.attention_mixin import KiviInt4AttentionMixin
    from .register import map_cache_dtype

    method_name = _resolve_method_name()
    cache_dtype_literal = map_cache_dtype(method_name)
    mixin = {
        "int8_dynamic": Int8DynamicAttentionMixin,
        "kivi_int4": KiviInt4AttentionMixin,
    }[method_name]

    class _Impl(mixin, AttentionImpl):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            AttentionImpl.__init__(self, *args, **kwargs)
            kv_cache_dtype = kwargs.get("kv_cache_dtype")
            if kv_cache_dtype is None and len(args) > 6:
                kv_cache_dtype = args[6]
            if method_name == "int8_dynamic":
                self._init_int8_dynamic_state(kv_cache_dtype or cache_dtype_literal)
            else:
                self._init_kivi_state(
                    kv_cache_dtype or cache_dtype_literal,
                    getattr(self, "vllm_config", None),
                )

    class HustQuantizedKvAttentionBackend(AttentionBackend):  # noqa: N801
        _method_name = method_name
        _cache_dtype_literal = cache_dtype_literal

        @classmethod
        def get_name(cls) -> str:
            return f"VLLM_HUST_QUANTIZED_KV_{method_name.upper()}"

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
            # 设备执行只走 Ascend NPU 内核；非 NPU 环境拒绝启动
            if not npu_available():
                raise RuntimeError(
                    f"the {cls._method_name} backend executes Ascend NPU "
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
