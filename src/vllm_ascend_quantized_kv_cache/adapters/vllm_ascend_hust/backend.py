# SPDX-License-Identifier: Apache-2.0
"""Ascend INT8 KV-cache attention backend selected by the vLLM CLI."""

from __future__ import annotations

from functools import cache
from typing import Any

_DISPATCH_MARKER = "_quantized_kv_int8_plugin_dispatch_installed"
_ORIGINAL_IMPL_GETTER = "_quantized_kv_original_get_impl_cls"


def _require_int8_cache_dtype() -> None:
    from vllm.config import get_current_vllm_config

    cache_dtype = get_current_vllm_config().cache_config.cache_dtype
    if cache_dtype != "int8":
        raise RuntimeError(
            "the quantized KV plugin backend is selected only by "
            f"--kv-cache-dtype int8; got {cache_dtype!r}"
        )


@cache
def _build_impl_cls() -> type:
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

    from ...methods.int8_dynamic.attention_backend import (
        AscendInt8AttentionBackendMixin,
    )

    class AscendInt8KvAttentionImpl(
        AscendInt8AttentionBackendMixin, AscendAttentionBackendImpl
    ):
        """Plugin-owned port bound to the live Ascend attention surface."""

    return AscendInt8KvAttentionImpl


def install_int8_impl_dispatch() -> type:
    """Route only INT8 KV cache layers to the plugin implementation.

    Ascend's platform selector returns ``AscendAttentionBackend`` directly, so
    overriding vLLM's generic CUSTOM registry slot does not affect selection.
    Patch the host backend's implementation factory instead, while delegating
    every non-INT8 configuration to the original host factory.
    """
    from vllm.config import get_current_vllm_config
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

    if getattr(AscendAttentionBackend, _DISPATCH_MARKER, False):
        return AscendAttentionBackend

    original_get_impl_cls = AscendAttentionBackend.get_impl_cls

    def get_impl_cls() -> type:
        cache_dtype = get_current_vllm_config().cache_config.cache_dtype
        if cache_dtype == "int8":
            from vllm_ascend.attention.utils import enable_cp

            if enable_cp():
                raise NotImplementedError(
                    "Ascend KV cache INT8 does not support context parallel yet."
                )
            return _build_impl_cls()
        return original_get_impl_cls()

    setattr(AscendAttentionBackend, _ORIGINAL_IMPL_GETTER, original_get_impl_cls)
    AscendAttentionBackend.get_impl_cls = staticmethod(get_impl_cls)
    setattr(AscendAttentionBackend, _DISPATCH_MARKER, True)
    return AscendAttentionBackend


def _build_backend_cls() -> type:
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

    class AscendInt8KvAttentionBackend(AscendAttentionBackend):
        """Backend activated exclusively by ``--kv-cache-dtype int8``."""

        @staticmethod
        def get_impl_cls() -> type:
            _require_int8_cache_dtype()
            return _build_impl_cls()

    return AscendInt8KvAttentionBackend


def __getattr__(name: str) -> Any:
    if name == "AscendInt8KvAttentionBackend":
        return _build_backend_cls()
    raise AttributeError(name)


__all__ = ["install_int8_impl_dispatch"]
