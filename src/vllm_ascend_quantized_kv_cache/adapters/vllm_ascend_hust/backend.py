# SPDX-License-Identifier: Apache-2.0
"""Ascend KV-cache attention backends selected by the vLLM CLI.

One dispatcher serves every quantized KV method the plugin ships: the cache
dtype literal on the command line picks the implementation class, and every
other dtype keeps the host's original behaviour untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any

_DISPATCH_MARKER = "_quantized_kv_plugin_dispatch_installed"
_ORIGINAL_IMPL_GETTER = "_quantized_kv_original_get_impl_cls"

# CLI cache dtype literals the plugin owns.
_INT8_CACHE_DTYPE = "int8"
_KIVI_CACHE_DTYPE = "kivi_int4"


def _require_cache_dtype(expected: str) -> None:
    from vllm.config import get_current_vllm_config

    cache_dtype = get_current_vllm_config().cache_config.cache_dtype
    if cache_dtype != expected:
        raise RuntimeError(
            f"the quantized KV plugin backend {expected!r} is selected only by "
            f"--kv-cache-dtype {expected}; got {cache_dtype!r}"
        )


def _host_attention_v1():
    """``vllm_ascend.attention.attention_v1``, regardless of import order.

    On vllm-ascend-hust ``17ed0571d`` that module cannot be the *first*
    ``vllm_ascend`` import: it goes through ``device_op`` into ``ops`` and back,
    which raises ``ImportError: cannot import name 'DeviceOperator' from
    partially initialized module``. Importing ``ops`` first breaks the cycle, so
    registration works even in a process where the host's attention stack has
    not been touched yet (``vllm.plugins.load_general_plugins()`` alone).
    """
    import importlib

    import vllm_ascend.ops  # noqa: F401

    return importlib.import_module("vllm_ascend.attention.attention_v1")


def _context_parallel_enabled() -> bool:
    """Ask the host whether context parallel is on, across host revisions.

    The INT8 release targeted a host with a single ``enable_cp()``; the 910B2
    container's host checkout (vllm-ascend-hust ``17ed0571d``) dropped it in
    favour of ``enable_dcp()`` / ``enable_pcp()``, where a bare
    ``from ... import enable_cp`` raises ImportError.
    """
    from vllm_ascend.attention import utils as attention_utils

    enable_cp = getattr(attention_utils, "enable_cp", None)
    if enable_cp is not None:
        return bool(enable_cp())

    split = [
        getattr(attention_utils, name)
        for name in ("enable_dcp", "enable_pcp")
        if hasattr(attention_utils, name)
    ]
    if not split:
        raise RuntimeError(
            "Cannot detect Ascend context parallel: vllm_ascend.attention.utils "
            "exposes neither enable_cp() nor enable_dcp()/enable_pcp()."
        )
    return any(bool(fn()) for fn in split)


@cache
def _build_int8_impl_cls() -> type:
    AscendAttentionBackendImpl = _host_attention_v1().AscendAttentionBackendImpl

    from ...methods.int8_dynamic.attention_backend import (
        AscendInt8AttentionBackendMixin,
    )

    class AscendInt8KvAttentionImpl(
        AscendInt8AttentionBackendMixin, AscendAttentionBackendImpl
    ):
        """Plugin-owned port bound to the live Ascend attention surface."""

    return AscendInt8KvAttentionImpl


@cache
def _build_kivi_impl_cls() -> type:
    """Compose the INT4 mixin over the live host implementation.

    KIVI state is initialised after the host constructor because it reads the
    ``kivi_group_size`` / ``kivi_residual_length`` knobs off ``vllm_config``.
    Only the ``kivi_int4`` cache dtype reaches this builder, so the literal is
    asserted by construction instead of sniffed out of the host's arguments.
    """
    AscendAttentionBackendImpl = _host_attention_v1().AscendAttentionBackendImpl

    from ...methods.kivi_int4.attention_backend import (
        AscendKiviInt4AttentionBackendMixin,
    )

    class AscendKiviInt4KvAttentionImpl(
        AscendKiviInt4AttentionBackendMixin, AscendAttentionBackendImpl
    ):
        """Plugin-owned INT4 (KIVI) attention impl on the host surface."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._init_kivi_state(_KIVI_CACHE_DTYPE, getattr(self, "vllm_config", None))

    return AscendKiviInt4KvAttentionImpl


_IMPL_BUILDERS: dict[str, Callable[[], type]] = {
    _INT8_CACHE_DTYPE: _build_int8_impl_cls,
    _KIVI_CACHE_DTYPE: _build_kivi_impl_cls,
}


def install_kv_impl_dispatch() -> type:
    """Route the plugin's quantized KV cache dtypes to plugin implementations.

    Ascend's platform selector returns ``AscendAttentionBackend`` directly, so
    overriding vLLM's generic CUSTOM registry slot does not affect selection.
    Patch the host backend's implementation factory instead, while delegating
    every other configuration to the original host factory.
    """
    from vllm.config import get_current_vllm_config

    AscendAttentionBackend = _host_attention_v1().AscendAttentionBackend

    if getattr(AscendAttentionBackend, _DISPATCH_MARKER, False):
        return AscendAttentionBackend

    original_get_impl_cls = AscendAttentionBackend.get_impl_cls

    def get_impl_cls() -> type:
        cache_dtype = get_current_vllm_config().cache_config.cache_dtype
        builder = _IMPL_BUILDERS.get(cache_dtype)
        if builder is None:
            return original_get_impl_cls()
        if _context_parallel_enabled():
            raise NotImplementedError(
                f"Ascend KV cache {cache_dtype} does not support context parallel yet."
            )
        return builder()

    setattr(AscendAttentionBackend, _ORIGINAL_IMPL_GETTER, original_get_impl_cls)
    AscendAttentionBackend.get_impl_cls = staticmethod(get_impl_cls)
    setattr(AscendAttentionBackend, _DISPATCH_MARKER, True)
    return AscendAttentionBackend


def _build_backend_cls(cache_dtype: str, class_name: str) -> type:
    AscendAttentionBackend = _host_attention_v1().AscendAttentionBackend

    class AscendKvAttentionBackend(AscendAttentionBackend):
        """Backend activated exclusively by one ``--kv-cache-dtype`` literal."""

        @staticmethod
        def get_impl_cls() -> type:
            _require_cache_dtype(cache_dtype)
            return _IMPL_BUILDERS[cache_dtype]()

    AscendKvAttentionBackend.__name__ = class_name
    AscendKvAttentionBackend.__qualname__ = class_name
    AscendKvAttentionBackend.__doc__ = (
        f"Backend activated exclusively by ``--kv-cache-dtype {cache_dtype}``."
    )
    return AscendKvAttentionBackend


_BACKENDS_BY_CLASS_NAME = {
    "AscendInt8KvAttentionBackend": _INT8_CACHE_DTYPE,
    "AscendKiviInt4KvAttentionBackend": _KIVI_CACHE_DTYPE,
}


def __getattr__(name: str) -> Any:
    cache_dtype = _BACKENDS_BY_CLASS_NAME.get(name)
    if cache_dtype is not None:
        return _build_backend_cls(cache_dtype, name)
    raise AttributeError(name)


__all__ = [
    "install_kv_impl_dispatch",
]
