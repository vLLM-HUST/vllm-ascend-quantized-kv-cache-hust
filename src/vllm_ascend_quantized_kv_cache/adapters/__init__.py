# SPDX-License-Identifier: Apache-2.0
"""vllm-ascend-hust 量化 KV attention backend 适配器层。

模块顶层不导入宿主；``register()`` 时才惰性导入并 fail-closed。

推荐入口是统一激活管线（自动选层、自动探测宿主）::

    from vllm_ascend_quantized_kv_cache import kv_methods
    kv_methods.activate("kivi_int4", host="vllm_ascend_hust")
"""

from __future__ import annotations

from .base import HostAdapter

__all__ = ["HostAdapter", "adapter_for"]


def adapter_for(host: str) -> type[HostAdapter]:
    """按宿主名取适配器类（惰性导入，未知宿主 fail-closed）。

    惰性是刻意的：``vllm_ascend_hust`` 子包会在 import 时加载
    attention mixin（torch），不应被任何"只看元数据"的路径连带拉起。
    """
    if host == "vllm_ascend_hust":
        from .vllm_ascend_hust import AscendHustAdapter

        return AscendHustAdapter
    from ..core.hosts import ALL_HOSTS

    raise ValueError(f"unknown host {host!r}; known hosts: {', '.join(ALL_HOSTS)}")
