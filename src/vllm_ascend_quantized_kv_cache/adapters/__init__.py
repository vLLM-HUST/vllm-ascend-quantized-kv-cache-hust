# SPDX-License-Identifier: Apache-2.0
"""vLLM-HUST 双宿主栈的适配器层（vllm_hust / vllm_ascend_hust）。

调用方契约——**每个宿主一个入口**，按宿主栈可导入性 fail-closed：

- ``adapters/vllm_hust``：面向 **vllm-hust 栈**（register 时要求可
  import ``vllm``）。经宿主 attention registry 的
  ``AttentionBackendEnum.CUSTOM`` 类路径挂载，并为方法协商既有
  CacheDType 字面量。
- ``adapters/vllm_ascend_hust``：面向 **vllm-ascend-hust 栈**
  （register 时要求可 import ``vllm_ascend``）。经宿主
  ``@register_scheme`` 量化注册表 + C8 式 impl 类手术挂载。

两个子包的模块顶层都不 import 宿主；``register()`` 内部才惰性导入并
用 ``require_host()`` 守卫——缺对应栈时抛 RuntimeError。双栈共存的
环境（如同时装了 vllm 与 vllm_ascend 的 NPU 开发机）两者都可显式
注册，二者的注册表相互独立；默认宿主由 ``core.hosts.detect_host()``
决定（vllm_ascend 优先）。注意 ``vllm_ascend_hust.__init__`` 会连带
导入 torch（attention mixin）——只有宿主进程与本库测试应直接 import
它，上层一律走 :func:`adapter_for` 或 ``kv_methods.activate``。

推荐入口是统一激活管线（自动选层、自动探测宿主）::

    from vllm_ascend_quantized_kv_cache import kv_methods
    kv_methods.activate("int8_dynamic", host="vllm_ascend_hust")
"""

from __future__ import annotations

from .base import HostAdapter

__all__ = ["HostAdapter", "adapter_for"]


def adapter_for(host: str) -> type[HostAdapter]:
    """按宿主名取适配器类（惰性导入，未知宿主 fail-closed）。

    惰性是刻意的：``vllm_ascend_hust`` 子包会在 import 时加载
    attention mixin（torch），不应被任何"只看元数据"的路径连带拉起。
    """
    if host == "vllm_hust":
        from .vllm_hust import VllmHustAdapter

        return VllmHustAdapter
    if host == "vllm_ascend_hust":
        from .vllm_ascend_hust import AscendHustAdapter

        return AscendHustAdapter
    from ..core.hosts import ALL_HOSTS

    raise ValueError(f"unknown host {host!r}; known hosts: {', '.join(ALL_HOSTS)}")
