# SPDX-License-Identifier: Apache-2.0
"""统一激活管线：方法名 → 宿主适配器 → register()。

这是"宿主进程如何点亮一个量化方法"的唯一路径，两处入口共用：
  - 门面 ``kv_methods.activate(name, host=...)``（显式、可编程）；
  - ``bootstrap.register_plugins``（``vllm.general_plugins`` 钩子，
    环境变量 ``VLLM_HUST_KV_METHODS`` opt-in）。

fail-closed 语义：未知名、非法配置、未知宿主、宿主栈缺失、宿主不
支持/未接线，全部立即抛 ValueError / RuntimeError，绝不静默跳过。
"""

from __future__ import annotations

from typing import Any


def activate(name: str, *, host: str | None = None, **config: Any) -> dict[str, Any]:
    """把方法 *name* 注册进 *host*（None 时自动探测宿主栈）。

    返回适配器给出的注册信息 dict（附 ``method`` 与 ``host`` 两个
    键）。**config** 透传给 :class:`~methods.base.MethodConfig` 做
    构造期校验。
    """
    # 延迟导入：首次激活才触发方法注册与适配器模块加载。
    from ..methods.registry import get_method
    from .hosts import ALL_HOSTS, require_host_stack

    if host is None:
        host = require_host_stack(f"activating method {name!r}")
    elif host not in ALL_HOSTS:
        raise ValueError(f"unknown host {host!r}; known hosts: {', '.join(ALL_HOSTS)}")

    method = get_method(name, **config)  # ValueError：未知名 / 非法配置
    adapter = method.host_adapter(host)  # ValueError：不支持 / 未接线
    info = dict(adapter.register())
    info.setdefault("host", host)
    info["method"] = name
    return info


__all__ = ["activate"]
