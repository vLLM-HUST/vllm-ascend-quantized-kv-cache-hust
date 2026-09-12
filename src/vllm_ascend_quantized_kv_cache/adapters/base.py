# SPDX-License-Identifier: Apache-2.0
"""适配器层的共享类型。"""

from __future__ import annotations

import importlib.util
from typing import Any


class HostAdapter:
    """各宿主集成适配器的基类。

    一个适配器把一个 :class:`~methods.base.KvQuantMethod` 绑定到一个宿主栈。
    铁律：对宿主的 import 只发生在方法内部，绝不出现在模块导入期——
    因此适配器可以在任何环境构造，宿主缺失时 fail-closed 并给出
    精确信息。
    """

    host: str = ""  # 宿主名（core.hosts 里的键）
    host_module: str = ""  # 宿主顶层包名（用于可用性探测）

    def __init__(self, method: Any) -> None:
        self.method = method

    @classmethod
    def available(cls) -> bool:
        """True when the host package is importable in this interpreter."""
        try:
            return importlib.util.find_spec(cls.host_module) is not None
        except (ImportError, ValueError):  # pragma: no cover - defensive
            return False

    def require_host(self) -> None:
        if not self.available():
            raise RuntimeError(
                f"host adapter {type(self).__name__} requires "
                f"{self.host_module!r}, which is not importable in this "
                "environment"
            )

    def register(self, **options: Any) -> dict[str, Any]:
        """把方法注册进宿主；由各子类实现（见 vllm_ascend_hust / vllm_hust）。"""
        raise NotImplementedError


__all__ = ["HostAdapter"]
