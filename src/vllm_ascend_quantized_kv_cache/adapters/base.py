# SPDX-License-Identifier: Apache-2.0
"""Adapter-layer shared types."""

from __future__ import annotations

import importlib.util
from typing import Any


class HostAdapter:
    """Base for per-host integration adapters.

    An adapter binds one :class:`~core.spec.KvSolution` to one host stack.
    All host imports happen inside methods, never at module import time, so
    adapters can be constructed anywhere and fail closed with precise
    messages when the host is missing.
    """

    host: str = ""
    host_module: str = ""

    def __init__(self, solution: Any) -> None:
        self.solution = solution

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
        raise NotImplementedError


__all__ = ["HostAdapter"]
