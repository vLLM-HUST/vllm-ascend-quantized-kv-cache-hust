# SPDX-License-Identifier: Apache-2.0
"""宿主名单、宿主探测与激活入口（横切层，任何进程可导入）。

宿主 = 我们要插拔进去的推理栈。两个宿主都支持零宿主改动的外部挂载：
  - vllm_ascend_hust: 通过宿主的 @register_scheme 量化注册表 + C8 式
    layer.impl 类手术挂载；
  - vllm_hust: 通过宿主 attention registry 的 CUSTOM 类路径挂载。

本模块是"哪个宿主进程能调什么"的权威出处：宿主探测（detect_host）
与激活守卫（require_host_stack）都定义在这里，bootstrap 钩子与
kv_methods.activate 共用同一条激活管线。
"""

from __future__ import annotations

import importlib.util

VLLM_HUST = "vllm_hust"  # vLLM-HUST 核心仓（vllm fork）
VLLM_ASCEND_HUST = "vllm_ascend_hust"  # vLLM-Ascend-HUST NPU 后端仓

ALL_HOSTS = (VLLM_HUST, VLLM_ASCEND_HUST)


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def detect_host() -> str | None:
    """探测当前解释器里可导入的宿主栈；两者都缺时返回 None。

    探测顺序：``vllm_ascend``（vllm-ascend-hust，NPU 栈）优先，
    其次 ``vllm``（vllm-hust 核心栈）。只查元数据，不真正导入。
    """
    if _importable("vllm_ascend"):
        return VLLM_ASCEND_HUST
    if _importable("vllm"):
        return VLLM_HUST
    return None


def require_host_stack(what: str) -> str:
    """激活路径的统一守卫：返回检测到的宿主名，否则抛带语境的 RuntimeError。"""
    host = detect_host()
    if host is None:
        raise RuntimeError(
            f"{what} requests quantized-KV methods, but neither "
            "vllm_ascend nor vllm is importable in this process; "
            "install a host stack first"
        )
    return host


__all__ = [
    "ALL_HOSTS",
    "VLLM_ASCEND_HUST",
    "VLLM_HUST",
    "detect_host",
    "require_host_stack",
]
