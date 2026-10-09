# SPDX-License-Identifier: Apache-2.0
"""``vllm.general_plugins`` 动态加载入口。

安装后，vLLM 在运行时导入该入口并为宿主 backend 装上量化 KV 的 impl
分派（INT8 与 INT4/KIVI 共用同一台分派器）。注册本身不启用量化；
只有用户传入 ``--kv-cache-dtype int8`` 或 ``--kv-cache-dtype kivi_int4``
才会实例化对应实现。
"""

from __future__ import annotations

from .core.hosts import VLLM_ASCEND_HUST, detect_host

#: 宿主进程加载时统一注册的方法（注册 = 装上分派器，不改变默认行为）。
REGISTERED_METHODS = ("int8_dynamic", "kivi_int4")

__all__ = ["detect_host", "register_plugins", "REGISTERED_METHODS"]


def register_plugins() -> list[str]:
    """在 Ascend 宿主中注册插件后端；其他宿主进程保持 no-op。"""
    if detect_host() != VLLM_ASCEND_HUST:
        return []

    from .core.activation import activate

    for name in REGISTERED_METHODS:
        activate(name, host=VLLM_ASCEND_HUST)
    print(
        "[vllm-ascend-quantized-kv] registered quantized KV attention backends; "
        "enable one of them with --kv-cache-dtype "
        f"{', '.join(REGISTERED_METHODS)}",
        flush=True,
    )
    return list(REGISTERED_METHODS)
