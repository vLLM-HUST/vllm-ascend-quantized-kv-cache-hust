# SPDX-License-Identifier: Apache-2.0
"""``vllm.general_plugins`` 动态加载入口。

安装后，vLLM 在运行时导入该入口并注册 INT8 attention backend。注册本身
不启用量化；只有用户传入 ``--kv-cache-dtype int8`` 才会实例化该后端。
"""

from __future__ import annotations

from .core.hosts import VLLM_ASCEND_HUST, detect_host

__all__ = ["detect_host", "register_plugins"]


def register_plugins() -> list[str]:
    """在 Ascend 宿主中注册插件后端；其他宿主进程保持 no-op。"""
    if detect_host() != VLLM_ASCEND_HUST:
        return []

    from .core.activation import activate

    info = activate("int8_dynamic", host=VLLM_ASCEND_HUST)
    print(
        "[vllm-ascend-int8-kv] registered dynamic INT8 attention backend; "
        "enable it with --kv-cache-dtype int8: "
        f"{info['backend_class_path']}",
        flush=True,
    )
    return ["int8_dynamic"]
