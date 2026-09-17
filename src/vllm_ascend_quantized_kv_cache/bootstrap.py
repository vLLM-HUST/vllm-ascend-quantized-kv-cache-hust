# SPDX-License-Identifier: Apache-2.0
"""``vllm.general_plugins`` 引导钩子（默认 no-op）。

铁律：安装绝不改变行为。注册在 vllm.general_plugins entry point 下的
这个钩子默认什么都不做；只有进程环境变量显式点名方法时才激活：

    VLLM_HUST_KV_METHODS=int8_dynamic,kivi_int4

每个被点名的方法经**统一激活管线**（``core.activation.activate``）
注册进"当前解释器可导入的那个宿主"（能 import vllm_ascend -> 走
Ascend 适配器；否则 import vllm -> 走 vllm-hust 适配器）。未知方法 /
缺宿主都 fail-closed 并给出明确错误。

这条 native 路径独立于 Extension Manager：静态 manifest 在 HOST_CONTRACT
四协议落地前保持 import_only；今天就想用方法的运维，按进程通过这个
环境变量显式选择加入（opt-in）。
"""

from __future__ import annotations

import os

from .core.hosts import detect_host  # noqa: F401  (兼容再导出；见 __all__)

ENV_KV_METHODS = "VLLM_HUST_KV_METHODS"

# 兼容导出：宿主探测的规范出处已移到 core.hosts。
__all__ = [
    "ENV_KV_METHODS",
    "detect_host",
    "register_plugins",
    "requested_methods",
]


def requested_methods() -> list[str]:
    """解析 opt-in 环境变量（去重、保序、容忍空白）。"""
    raw = os.environ.get(ENV_KV_METHODS, "")
    requested: list[str] = []
    for token in raw.split(","):
        name = token.strip()
        if name and name not in requested:
            requested.append(name)
    return requested


def register_plugins() -> list[str]:
    """entry point 钩子：激活被显式点名的方法。

    返回激活成功的方法名列表；环境变量未设置时返回空表（no-op）。
    """
    requested = requested_methods()
    if not requested:
        return []

    from .core.activation import activate

    # 宿主探测在本模块命名空间内完成（便于测试替身），随后把确定的
    # host 交给统一激活管线——本钩子不复刻任何注册逻辑。
    host = detect_host()
    if host is None:
        raise RuntimeError(
            f"{ENV_KV_METHODS}={','.join(requested)} requests quantized-KV "
            "methods, but neither vllm_ascend nor vllm is importable in "
            "this process; install a host stack first"
        )

    activated: list[str] = []
    for name in requested:
        info = activate(name, host=host)
        activated.append(name)
        detail = info.get("quant_type") or info.get("attention_backend", "")
        print(
            f"[vllm-hust-quantized-kv-cache] activated method "
            f"{name!r} on host {host!r}: {detail}",
            flush=True,
        )
    return activated
