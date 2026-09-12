# SPDX-License-Identifier: Apache-2.0
"""设备能力探测：供内核层的设备代码路径使用。

导入本模块永远安全（顶层没有任何重依赖）。这里的辅助函数存在的目的，
是让内核层（triton-ascend / torch_npu，仅 Ascend NPU）在非 NPU 环境下
fail-closed 并给出精确的错误信息，而不是抛出一串缺依赖的堆栈。
"""

from __future__ import annotations

import importlib
import importlib.util


def module_available(name: str) -> bool:
    """某个模块在当前解释器是否可导入（只查元数据，不真正导入）。"""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def npu_available() -> bool:
    """torch 是否能看到可用的 Ascend NPU 设备。"""
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return False
    npu = getattr(torch, "npu", None)
    try:
        return bool(npu is not None and npu.is_available())
    except Exception:  # pragma: no cover - defensive
        return False


def require_npu(what: str) -> None:
    """设备路径的统一守卫：无 NPU 时抛出带上下文的 RuntimeError。"""
    if not npu_available():
        raise RuntimeError(
            f"{what} requires an Ascend NPU device (torch_npu), but none is "
            "available. The quantized-KV kernel layer targets Ascend NPU "
            "only; the pure semantics API works on any machine."
        )


def import_torch(what: str = "this code path"):
    """惰性导入 torch；失败时错误信息带上调用方语境。"""
    try:
        return importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"{what} requires torch, which is not installed") from exc


def import_torch_npu(what: str = "this code path"):
    """惰性导入 torch_npu（仅 Ascend NPU 环境）。"""
    try:
        return importlib.import_module("torch_npu")
    except ImportError as exc:
        raise RuntimeError(
            f"{what} requires torch_npu on an Ascend NPU device; "
            "torch_npu is not importable in this environment"
        ) from exc


def import_triton(what: str = "this code path"):
    """惰性导入 triton；优先走 vllm 的 triton shim，失败再直接 import。

    triton-ascend 在 Ascend NPU 上执行 @triton.jit 内核；宿主环境
    （vllm 系）通过 vllm.triton_utils 暴露，独立环境直接装 triton 包。
    """
    try:
        triton = importlib.import_module("triton")
        tl = importlib.import_module("triton.language")
        return triton, tl
    except ImportError as exc:
        raise RuntimeError(
            f"{what} requires triton (triton-ascend on Ascend NPU), which is "
            "not importable in this environment"
        ) from exc


__all__ = [
    "import_torch",
    "import_torch_npu",
    "import_triton",
    "module_available",
    "npu_available",
    "require_npu",
]
