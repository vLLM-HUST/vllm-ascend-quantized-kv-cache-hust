# SPDX-License-Identifier: Apache-2.0
"""Runtime capability probes for device code paths.

Importing this module is safe everywhere. The helpers exist so that the
kernel layer (triton-ascend / torch_npu, Ascend NPU only) can fail closed
with a precise message instead of a stack trace from a missing dependency.
"""

from __future__ import annotations

import importlib
import importlib.util


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def npu_available() -> bool:
    """True when torch sees an Ascend NPU device."""
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
    if not npu_available():
        raise RuntimeError(
            f"{what} requires an Ascend NPU device (torch_npu), but none is "
            "available. The quantized-KV kernel layer targets Ascend NPU "
            "only; the pure semantics API works on any machine."
        )


def import_torch(what: str = "this code path"):
    try:
        return importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"{what} requires torch, which is not installed") from exc


def import_torch_npu(what: str = "this code path"):
    try:
        return importlib.import_module("torch_npu")
    except ImportError as exc:
        raise RuntimeError(
            f"{what} requires torch_npu on an Ascend NPU device; "
            "torch_npu is not importable in this environment"
        ) from exc


def import_triton(what: str = "this code path"):
    """Resolve triton, preferring the vllm shim when the host provides it."""
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
