# SPDX-License-Identifier: Apache-2.0
"""``vllm.general_plugins`` bootstrap hook.

Installation must never change behaviour, so the hook registered under the
``vllm.general_plugins`` entry-point group is a **no-op by default**. It
activates solutions only when the process environment names them
explicitly::

    VLLM_HUST_QUANT_KV_SOLUTIONS=int8_dynamic,kivi_int4

Each named solution is registered into the host that is importable in the
current interpreter (``vllm_ascend`` -> the Ascend adapter, otherwise
``vllm`` -> the vllm-hust adapter). Unknown solutions or missing hosts
fail closed with an explicit error.

This native path is independent of the Extension Manager: the static
manifest stays ``import_only`` until the HOST_CONTRACT protocols land,
while operators who want the solutions today opt in per process through
this variable.
"""

from __future__ import annotations

import importlib.util
import os

ENV_SOLUTIONS = "VLLM_HUST_QUANT_KV_SOLUTIONS"


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def detect_host() -> str | None:
    """Return the host adapter name for this interpreter, or ``None``."""
    if _importable("vllm_ascend"):
        return "vllm_ascend_hust"
    if _importable("vllm"):
        return "vllm_hust"
    return None


def requested_solutions() -> list[str]:
    """Parse the opt-in environment variable (deduplicated, ordered)."""
    raw = os.environ.get(ENV_SOLUTIONS, "")
    requested: list[str] = []
    for token in raw.split(","):
        name = token.strip()
        if name and name not in requested:
            requested.append(name)
    return requested


def register_plugins() -> list[str]:
    """Entry-point hook: activate explicitly requested solutions.

    Returns the list of activated solution names (empty when the
    environment does not opt in).
    """
    requested = requested_solutions()
    if not requested:
        return []

    from .core.registry import get_solution

    host = detect_host()
    if host is None:
        raise RuntimeError(
            f"{ENV_SOLUTIONS}={','.join(requested)} requests quantized-KV "
            "solutions, but neither vllm_ascend nor vllm is importable in "
            "this process; install a host stack first"
        )

    activated: list[str] = []
    for name in requested:
        solution = get_solution(name)  # ValueError on unknown names
        adapter = solution.host_adapter(host)  # ValueError if unsupported
        info = adapter.register()
        activated.append(name)
        detail = info.get("quant_type") or info.get("attention_backend", "")
        print(
            f"[vllm-hust-quantized-kv-cache] activated solution "
            f"{name!r} on host {host!r}: {detail}",
            flush=True,
        )
    return activated


__all__ = ["ENV_SOLUTIONS", "detect_host", "register_plugins", "requested_solutions"]
