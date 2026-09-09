# SPDX-License-Identifier: Apache-2.0
"""Quantized KV-cache solutions for the vLLM-HUST stack.

Unified entry point. Importing this package is inert: no torch, no vllm,
no device module, no threads. Everything heavy is loaded lazily when a
solution's semantics or a host adapter is actually used.

Quick start::

    from vllm_ascend_quantized_kv_cache import kv_solutions

    kv_solutions.list()                       # every registered solution
    kv_solutions.list(host="vllm_ascend_hust")
    sol = kv_solutions.get("kivi_int4", head_size=128, block_size=128)
    sol.resolve_layout()                      # -> KVCacheLayout contract
    sol.semantics                             # pure, CPU-testable math
    sol.host_adapter("vllm_ascend_hust").register()

Device kernels execute on Ascend NPU only; every device path fails closed
with an explicit error elsewhere. See ``docs/architecture.md`` for the
layered design and ``docs/packaging-and-release.md`` for the release flow.
"""

from ._version import __version__
from .core import KvSolution, SolutionConfig, SolutionSpec
from .core.hosts import VLLM_ASCEND_HUST, VLLM_HUST
from .core.registry import get_spec as _get_spec
from .core.registry import list_solutions as _list_solutions
from .core.registry import reset_registry as _reset_registry
from .dtypes import (
    KVCacheLayout,
    KVQuantMode,
    fp4_e2m1_packed_dim,
    get_kv_quant_mode,
    int4_packed_dim,
    is_quantized_kv_cache,
    nvfp4_packed_dim,
    resolve_layout,
)


def _load_solutions() -> None:
    """Register the built-in solutions (metadata only) on first use."""
    from . import solutions  # noqa: F401  (registration side effect)


class _KvSolutionsFacade:
    """Unified discovery / configuration / binding API."""

    @staticmethod
    def get(name: str, **config) -> KvSolution:
        """Return a configured solution handle.

        Raises ``ValueError`` for unknown names (listing the known ones)
        or invalid configuration, mirroring the fail-closed style of the
        layout contract.
        """
        _load_solutions()
        from .core.registry import get_solution

        return get_solution(name, **config)

    @staticmethod
    def list(host: str | None = None) -> list[str]:
        """Registered solution names, optionally filtered by host."""
        _load_solutions()
        return list(_list_solutions(host))

    @staticmethod
    def describe(name: str) -> dict:
        """Metadata dict for one solution (no heavy imports)."""
        _load_solutions()
        return _get_spec(name).describe()

    @staticmethod
    def describe_all(host: str | None = None) -> dict[str, dict]:
        _load_solutions()
        return {name: _get_spec(name).describe() for name in _list_solutions(host)}

    @staticmethod
    def reset() -> None:
        """Test-only: drop all registrations."""
        _reset_registry()


kv_solutions = _KvSolutionsFacade()


class VllmAscendQuantizedKvCacheContractProposal:
    """Metadata-only proposal; this class performs no runtime activation."""


__all__ = [
    "KVCacheLayout",
    "KVQuantMode",
    "KvSolution",
    "SolutionConfig",
    "SolutionSpec",
    "VLLM_ASCEND_HUST",
    "VLLM_HUST",
    "VllmAscendQuantizedKvCacheContractProposal",
    "__version__",
    "fp4_e2m1_packed_dim",
    "get_kv_quant_mode",
    "int4_packed_dim",
    "is_quantized_kv_cache",
    "kv_solutions",
    "nvfp4_packed_dim",
    "resolve_layout",
]
