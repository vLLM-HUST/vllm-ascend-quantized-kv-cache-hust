# SPDX-License-Identifier: Apache-2.0
"""Core solution model: descriptors, configs, and the solution handle.

This module is intentionally dependency-free (no torch, no vllm) so that
importing the package stays inert in any environment.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from ..dtypes import KVCacheLayout, KVQuantMode, resolve_layout
from .hosts import ALL_HOSTS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .registry import KvSolution


@dataclass(frozen=True)
class SolutionConfig:
    """Validated configuration shared by all solutions.

    Values mirror the knobs the legacy implementations read from the host
    ``cache_config``: ``kivi_group_size`` / ``kivi_residual_length`` for the
    KIVI family, and the always-required geometry fields for layout math.
    """

    head_size: int = 128
    num_kv_heads: int = 8
    num_heads: int = 32
    block_size: int = 128
    group_size: int = 128
    residual_length: int = 128

    def validated(self) -> SolutionConfig:
        if self.head_size <= 0:
            raise ValueError(f"head_size must be positive, got {self.head_size}")
        if self.num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {self.num_kv_heads}")
        if self.num_heads < self.num_kv_heads:
            raise ValueError(
                "num_heads must be >= num_kv_heads, got "
                f"{self.num_heads} < {self.num_kv_heads}"
            )
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        return self


@dataclass(frozen=True)
class SolutionSpec:
    """Static description of a quantized-KV solution.

    A spec carries metadata only: constructing or registering one never
    imports torch, vllm, or any device module. Heavy modules are referenced
    through the lazy ``semantics_loader`` / adapter factories.
    """

    name: str
    dtype: str
    summary: str
    provenance: str
    quant_mode: KVQuantMode
    supports: tuple[str, ...]
    requires_npu_kernels: bool = True
    config_validator: Callable[[SolutionConfig], None] | None = None
    semantics_loader: Callable[[], Any] | None = None
    adapter_factories: Mapping[str, Callable[[KvSolution], Any]] = field(
        default_factory=dict
    )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "quant_mode": int(self.quant_mode),
            "summary": self.summary,
            "provenance": self.provenance,
            "requires_npu_kernels": self.requires_npu_kernels,
            "supports": list(self.supports),
        }


class KvSolution:
    """Unified handle returned by :func:`core.registry.get_solution`.

    The same object exposes the layout contract, the pure semantics module,
    and the per-host adapters, so callers never need to know which legacy
    implementation a solution was mined from.
    """

    def __init__(self, spec: SolutionSpec, config: SolutionConfig) -> None:
        if spec.config_validator is not None:
            spec.config_validator(config)
        self.spec = spec
        self.config = config.validated()

    # -- introspection -----------------------------------------------------

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def descriptor(self) -> dict[str, Any]:
        payload = self.spec.describe()
        payload["config"] = {
            f: getattr(self.config, f)
            for f in (
                "head_size",
                "num_kv_heads",
                "num_heads",
                "block_size",
                "group_size",
                "residual_length",
            )
        }
        return payload

    def supports(self, host: str) -> bool:
        return host in self.spec.supports

    # -- contracts ---------------------------------------------------------

    def resolve_layout(self) -> KVCacheLayout:
        return resolve_layout(self.spec.dtype, self.config.head_size)

    # -- semantics ---------------------------------------------------------

    @property
    def semantics(self) -> Any:
        if self.spec.semantics_loader is None:
            raise ValueError(f"solution {self.spec.name!r} exposes no semantics module")
        return self.spec.semantics_loader(self.config)

    # -- host adapters -----------------------------------------------------

    def host_adapter(self, host: str) -> Any:
        if host not in ALL_HOSTS:
            raise ValueError(
                f"unknown host {host!r}; known hosts: {', '.join(ALL_HOSTS)}"
            )
        if not self.supports(host):
            raise ValueError(
                f"solution {self.spec.name!r} does not support host {host!r} "
                f"(supports: {', '.join(self.spec.supports) or 'none'})"
            )
        factory = self.spec.adapter_factories.get(host)
        if factory is None:
            raise ValueError(
                f"solution {self.spec.name!r} has no adapter wired for host "
                f"{host!r} yet"
            )
        return factory(self)

    def with_config(self, **overrides: Any) -> KvSolution:
        return KvSolution(self.spec, replace(self.config, **overrides))


__all__ = ["KvSolution", "SolutionConfig", "SolutionSpec"]
