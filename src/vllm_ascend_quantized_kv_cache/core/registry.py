# SPDX-License-Identifier: Apache-2.0
"""Fail-closed solution registry.

Registration is metadata-only. Unknown names raise ``ValueError`` listing
the known solutions, mirroring the fail-closed style of
``dtypes.resolve_layout``.
"""

from __future__ import annotations

from typing import Any

from .spec import KvSolution, SolutionConfig, SolutionSpec

_REGISTRY: dict[str, SolutionSpec] = {}


def register_solution(spec: SolutionSpec) -> SolutionSpec:
    if not isinstance(spec, SolutionSpec):  # pragma: no cover - defensive
        raise TypeError(f"register_solution expects a SolutionSpec, got {type(spec)!r}")
    if spec.name in _REGISTRY:
        prev = _REGISTRY[spec.name].provenance
        raise ValueError(
            f"solution {spec.name!r} is already registered (provenance {prev!r})"
        )
    _REGISTRY[spec.name] = spec
    return spec


def known_solutions() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def list_solutions(host: str | None = None) -> tuple[str, ...]:
    if host is None:
        return known_solutions()
    return tuple(name for name in known_solutions() if host in _REGISTRY[name].supports)


def get_solution(name: str, **config_overrides: Any) -> KvSolution:
    spec = _REGISTRY.get(name)
    if spec is None:
        known = ", ".join(known_solutions()) or "<none>"
        raise ValueError(f"unknown quantized KV solution {name!r}; known: {known}")
    if config_overrides:
        config = SolutionConfig(**config_overrides)
    else:
        config = SolutionConfig()
    return KvSolution(spec, config)


def get_spec(name: str) -> SolutionSpec:
    spec = _REGISTRY.get(name)
    if spec is None:
        known = ", ".join(known_solutions()) or "<none>"
        raise ValueError(f"unknown quantized KV solution {name!r}; known: {known}")
    return spec


def reset_registry() -> None:
    """Test-only helper: drop every registration."""
    _REGISTRY.clear()


__all__ = [
    "get_solution",
    "get_spec",
    "known_solutions",
    "list_solutions",
    "register_solution",
    "reset_registry",
]
