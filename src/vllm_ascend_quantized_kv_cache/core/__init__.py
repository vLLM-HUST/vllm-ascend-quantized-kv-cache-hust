# SPDX-License-Identifier: Apache-2.0
"""Core layer of the quantized-KV solution library."""

from .hosts import ALL_HOSTS, VLLM_ASCEND_HUST, VLLM_HUST
from .registry import (
    get_solution,
    get_spec,
    known_solutions,
    list_solutions,
    register_solution,
)
from .spec import KvSolution, SolutionConfig, SolutionSpec

__all__ = [
    "ALL_HOSTS",
    "KvSolution",
    "SolutionConfig",
    "SolutionSpec",
    "VLLM_ASCEND_HUST",
    "VLLM_HUST",
    "get_solution",
    "get_spec",
    "known_solutions",
    "list_solutions",
    "register_solution",
]
