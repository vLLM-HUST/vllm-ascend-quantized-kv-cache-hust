# SPDX-License-Identifier: Apache-2.0
"""KIVI INT4 KV-cache solution (quantized history + FP residual window).

Mined from legacy ascend PR #116 commits 0003-0013. See ``semantics.py``
for the quantization math and window bookkeeping, ``attention_mixin.py``
for the NPU forward paths, and ``ops.triton.kivi_cache`` for the ported
triton-ascend kernels.
"""

from __future__ import annotations

from ...core.hosts import ALL_HOSTS
from ...core.registry import register_solution
from ...core.spec import SolutionConfig, SolutionSpec
from ...dtypes import KVQuantMode

PROVENANCE = "ascend-pr-116/0003-0013"


def _validate_config(config: SolutionConfig) -> None:
    from .semantics import validate_kivi_config

    validate_kivi_config(config)


def _load_semantics(config):
    from .semantics import KiviInt4Semantics

    return KiviInt4Semantics(config)


def _make_spec() -> SolutionSpec:
    def ascend_adapter(solution):
        from ...adapters.vllm_ascend_hust import AscendHustAdapter

        return AscendHustAdapter(solution)

    def vllm_adapter(solution):
        from ...adapters.vllm_hust import VllmHustAdapter

        return VllmHustAdapter(solution)

    return SolutionSpec(
        name="kivi_int4",
        dtype="kivi_int4",
        summary=(
            "KIVI INT4: the most recent residual window stays full precision "
            "per request; older token groups are packed into a paged int4 "
            "history cache by triton-ascend kernels and gathered + dequantized "
            "for TND fused-inference attention."
        ),
        provenance=PROVENANCE,
        quant_mode=KVQuantMode.KIVI_INT4,
        supports=tuple(ALL_HOSTS),
        requires_npu_kernels=True,
        config_validator=_validate_config,
        semantics_loader=_load_semantics,
        adapter_factories={
            "vllm_ascend_hust": ascend_adapter,
            "vllm_hust": vllm_adapter,
        },
    )


SOLUTION_SPEC = _make_spec()
register_solution(SOLUTION_SPEC)

__all__ = ["SOLUTION_SPEC"]
