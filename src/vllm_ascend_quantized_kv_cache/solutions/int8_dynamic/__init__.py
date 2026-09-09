# SPDX-License-Identifier: Apache-2.0
"""Dynamic per-channel INT8 KV-cache solution.

Mined from legacy ascend PR #116 commit 0001. See ``semantics.py`` for the
quantization math and ``attention_mixin.py`` for the NPU forward paths.
"""

from __future__ import annotations

from ...core.hosts import ALL_HOSTS
from ...core.registry import register_solution
from ...core.spec import SolutionConfig, SolutionSpec
from ...dtypes import KVQuantMode

PROVENANCE = "ascend-pr-116/0001"


def _validate_config(config: SolutionConfig) -> None:
    # Dynamic per-channel INT8 stores full-precision-width int8 values, so
    # there are no packing constraints; head_size parity is all we ask.
    if config.head_size % 8:
        raise ValueError(
            "int8_dynamic expects head_size divisible by 8 for the NPU fused "
            f"attention path, got {config.head_size}"
        )


def _load_semantics(config):
    from .semantics import Int8DynamicSemantics

    return Int8DynamicSemantics(config)


def _make_spec() -> SolutionSpec:
    def ascend_adapter(solution):
        from ...adapters.vllm_ascend_hust import AscendHustAdapter

        return AscendHustAdapter(solution)

    def vllm_adapter(solution):
        from ...adapters.vllm_hust import VllmHustAdapter

        return VllmHustAdapter(solution)

    return SolutionSpec(
        name="int8_dynamic",
        # The legacy patch keyed on kv_cache_dtype == "int8"; the layout
        # contract resolves the same int8 storage through the per-token-head
        # mode, which is the closest registered contract for dynamic scales.
        dtype="int8_per_token_head",
        summary=(
            "Dynamic per-channel INT8 KV cache: amax over the token dim on "
            "the first prefill, symmetric zero offset, online antiquant on "
            "the NPU fused-inference attention path."
        ),
        provenance=PROVENANCE,
        quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
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
