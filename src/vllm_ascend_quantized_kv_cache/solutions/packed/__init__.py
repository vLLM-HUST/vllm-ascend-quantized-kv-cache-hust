# SPDX-License-Identifier: Apache-2.0
"""Packed KV-cache solutions: int4 / fp4_e2m1 / fp8_e4m3 / nvfp4.

Registering these solutions imports metadata only — ``torch`` and the host
stacks are imported lazily when semantics or adapters are actually used.
"""

from __future__ import annotations

from ...core.hosts import ALL_HOSTS
from ...core.registry import register_solution
from ...core.spec import SolutionSpec
from ...dtypes import get_kv_quant_mode

PROVENANCE = "ascend-pr-160/0001+0004+0005+0007"

_SUMMARIES = {
    "int4": "INT4 per-token-head symmetric quantization, 2x int4 per byte.",
    "fp4_e2m1": "FP4 E2M1 microscaling (MXFP4): 16-element blocks with one fp8 scale.",
    "fp8_e4m3": "FP8 E4M3 per-tensor scaling (static or dynamic scales).",
    "nvfp4": "NVFP4: packed fp4 data plus fp8 block scales (16-element blocks).",
}

_NPU_KERNEL_NOTE = (
    "Quantization is executed by the attention backend on Ascend NPU; the "
    "handler itself only decides the storage dtype and carries scales."
)


def _make_spec(name: str) -> SolutionSpec:
    def semantics_loader(config):
        from .schemes import get_packed_scheme

        return get_packed_scheme(name)

    def ascend_adapter(solution):
        from ...adapters.vllm_ascend_hust import AscendHustAdapter

        return AscendHustAdapter(solution)

    def vllm_adapter(solution):
        from ...adapters.vllm_hust import VllmHustAdapter

        return VllmHustAdapter(solution)

    return SolutionSpec(
        name=name,
        dtype=name,
        summary=f"{_SUMMARIES[name]} {_NPU_KERNEL_NOTE}",
        provenance=PROVENANCE,
        quant_mode=get_kv_quant_mode(name),
        supports=tuple(ALL_HOSTS),
        requires_npu_kernels=True,
        semantics_loader=semantics_loader,
        adapter_factories={
            "vllm_ascend_hust": ascend_adapter,
            "vllm_hust": vllm_adapter,
        },
    )


SOLUTION_SPECS: tuple[SolutionSpec, ...] = tuple(
    _make_spec(name) for name in ("int4", "fp4_e2m1", "fp8_e4m3", "nvfp4")
)

for _spec in SOLUTION_SPECS:
    register_solution(_spec)

__all__ = ["SOLUTION_SPECS"]
