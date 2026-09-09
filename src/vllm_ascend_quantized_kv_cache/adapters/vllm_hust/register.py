# SPDX-License-Identifier: Apache-2.0
"""Registration entry point for the vllm-hust host.

Zero-host-change attach path: the external package registers a fully
qualified class path into the host attention registry under
``AttentionBackendEnum.CUSTOM`` (the registry stores string paths and
imports them lazily at selection time), then the engine picks it up with
``--attention-backend CUSTOM``.

``CacheDType`` on the host is a closed Literal enforced at three layers
(pydantic config, backend selector, torch-dtype lookup), so a solution
must reuse an existing dtype literal; :func:`map_cache_dtype` provides
that negotiation and fails closed when no literal fits.
"""

from __future__ import annotations

from typing import Any

from ..base import HostAdapter

#: Solution name -> nearest existing ``CacheDType`` literal on vllm-hust.
#: Literal set (host cache.py): auto, float16, bfloat16, fp8, fp8_e4m3,
#: fp8_e5m2, fp8_inc, fp8_ds_mla, turboquant_*, int4_per_token_head,
#: int8_per_token_head, fp8_per_token_head, nvfp4.
DTYPE_LITERAL_MAP: dict[str, str] = {
    "int8_dynamic": "int8_per_token_head",
    "kivi_int4": "int4_per_token_head",
    "int4": "int4_per_token_head",
    "nvfp4": "nvfp4",
    "fp8_e4m3": "fp8_e4m3",
}


def map_cache_dtype(solution_name: str) -> str:
    """Negotiate the host ``CacheDType`` literal for a solution.

    Fail-closed: solutions without a fitting literal (fp4_e2m1 today) raise
    instead of silently reusing a wrong layout.
    """
    try:
        return DTYPE_LITERAL_MAP[solution_name]
    except KeyError:
        raise ValueError(
            f"solution {solution_name!r} has no vllm-hust CacheDType "
            "literal mapping yet; adding one requires the host-side "
            "roadmap item (see docs/architecture.md)"
        ) from None


BACKEND_CLASS_PATH = (
    "vllm_ascend_quantized_kv_cache.adapters.vllm_hust.backend:"
    "HustQuantizedKvAttentionBackend"
)


class VllmHustAdapter(HostAdapter):
    host = "vllm_hust"
    host_module = "vllm"

    def register(self) -> dict[str, Any]:
        """Point ``AttentionBackendEnum.CUSTOM`` at our backend class path."""
        self.require_host()
        try:
            from vllm.v1.attention.backends.registry import (
                AttentionBackendEnum,
                register_backend,
            )
        except ImportError as exc:
            raise RuntimeError(
                "vllm.v1.attention.backends.registry is unavailable; this "
                "adapter targets the vllm-hust fork"
            ) from exc

        solution_name = self.solution.name
        cache_dtype_literal = map_cache_dtype(solution_name)
        register_backend(AttentionBackendEnum.CUSTOM, BACKEND_CLASS_PATH)
        return {
            "host": self.host,
            "solution": solution_name,
            "attention_backend": "CUSTOM",
            "cache_dtype_literal": cache_dtype_literal,
            "backend_class_path": BACKEND_CLASS_PATH,
            "usage": (
                "start the engine with --attention-backend CUSTOM and "
                f"--kv-cache-dtype {cache_dtype_literal}; device "
                "execution runs on Ascend NPU and fails closed elsewhere"
            ),
        }


__all__ = [
    "BACKEND_CLASS_PATH",
    "DTYPE_LITERAL_MAP",
    "VllmHustAdapter",
    "map_cache_dtype",
]
