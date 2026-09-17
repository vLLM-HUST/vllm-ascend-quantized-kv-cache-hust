# SPDX-License-Identifier: Apache-2.0
"""vllm-hust 宿主的注册入口。

零宿主改动的挂载路径：本包把一个"全限定类路径"注册进宿主 attention
注册表的 ``AttentionBackendEnum.CUSTOM`` 槽位（注册表只存字符串路径，
选定后端时才惰性 import），引擎以 ``--attention-backend CUSTOM`` 生效。

注意：宿主的 CacheDType 是封闭 Literal，在三层各自 fail-closed
（pydantic 配置、后端选择器、torch dtype 查表）。方法必须复用某个
既有字面量；:func:`map_cache_dtype` 负责这个协商，没有合适字面量时
fail-closed（如 fp4_e2m1，需要宿主路线图加字面量）。
"""

from __future__ import annotations

from typing import Any

from ..base import HostAdapter

#: 方法名 -> vllm-hust 上最近的既有 CacheDType 字面量。
#: Literal set (host cache.py): auto, float16, bfloat16, fp8, fp8_e4m3,
#: fp8_e5m2, fp8_inc, fp8_ds_mla, turboquant_*, int4_per_token_head,
#: int8_per_token_head, fp8_per_token_head, nvfp4.
DTYPE_LITERAL_MAP: dict[str, str] = {
    "int8_dynamic": "int8_per_token_head",
    "kivi_int4": "int4_per_token_head",
    "int4_packed": "int4_per_token_head",
    "nvfp4": "nvfp4",
    "fp8_e4m3": "fp8_e4m3",
}


def map_cache_dtype(method_name: str) -> str:
    """为方法协商宿主可用的 CacheDType 字面量。

    Fail-closed：没有合适字面量的方法（当前是 fp4_e2m1）直接抛错，
    绝不静默套用错误布局。
    """
    try:
        return DTYPE_LITERAL_MAP[method_name]
    except KeyError:
        raise ValueError(
            f"method {method_name!r} has no vllm-hust CacheDType "
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
        """把 AttentionBackendEnum.CUSTOM 指到我们的 backend 类路径。"""
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

        method_name = self.method.name
        cache_dtype_literal = map_cache_dtype(method_name)
        register_backend(AttentionBackendEnum.CUSTOM, BACKEND_CLASS_PATH)
        return {
            "host": self.host,
            "method": method_name,
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
