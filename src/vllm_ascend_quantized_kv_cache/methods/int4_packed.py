# SPDX-License-Identifier: Apache-2.0
"""INT4 打包量化方法（per-token-head，每字节 2 个 int4）。

挖掘自 legacy ascend PR #160（格式 handler + 分发表）；与其他量化方法
一样经 ``kv_methods.get("int4_packed")`` 取用。每个 (token, head) 对
各自算 scale，实际打包在 backend 内核。
"""

from __future__ import annotations

from ..core.hosts import ALL_HOSTS
from ..dtypes import get_kv_quant_mode
from .base import MethodSpec
from .packed_base import (
    FORMAT_SEMANTICS,
    METHOD_NAME_TO_SEMANTICS,
    PackedFormatSemantics,
)
from .registry import register_method

PROVENANCE = "ascend-pr-160/0001+0004+0005+0007"

_NPU_KERNEL_NOTE = (
    "Quantization is executed by the attention backend on Ascend NPU; the "
    "semantics object only decides the storage dtype and carries scales."
)


class Int4PackedSemantics(PackedFormatSemantics):
    """稠密注意力模型的 INT4 KV 量化。"""

    scheme_key = "VLLM_HUST_KV_INT4"
    cache_dtype = "int4"
    storage_torch_dtype_name = "uint8"
    uses_scales = True


def _load_semantics(config):
    """惰性语义加载器：返回本格式的语义实例（config 不影响单例行为）。"""
    return Int4PackedSemantics()


def _make_spec() -> MethodSpec:
    """构造格式方法的 spec；适配器工厂在函数体内惰性 import 适配器模块。"""

    def ascend_adapter(method):
        from ..adapters.vllm_ascend_hust import AscendHustAdapter

        return AscendHustAdapter(method)

    def vllm_adapter(method):
        from ..adapters.vllm_hust import VllmHustAdapter

        return VllmHustAdapter(method)

    return MethodSpec(
        name="int4_packed",
        dtype="int4",
        summary=(
            "INT4 per-token-head symmetric quantization, 2x int4 per "
            f"byte. {_NPU_KERNEL_NOTE}"
        ),
        provenance=PROVENANCE,
        quant_mode=get_kv_quant_mode("int4"),
        supports=tuple(ALL_HOSTS),
        requires_npu_kernels=True,
        semantics_loader=_load_semantics,
        adapter_factories={
            "vllm_ascend_hust": ascend_adapter,
            "vllm_hust": vllm_adapter,
        },
    )


# import 本模块即完成注册（只写元数据，零重导入）
METHOD_SPEC = _make_spec()
register_method(METHOD_SPEC)

# 登记进共享分发表（供 --kv-cache-dtype 分发 / 宿主 scheme 构建使用）
FORMAT_SEMANTICS.setdefault("int4", Int4PackedSemantics)
METHOD_NAME_TO_SEMANTICS.setdefault("int4_packed", Int4PackedSemantics)

__all__ = ["METHOD_SPEC"]
