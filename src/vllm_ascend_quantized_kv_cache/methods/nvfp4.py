# SPDX-License-Identifier: Apache-2.0
"""NVFP4 量化方法（fp4 数据 + fp8 块 scale）。

挖掘自 legacy ascend PR #160（格式 handler + 分发表）；与其他量化方法
一样经 ``kv_methods.get("nvfp4")`` 取用。每 16 个 fp4 元素共享一个
fp8 scale。
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


class NVFP4Semantics(PackedFormatSemantics):
    """稠密注意力模型的 NVFP4 KV 量化。"""

    scheme_key = "VLLM_HUST_KV_NVFP4"
    cache_dtype = "nvfp4"
    storage_torch_dtype_name = "uint8"
    uses_scales = False


def _load_semantics(config):
    """惰性语义加载器：返回本格式的语义实例（config 不影响单例行为）。"""
    return NVFP4Semantics()


def _make_spec() -> MethodSpec:
    """构造格式方法的 spec；适配器工厂在函数体内惰性 import 适配器模块。"""

    def ascend_adapter(method):
        from ..adapters.vllm_ascend_hust import AscendHustAdapter

        return AscendHustAdapter(method)

    def vllm_adapter(method):
        from ..adapters.vllm_hust import VllmHustAdapter

        return VllmHustAdapter(method)

    return MethodSpec(
        name="nvfp4",
        dtype="nvfp4",
        summary=(
            "NVFP4: packed fp4 data plus fp8 block scales "
            f"(16-element blocks). {_NPU_KERNEL_NOTE}"
        ),
        provenance=PROVENANCE,
        quant_mode=get_kv_quant_mode("nvfp4"),
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
FORMAT_SEMANTICS.setdefault("nvfp4", NVFP4Semantics)
METHOD_NAME_TO_SEMANTICS.setdefault("nvfp4", NVFP4Semantics)

__all__ = ["METHOD_SPEC"]
