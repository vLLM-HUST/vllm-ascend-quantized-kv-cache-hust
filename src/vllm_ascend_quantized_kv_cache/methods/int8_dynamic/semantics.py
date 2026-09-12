# SPDX-License-Identifier: Apache-2.0
"""动态 per-channel INT8 方案的纯数学语义。

挖掘自 legacy ascend PR #116 提交 0001（``_calc_int8_scales`` 与
``_quantize_kv_to_int8``）。全部是普通 torch 运算，CPU 张量同样能跑，
因此数值可以脱离 NPU 做单元测试。

量化语义：首次 prefill 时沿 token 维（dim=0）取 amax，得到每个
(kv_head, head_dim) 通道对一个 scale（keepdim 后形状 [1, H, D]）；
``inv_scale = 127 / amax``，offset 恒为零（对称量化）。
量化公式：``clamp(round(x * inv_scale + offset), -128, 127)``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...core.runtime import import_torch


@dataclass
class Int8DynamicScales:
    """一个注意力层的在线 per-channel scale 集合。

    ``*_inv_scale`` / ``*_offset`` 与 amax 同形：[1, H, D]；
    ``*_aq_*`` 是同一组值按 BNSD 重排后的视图（[1, H, 1, D]），供 NPU
    fused-inference 算子的 antiquant 参数使用。
    """

    k_inv_scale: Any  # K 的 1/scale（量化时乘）
    k_offset: Any  # K 的零偏移（对称量化恒为 0）
    v_inv_scale: Any
    v_offset: Any
    k_aq_scale: Any  # K 的反量化 scale，BNSD 视图
    k_aq_offset: Any
    v_aq_scale: Any
    v_aq_offset: Any


class Int8DynamicSemantics:
    """INT8 动态量化的纯语义对象（宿主无关、可测试）。"""

    def __init__(self, config: Any) -> None:
        # config 需要 num_kv_heads / head_size 两个属性；
        # 传 MethodConfig 或宿主 impl 对象都可以。
        self.config = config

    def calc_scales(self, key: Any, value: Any) -> Int8DynamicScales:
        """与 legacy 实现逐步一致的在线 scale 计算。"""
        torch = import_torch("Int8DynamicSemantics.calc_scales")
        # amax 沿 token 维；clamp 防止全零通道除零
        k_max = key.abs().amax(dim=0, keepdim=True).clamp(min=1e-12)
        v_max = value.abs().amax(dim=0, keepdim=True).clamp(min=1e-12)
        k_inv_scale = 127.0 / k_max
        k_offset = torch.zeros_like(k_inv_scale)
        v_inv_scale = 127.0 / v_max
        v_offset = torch.zeros_like(v_inv_scale)
        # BNSD 视图：NPU 算子的 antiquant_scale 形状要求 [1, H, 1, D]
        bnsd = (1, self.config.num_kv_heads, 1, self.config.head_size)
        return Int8DynamicScales(
            k_inv_scale=k_inv_scale,
            k_offset=k_offset,
            v_inv_scale=v_inv_scale,
            v_offset=v_offset,
            k_aq_scale=(1.0 / k_inv_scale).view(bnsd).contiguous(),
            k_aq_offset=k_offset.view(bnsd).contiguous(),
            v_aq_scale=(1.0 / v_inv_scale).view(bnsd).contiguous(),
            v_aq_offset=v_offset.view(bnsd).contiguous(),
        )

    @staticmethod
    def quantize(x: Any, inv_scale: Any, offset: Any) -> Any:
        """浮点 K/V -> INT8（per-channel scale，对称）。"""
        torch = import_torch("Int8DynamicSemantics.quantize")
        return torch.clamp(torch.round(x * inv_scale + offset), -128, 127).to(
            torch.int8
        )

    @staticmethod
    def dequantize(x: Any, inv_scale: Any, offset: Any, target_dtype: Any) -> Any:
        """量化的逆操作：``(x - offset) * (1 / inv_scale)``。"""
        return (x.to(target_dtype) - offset) * (1.0 / inv_scale)

    def describe(self) -> dict[str, Any]:
        """方案语义的自描述（供文档/工具展示）。"""
        return {
            "scheme": "int8_dynamic_per_channel",
            "scale_shape": [1, self.config.num_kv_heads, self.config.head_size],
            "scale_axis": "token-amax (dim=0), computed on first prefill",
            "symmetric": True,
        }


__all__ = ["Int8DynamicScales", "Int8DynamicSemantics"]
