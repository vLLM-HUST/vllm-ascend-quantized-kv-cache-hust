# SPDX-License-Identifier: Apache-2.0
"""FP8 per-token-head 的纯语义（纯 torch，CPU 可测）。

方案定位（调研文档 kvquant-schemes-beyond-int8-int4.md §9.3/§10.1）：
宿主 vllm-hust 的 triton 后端原生支持 ``fp8_per_token_head``（per-token-head
动态 scale 的 E4M3 存储），是 NPU 上不依赖任何厂商新算子的最低成本新
dtype。与本仓库 int8_dynamic 同一"在线 scale、免校准、免预量化
checkpoint"定位，但粒度细到 (token, head) 且带 E4M3 的浮点动态范围——
int8_dynamic 的 scale 只在首个 prefill 算一次（会陈旧），这里每个
token 写入时独立计算。

数值口径（本插件的 CPU 参考实现，``ops.triton.per_token_head_store``
内核的对拍基准，与宿主 triton 内核逐步同构）::

    scale = max(|x|) / 448            # 448 = E4M3fn 最大正规数
    scale = max(scale, 1e-6)          # 与宿主内核同一下限，防全零槽位
    q     = e4m3(clamp(x / scale, -448, 448))
    x̂     = float(q) * scale

clamp 在 cast 之前，规避 torch fp8 cast 对超范围的 NaN 行为。
"""

from __future__ import annotations

from typing import Any

import torch

_E4M3_MAX = 448.0
# 与宿主 triton 内核同口径的下限（triton_reshape_and_cache_flash.py：
# ``tl.maximum(absmax / QUANT_MAX, 1e-6)``）。必须逐位一致，否则这份
# CPU 参考与内核对拍时全零 (token, head) 槽位的 scale 会差 1e6 倍。
_SCALE_FLOOR = 1e-6


class Fp8PerTokenHeadSemantics:
    """E4M3 + 每 (token, head) 动态 fp32 scale 的纯语义对象。"""

    e4m3_max = _E4M3_MAX

    def __init__(self, config: Any) -> None:
        head_size = getattr(config, "head_size", None)
        if not isinstance(head_size, int) or head_size <= 0 or head_size % 4:
            raise ValueError(
                "fp8_per_token_head requires head_size positive and divisible "
                f"by 4 (fp32 scale alignment), got {head_size!r}"
            )
        self.head_size = head_size

    # -- 量化数学 ------------------------------------------------------------

    def compute_scale(self, values: torch.Tensor) -> torch.Tensor:
        """每 (token, head) 的动态 scale：``[T, KVH, D] -> [T, KVH, 1]``。"""
        amax = values.to(torch.float32).abs().amax(dim=-1, keepdim=True)
        return (amax / _E4M3_MAX).clamp(min=_SCALE_FLOOR)

    def quantize(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[T, KVH, D]`` -> (E4M3 数据同形, fp32 scale ``[T, KVH, 1]``)。"""
        scale = self.compute_scale(values)
        normalized = (values.to(torch.float32) / scale).clamp(-_E4M3_MAX, _E4M3_MAX)
        return normalized.to(torch.float8_e4m3fn), scale

    def dequantize(self, packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """(E4M3 数据, scale) -> fp32 稠密 ``[T, KVH, D]``。"""
        return packed.to(torch.float32) * scale

    def fake_quant(self, values: torch.Tensor) -> torch.Tensor:
        """参考往返：真机内核的对拍基准（写入再读出的一次性口径）。"""
        packed, scale = self.quantize(values)
        return self.dequantize(packed, scale).to(values.dtype)

    def describe(self) -> dict[str, Any]:
        return {
            "scheme": "fp8_per_token_head",
            "bits": 8,
            "encoding": "e4m3fn (max normal 448)",
            "scale_granularity": "per (token, head), fp32, dynamic",
            "calibration": "none (online amax)",
        }


__all__ = ["Fp8PerTokenHeadSemantics"]
