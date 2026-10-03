# SPDX-License-Identifier: Apache-2.0
"""FP8 per-token-head 分页缓存的字节区域布局（纯 torch，CPU 可测）。

与 KIVI 同一"两张等大宿主缓冲、插件切视图"的接法（HOST_CONTRACT 第一
协议），但每侧只有两张视图（无 mn、无残差区）::

    K: [k_data uint8 (head_size B) | k_scale fp32 (4 B)]   每 (token, head)
    V: [v_data uint8 (head_size B) | v_scale fp32 (4 B)]

每 token 每 head 单侧字节数 ``S = head_size + 4``；K、V 公式相同所以
两侧等大。这与宿主页预算逐位一致（vllm/v1/kv_cache_interface.py 的
per_token_head 分支：数据 ``2*block*kvh*head_size*1`` + scale
``2*block*kvh*4`` = ``2*block*kvh*(head_size+4)``，scale 的显存从 KV
cache 原始分配里刻出）。

数据字节按 E4M3 解释（``Tensor.view(torch.float8_e4m3fn)``），
scale 每 (token, head) 一份 fp32；head_size % 4 == 0 保证每个
(token, head) 槽内的 scale 落在 4 字节边界上。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...ops.kivi_layout import _check_contiguous


@dataclass(frozen=True)
class Fp8PerTokenHeadLayout:
    """一次 FP8 per-token-head 分页分配的形状与视图切法。"""

    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_size: int

    def __post_init__(self) -> None:
        if self.num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {self.num_blocks}")
        if self.head_size <= 0 or self.head_size % 4:
            raise ValueError(
                "fp8_per_token_head requires head_size positive and divisible "
                f"by 4, got {self.head_size}"
            )

    # -- 视图规格（K、V 同构） ----------------------------------------------

    def side_specs(self) -> tuple[tuple[tuple[int, ...], Any], ...]:
        nb, block, kvh, head = (
            self.num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_size,
        )
        return (
            ((nb, block, kvh, head), torch.uint8),
            ((nb, block, kvh, 1), torch.float32),
        )

    # 正确性同 KIVI：两侧字节预算相等是"两张宿主张量够用"的前提。
    key_specs = side_specs
    value_specs = side_specs

    # -- 字节预算 ------------------------------------------------------------

    @property
    def bytes_per_token_head(self) -> int:
        """单侧、单 (token, head) 的字节数：E4M3 数据 + fp32 scale。"""
        return self.head_size + 4

    @property
    def region_bytes(self) -> int:
        """单张缓冲的字节数（K、V 相同）。"""
        return (
            self.num_blocks
            * self.block_size
            * self.num_kv_heads
            * self.bytes_per_token_head
        )

    def compression_vs_fp16(self) -> float:
        """相对 fp16 稠密 KV 的单侧压缩比。"""
        return (2 * self.head_size) / self.bytes_per_token_head

    def describe(self) -> dict[str, Any]:
        return {
            "region_bytes": self.region_bytes,
            "bytes_per_token": 2 * self.num_kv_heads * self.bytes_per_token_head,
            "compression_vs_fp16": self.compression_vs_fp16(),
            "side": [
                {"shape": list(shape), "dtype": str(dtype).replace("torch.", "")}
                for shape, dtype in self.side_specs()
            ],
        }


def fp8_per_token_head_layout(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    num_kv_heads: int,
    head_size: int,
) -> Fp8PerTokenHeadLayout:
    """从两张宿主缓冲反推分页布局；对不上就 fail-closed。

    宿主只给张量形状，块大小自己算：
    ``region_bytes = num_blocks * block_size * num_kv_heads * (head_size + 4)``。
    """
    _check_contiguous("key_cache", key_cache)
    _check_contiguous("value_cache", value_cache)
    if key_cache.device != value_cache.device:
        raise RuntimeError(
            f"fp8_per_token_head key/value caches must share a device, got "
            f"{key_cache.device} and {value_cache.device}."
        )
    key_bytes = key_cache.numel() * key_cache.element_size()
    value_bytes = value_cache.numel() * value_cache.element_size()
    if key_bytes != value_bytes:
        raise RuntimeError(
            f"fp8_per_token_head key cache holds {key_bytes} bytes but value "
            f"cache holds {value_bytes}; the layout needs equal regions."
        )
    num_blocks = int(key_cache.shape[0])
    bytes_per_token_head = head_size + 4
    stride = num_blocks * num_kv_heads * bytes_per_token_head
    if stride <= 0 or key_bytes % stride:
        raise RuntimeError(
            f"fp8_per_token_head cache region of {key_bytes} bytes is not a "
            f"whole number of pages for head_size={head_size}, "
            f"num_kv_heads={num_kv_heads}, num_blocks={num_blocks}."
        )
    block_size = key_bytes // stride
    return Fp8PerTokenHeadLayout(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
    )


def _views_for_region(
    cache: torch.Tensor,
    specs: tuple[tuple[tuple[int, ...], Any], ...],
    region_bytes: int,
    name: str,
) -> list[torch.Tensor]:
    if not cache.is_contiguous():
        raise RuntimeError(f"fp8_per_token_head {name} cache must be contiguous.")
    available = cache.numel() * cache.element_size()
    if available != region_bytes:
        raise RuntimeError(
            f"fp8_per_token_head {name} cache must hold {region_bytes} bytes, "
            f"got {available}."
        )
    flat = cache.reshape(-1).view(torch.uint8)
    views: list[torch.Tensor] = []
    offset = 0
    for shape, dtype in specs:
        span = 1
        for dim in shape:
            span *= dim
        span *= dtype.itemsize
        views.append(flat.narrow(0, offset, span).view(dtype).view(shape))
        offset += span
    if offset != flat.numel():
        raise RuntimeError(
            f"fp8_per_token_head {name} region bookkeeping error: {offset} vs "
            f"{flat.numel()} bytes."
        )
    return views


def fp8_caches_from_byte_tensors(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    layout: Fp8PerTokenHeadLayout,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """两张宿主缓冲 -> ``(k_data, k_scale, v_data, v_scale)`` 四视图。"""
    region_bytes = layout.region_bytes
    k_data, k_scale = _views_for_region(
        key_cache, layout.key_specs(), region_bytes, "key"
    )
    v_data, v_scale = _views_for_region(
        value_cache, layout.value_specs(), region_bytes, "value"
    )
    return k_data, k_scale, v_data, v_scale


__all__ = [
    "Fp8PerTokenHeadLayout",
    "fp8_caches_from_byte_tensors",
    "fp8_per_token_head_layout",
]
