# SPDX-License-Identifier: Apache-2.0
"""KIVI INT4 分页缓存的字节区域布局（纯 torch，CPU 可测）。

vLLM 这一代没有"一层发多张 KV 缓存"的机制：宿主的 ``bind_kv_cache`` 只
会把键、值两张缓冲交给 impl。KIVI 的打包内核与 gather 路径却按 6 个具名
张量写。两侧需要的字节数恰好相等，所以两张缓冲就够，另外 4 个是视图
（视图不复制数据，内核写的仍是宿主那块显存）。

张量内顺序切块、不交错::

    K: [k_quant int32 | k_scale fp32 | k_mn fp32]
    V: [v_quant int32 | v_scale fp32 | v_mn fp32]

每 token 每 head 的字节数 ``S = head_size/2 + 8*head_size/group_size``：
int4 数据 0.5 B/lane，每组一份的 scale 与 min 各 4 B、摊到 group 个 token
上即每 token ``8*head_size/group_size`` B。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...ops.kivi_layout import _check_contiguous
from .geometry import validate_kivi_geometry

_INT4_LANES_PER_WORD = 8


@dataclass(frozen=True)
class KiviByteCacheLayout:
    """一次 KIVI 分页分配的形状，以及两张缓冲到 6 个视图的切法。"""

    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_size: int
    group_size: int

    def __post_init__(self) -> None:
        if self.num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {self.num_blocks}")
        validate_kivi_geometry(
            head_size=self.head_size,
            group_size=self.group_size,
            residual_length=self.group_size,
            block_size=self.block_size,
        )

    # -- 视图规格 -----------------------------------------------------------

    def key_specs(self) -> tuple[tuple[tuple[int, ...], Any], ...]:
        nb, kvh, head = self.num_blocks, self.num_kv_heads, self.head_size
        block, group = self.block_size, self.group_size
        return (
            ((nb, kvh, head, block // _INT4_LANES_PER_WORD), torch.int32),
            ((nb, kvh, head, block // group), torch.float32),
            ((nb, kvh, head, block // group), torch.float32),
        )

    def value_specs(self) -> tuple[tuple[tuple[int, ...], Any], ...]:
        nb, block, kvh, head = (
            self.num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_size,
        )
        group = self.group_size
        return (
            ((nb, block, kvh, head // _INT4_LANES_PER_WORD), torch.int32),
            ((nb, block, kvh, head // group), torch.float32),
            ((nb, block, kvh, head // group), torch.float32),
        )

    # -- 字节预算 -----------------------------------------------------------

    @property
    def key_bytes(self) -> int:
        return _specs_bytes(self.key_specs())

    @property
    def value_bytes(self) -> int:
        return _specs_bytes(self.value_specs())

    @property
    def region_bytes(self) -> int:
        """单张缓冲的字节数。两侧相等是"两张宿主张量够用"的前提。"""
        if self.key_bytes != self.value_bytes:
            raise RuntimeError(
                "KIVI byte layout requires equal key/value regions, got "
                f"{self.key_bytes} vs {self.value_bytes}."
            )
        return self.key_bytes

    @property
    def bytes_per_token_head(self) -> int:
        """int4 数据 + 摊到每个 token 的 scale/min 字节数（单侧、单 head）。"""
        return self.head_size // 2 + 8 * self.head_size // self.group_size

    @property
    def bytes_per_token(self) -> int:
        """K+V 每 token 的字节数（含 scale/min 开销）。"""
        return 2 * self.num_kv_heads * self.bytes_per_token_head

    def compression_vs_fp16(self) -> float:
        """相对 fp16 稠密 KV 的实际压缩比（不是只看 int4 数据的 4x）。"""
        fp16_bytes = 2 * self.num_kv_heads * self.head_size * 2
        return fp16_bytes / self.bytes_per_token

    def describe(self) -> dict[str, Any]:
        return {
            "region_bytes": self.region_bytes,
            "bytes_per_token": self.bytes_per_token,
            "compression_vs_fp16": self.compression_vs_fp16(),
            "key": _specs_summary(self.key_specs()),
            "value": _specs_summary(self.value_specs()),
        }


def _specs_bytes(specs: tuple[tuple[tuple[int, ...], Any], ...]) -> int:
    return sum(_elements(shape) * dtype.itemsize for shape, dtype in specs)


def _elements(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _specs_summary(
    specs: tuple[tuple[tuple[int, ...], Any], ...],
) -> list[dict[str, Any]]:
    return [
        {"shape": list(shape), "dtype": str(dtype).replace("torch.", "")}
        for shape, dtype in specs
    ]


def kivi_byte_cache_layout(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    num_kv_heads: int,
    head_size: int,
    group_size: int,
) -> KiviByteCacheLayout:
    """从两张宿主缓冲反推分页布局；除不尽就 fail-closed。

    宿主只告诉我们张量形状，块大小要自己算：
    ``region_bytes = num_blocks * block_size * num_kv_heads * S``。
    """
    _check_contiguous("key_cache", key_cache)
    _check_contiguous("value_cache", value_cache)
    if key_cache.device != value_cache.device:
        raise RuntimeError(
            f"KIVI key/value caches must share a device, got {key_cache.device} "
            f"and {value_cache.device}."
        )
    key_bytes = key_cache.numel() * key_cache.element_size()
    value_bytes = value_cache.numel() * value_cache.element_size()
    if key_bytes != value_bytes:
        raise RuntimeError(
            f"KIVI key cache holds {key_bytes} bytes but value cache holds "
            f"{value_bytes}; the int4 layout needs equal regions."
        )
    num_blocks = int(key_cache.shape[0])
    bytes_per_token_head = head_size // 2 + 8 * head_size // group_size
    stride = num_blocks * num_kv_heads * bytes_per_token_head
    if stride <= 0 or key_bytes % stride:
        raise RuntimeError(
            f"KIVI cache region of {key_bytes} bytes is not a whole number of "
            f"pages for head_size={head_size}, group_size={group_size}, "
            f"num_kv_heads={num_kv_heads}, num_blocks={num_blocks}."
        )
    block_size = key_bytes // stride
    return KiviByteCacheLayout(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        group_size=group_size,
    )


def _views_for_region(
    cache: torch.Tensor,
    specs: tuple[tuple[tuple[int, ...], Any], ...],
    region_bytes: int,
    name: str,
) -> list[torch.Tensor]:
    if not cache.is_contiguous():
        raise RuntimeError(f"KIVI {name} cache must be contiguous.")
    available = cache.numel() * cache.element_size()
    if available != region_bytes:
        raise RuntimeError(
            f"KIVI {name} cache must hold {region_bytes} bytes, got {available}."
        )
    flat = cache.reshape(-1).view(torch.uint8)
    views: list[torch.Tensor] = []
    offset = 0
    for shape, dtype in specs:
        span = _elements(shape) * dtype.itemsize
        views.append(flat.narrow(0, offset, span).view(dtype).view(shape))
        offset += span
    if offset != flat.numel():
        raise RuntimeError(
            f"KIVI {name} region bookkeeping error: {offset} vs {flat.numel()} bytes."
        )
    return views


def kivi_caches_from_byte_tensors(
    key_cache: torch.Tensor, value_cache: torch.Tensor, layout: KiviByteCacheLayout
) -> tuple[torch.Tensor, ...]:
    """两张宿主缓冲 -> ``(k_quant, k_scale, k_mn, v_quant, v_scale, v_mn)`` 视图。"""
    region_bytes = layout.region_bytes
    return (
        *_views_for_region(key_cache, layout.key_specs(), region_bytes, "key"),
        *_views_for_region(value_cache, layout.value_specs(), region_bytes, "value"),
    )


__all__ = [
    "KiviByteCacheLayout",
    "kivi_byte_cache_layout",
    "kivi_caches_from_byte_tensors",
]
