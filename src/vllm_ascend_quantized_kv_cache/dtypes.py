# SPDX-License-Identifier: Apache-2.0
"""量化 KV 的 dtype / 布局契约（层 0，挖掘自 legacy core PR #181）。

本模块是全库的"宪法"：dtype 字符串 -> 量化模式 -> 存储布局 的唯一裁决
处。纯 Python、零依赖、设备无关；所有未知 dtype 一律 fail-closed（返回
NONE 或抛 ValueError），绝不静默回退。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class KVQuantMode(IntEnum):
    """量化模式枚举（数值与 legacy vllm 的 KVQuantMode 对齐）。"""

    NONE = 0
    KIVI_INT4 = 9  # KIVI：int4 历史区 + 全精度残差窗口（ascend#116 口径）
    INT8_PER_TENSOR = 8


def get_kv_quant_mode(dtype: str) -> KVQuantMode:
    """dtype 字符串 -> 量化模式。

    仅接受 CLI 字面量 ``int8`` 与 ``kivi_int4``；其余返回 NONE。
    """
    if dtype == "int8":
        return KVQuantMode.INT8_PER_TENSOR
    if dtype == "kivi_int4":
        return KVQuantMode.KIVI_INT4
    return KVQuantMode.NONE


def is_quantized_kv_cache(dtype: str) -> bool:
    """该 dtype 是否是量化 KV（即能否解析出非 NONE 的量化模式）。"""
    return get_kv_quant_mode(dtype) is not KVQuantMode.NONE


def int4_packed_dim(head_size: int) -> int:
    """INT4 打包后的每头维度：2 个 int4 共 1 字节，故为 head_size/2。"""
    if head_size <= 0 or head_size % 2:
        raise ValueError("INT4 head_size must be a positive even number")
    return head_size // 2


@dataclass(frozen=True)
class KVCacheLayout:
    """解析后的缓存存储布局（不可变值对象）。

    dtype:        契约层的 dtype 字符串
    head_size:    每头维度
    storage_dtype: 实际存储的元素类型（"uint8" / "int8"）
    packed_last_dim: 打包后每头（或每 token）的最后一维大小
    quant_mode:   对应的量化模式
    """

    dtype: str
    head_size: int
    storage_dtype: str
    packed_last_dim: int
    quant_mode: KVQuantMode


def resolve_layout(dtype: str, head_size: int) -> KVCacheLayout:
    """dtype + head_size -> 存储布局（fail-closed）。

    ``int8`` 使用未打包的 int8 存储；``kivi_int4`` 的历史区是 4bit，
    两个 int4 打进一字节（uint8 存储，packed 维度减半）；其他 dtype
    直接抛 ValueError。
    """
    mode = get_kv_quant_mode(dtype)
    if mode is KVQuantMode.INT8_PER_TENSOR:
        if head_size <= 0:
            raise ValueError("head_size must be positive")
        packed, storage = head_size, "int8"
    elif mode is KVQuantMode.KIVI_INT4:
        packed, storage = int4_packed_dim(head_size), "uint8"
    else:
        raise ValueError(f"dtype is not a registered quantized KV layout: {dtype}")
    return KVCacheLayout(dtype, head_size, storage, packed, mode)


__all__ = [
    "KVCacheLayout",
    "KVQuantMode",
    "get_kv_quant_mode",
    "int4_packed_dim",
    "is_quantized_kv_cache",
    "resolve_layout",
]
