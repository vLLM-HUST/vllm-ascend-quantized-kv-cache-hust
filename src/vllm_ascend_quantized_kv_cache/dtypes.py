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
    FP8_PER_TENSOR = 1  # FP8 按张量
    INT8_PER_TOKEN_HEAD = 2  # INT8 按 token-head
    FP8_PER_TOKEN_HEAD = 3
    INT4_PER_TOKEN_HEAD = 4
    NVFP4 = 5  # fp4 数据 + fp8 块 scale（16 元素一组）
    INT4 = 6
    FP4_E2M1 = 7  # MXFP4 微缩格式：16 元素共享 1 个 fp8 指数 scale
    INT8_PER_TENSOR = 8
    KIVI_INT4 = 9  # KIVI：int4 历史 + 全精度残差窗口


def get_kv_quant_mode(dtype: str) -> KVQuantMode:
    """dtype 字符串 -> 量化模式。

    精确匹配优先；``fp8`` 前缀统一归入 FP8_PER_TENSOR；其余返回 NONE
    （fail-closed：调用方据此拒绝，而不是猜一个布局）。
    """
    exact = {
        "int4_per_token_head": KVQuantMode.INT4_PER_TOKEN_HEAD,
        "int8_per_token_head": KVQuantMode.INT8_PER_TOKEN_HEAD,
        "fp8_per_token_head": KVQuantMode.FP8_PER_TOKEN_HEAD,
        "int4": KVQuantMode.INT4,
        "nvfp4": KVQuantMode.NVFP4,
        "fp4_e2m1": KVQuantMode.FP4_E2M1,
        "int8": KVQuantMode.INT8_PER_TENSOR,
        "kivi_int4": KVQuantMode.KIVI_INT4,
    }
    if dtype in exact:
        return exact[dtype]
    if dtype.startswith("fp8"):
        return KVQuantMode.FP8_PER_TENSOR
    return KVQuantMode.NONE


def is_quantized_kv_cache(dtype: str) -> bool:
    """该 dtype 是否是量化 KV（即能否解析出非 NONE 的量化模式）。"""
    return get_kv_quant_mode(dtype) is not KVQuantMode.NONE


def int4_packed_dim(head_size: int) -> int:
    """INT4 打包后的每头维度：2 个 int4 共 1 字节，故为 head_size/2。"""
    if head_size <= 0 or head_size % 2:
        raise ValueError("INT4 head_size must be a positive even number")
    return head_size // 2


def fp4_e2m1_packed_dim(head_size: int) -> int:
    """FP4 E2M1（MXFP4）打包维度：每 16 元素 8B 数据 + 1B fp8 scale，

    即 ceil(head_size/16) * 9（head_size=128 时为 72）。
    """
    if head_size <= 0:
        raise ValueError("FP4 head_size must be positive")
    blocks = (head_size + 15) // 16
    return blocks * 9


def nvfp4_packed_dim(head_size: int) -> int:
    """NVFP4 打包维度：fp4 数据 half + fp8 块 scale，且要求 16 对齐。"""
    if head_size <= 0 or head_size % 16:
        raise ValueError("NVFP4 head_size must be divisible by 16")
    return head_size // 2 + head_size // 16


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

    INT4 系 / FP4_E2M1 / NVFP4 -> uint8 存储（4bit 打包）；
    FP8 系 -> uint8 存储（1 字节元素，packed = head_size）；
    INT8 系 -> int8 存储；
    未注册 dtype 直接抛 ValueError——"一个 uint8 分配不能作为支持某
    dtype 的证据"（HOST_CONTRACT 的 fail-closed 原则）。
    """
    mode = get_kv_quant_mode(dtype)
    if mode in {
        KVQuantMode.INT4,
        KVQuantMode.INT4_PER_TOKEN_HEAD,
        KVQuantMode.KIVI_INT4,
    }:
        packed = int4_packed_dim(head_size)
        storage = "uint8"
    elif mode is KVQuantMode.FP4_E2M1:
        packed = fp4_e2m1_packed_dim(head_size)
        storage = "uint8"
    elif mode is KVQuantMode.NVFP4:
        packed = nvfp4_packed_dim(head_size)
        storage = "uint8"
    elif mode in {KVQuantMode.FP8_PER_TENSOR, KVQuantMode.FP8_PER_TOKEN_HEAD}:
        if head_size <= 0:
            raise ValueError("head_size must be positive")
        packed, storage = head_size, "uint8"
    elif mode in {KVQuantMode.INT8_PER_TENSOR, KVQuantMode.INT8_PER_TOKEN_HEAD}:
        if head_size <= 0:
            raise ValueError("head_size must be positive")
        packed, storage = head_size, "int8"
    else:
        raise ValueError(f"dtype is not a registered quantized KV layout: {dtype}")
    return KVCacheLayout(dtype, head_size, storage, packed, mode)


__all__ = [
    "KVCacheLayout",
    "KVQuantMode",
    "fp4_e2m1_packed_dim",
    "get_kv_quant_mode",
    "int4_packed_dim",
    "is_quantized_kv_cache",
    "nvfp4_packed_dim",
    "resolve_layout",
]
