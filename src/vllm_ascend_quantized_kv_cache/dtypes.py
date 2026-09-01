# SPDX-License-Identifier: Apache-2.0
"""Packed KV dtype/layout contract migrated from legacy PR #181."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class KVQuantMode(IntEnum):
    NONE = 0
    FP8_PER_TENSOR = 1
    INT8_PER_TOKEN_HEAD = 2
    FP8_PER_TOKEN_HEAD = 3
    INT4_PER_TOKEN_HEAD = 4
    NVFP4 = 5
    INT4 = 6
    FP4_E2M1 = 7
    INT8_PER_TENSOR = 8
    KIVI_INT4 = 9


def get_kv_quant_mode(dtype: str) -> KVQuantMode:
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
    return get_kv_quant_mode(dtype) is not KVQuantMode.NONE


def int4_packed_dim(head_size: int) -> int:
    if head_size <= 0 or head_size % 2:
        raise ValueError("INT4 head_size must be a positive even number")
    return head_size // 2


def fp4_e2m1_packed_dim(head_size: int) -> int:
    if head_size <= 0:
        raise ValueError("FP4 head_size must be positive")
    blocks = (head_size + 15) // 16
    return blocks * 9


def nvfp4_packed_dim(head_size: int) -> int:
    if head_size <= 0 or head_size % 16:
        raise ValueError("NVFP4 head_size must be divisible by 16")
    return head_size // 2 + head_size // 16


@dataclass(frozen=True)
class KVCacheLayout:
    dtype: str
    head_size: int
    storage_dtype: str
    packed_last_dim: int
    quant_mode: KVQuantMode


def resolve_layout(dtype: str, head_size: int) -> KVCacheLayout:
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
