# SPDX-License-Identifier: Apache-2.0
"""per-token-head KV store 内核（triton-ascend，fp8 E4M3 路径）。

1:1 移植自宿主 vllm-hust 的 ``triton_reshape_and_cache_flash.py``：
``triton_reshape_and_cache_flash_per_token_head_quant`` +
``_reshape_cache_per_token_head``（CUDA triton 参考实现，2026-10-03 读取）。
每 (token, head) 一个程序：absmax/448 得 fp32 scale（下限 1e-6，与
``methods/fp8_per_token_head/semantics`` 的 CPU 参考逐位同口径）、clamp
后以张量 dtype 完成 store cast（fp8 无舍入步；int8 变体半步远离零舍入，
参数表已备、路径未启用）。

与宿主的两处刻意差异：

* 校验器是普通函数、triton 在启动函数内惰性导入——CPU 测试不需要
  triton（kivi_pack 把校验器拆到 ops/kivi_layout.py 是因为它的 triton
  在模块顶层导入；这里单模块即可保持同一导入卫生）。
* 数据指针以 ``uint8 -> float8_e4m3fn`` 视图传入，与宿主内核同构。
  **triton-ascend 对 fp8 指针的支持未上机验证**——这是形状包络探针的
  第一个要回答的问题；若不支持，备选方案是内核内手工编码 E4M3 位型
  （RNE 舍入）后写 uint8，改动只落在本模块。

写目标就是 ``methods/fp8_per_token_head/byte_cache`` 切出的四个视图
（数据 ``[nb, block, kvh, head]`` uint8 + scale ``[nb, block, kvh, 1]``
fp32），宿主内核的缓存张量形状与之完全一致。**尚未路由**：设备验证
（对拍 CPU 参考 + 包络测量）完成前，方法适配器保持缺位。
"""

from __future__ import annotations

import torch

# 与宿主 _PER_TOKEN_HEAD_QUANT_PARAMS 同表：量化范围由缓存 dtype 决定。
# int8 一并移植进参数表（IS_INT_QUANT 分支同源），启用与否由调用方定。
QUANT_PARAMS: dict[torch.dtype, tuple[float, float]] = {
    torch.float8_e4m3fn: (448.0, -448.0),
    torch.int8: (127.0, -128.0),
}


def _check_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_contiguous():
        raise RuntimeError(f"fp8_per_token_head {name} must be contiguous.")


def _check_same_device(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    if a.device != b.device:
        raise RuntimeError(
            f"fp8_per_token_head {name} must share a device with the KV "
            f"tensors, got {a.device} and {b.device}."
        )


def validate_store_args(
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_data: torch.Tensor,
    k_scale: torch.Tensor,
    v_data: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """内核启动前的全部布局检查（fail-closed，CPU 可测）。"""
    num_tokens, num_kv_heads, head_size = key.shape
    if value.shape[0] != num_tokens or value.shape[1] != num_kv_heads:
        raise RuntimeError(
            "fp8_per_token_head key/value token-head grids must match, got "
            f"key {tuple(key.shape)} vs value {tuple(value.shape)}."
        )
    if value.shape[2] != head_size:
        raise RuntimeError(
            "fp8_per_token_head requires equal key/value head sizes, got "
            f"{head_size} vs {value.shape[2]}."
        )
    if head_size % 4:
        raise RuntimeError(
            f"fp8_per_token_head requires head_size divisible by 4 (fp32 "
            f"scale alignment), got {head_size}."
        )
    if slot_mapping.shape != (num_tokens,):
        raise RuntimeError(
            f"fp8_per_token_head slot_mapping must be 1D of {num_tokens} "
            f"slots, got {tuple(slot_mapping.shape)}."
        )
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(
            "fp8_per_token_head slot_mapping must be int32/int64, got "
            f"{slot_mapping.dtype}."
        )

    expected_data = (k_data.shape[0], k_data.shape[1], num_kv_heads, head_size)
    for name, data, scale in (
        ("key", k_data, k_scale),
        ("value", v_data, v_scale),
    ):
        _check_contiguous(f"{name} data", data)
        _check_contiguous(f"{name} scale", scale)
        _check_same_device(f"{name} data", data, key)
        _check_same_device(f"{name} scale", scale, key)
        if data.shape != expected_data or data.dtype is not torch.uint8:
            raise RuntimeError(
                f"fp8_per_token_head {name} data cache must be uint8 "
                f"{expected_data}, got {tuple(data.shape)} / {data.dtype}."
            )
        expected_scale = (data.shape[0], data.shape[1], num_kv_heads, 1)
        if scale.shape != expected_scale or scale.dtype is not torch.float32:
            raise RuntimeError(
                f"fp8_per_token_head {name} scale cache must be float32 "
                f"{expected_scale}, got {tuple(scale.shape)} / {scale.dtype}."
            )


def _load_triton():
    """惰性导入 triton（vllm 宿主走 vllm.triton_utils，独立环境走裸包）。"""
    try:
        from vllm.triton_utils import tl, triton
    except ImportError:
        try:
            import triton
            import triton.language as tl
        except ImportError as exc:
            raise ImportError(
                "the per-token-head store kernel requires triton "
                "(triton-ascend on Ascend NPU), importable directly or "
                "through vllm.triton_utils"
            ) from exc
    return tl, triton


_STORE_KERNEL = None  # 惰性构建的 jitted 内核；进程内只建一次


def _build_store_kernel():
    """构建 jitted 内核（首次启动时调用一次，之后复用同一函数对象）。

    不在模块顶层定义：triton 的导入是惰性的（CPU 测试环境没有
    triton），而 jit 函数对象必须在调用间保持同一实例，否则 triton
    的编译缓存按函数对象失效，设备上每步重编译。
    """
    tl, triton = _load_triton()

    @triton.jit
    def _per_token_head_store_kernel(
        key_ptr,
        value_ptr,
        slot_mapping_ptr,
        k_data_ptr,
        k_scale_ptr,
        v_data_ptr,
        v_scale_ptr,
        num_kv_heads: tl.constexpr,
        block_size: tl.constexpr,
        head_size: tl.constexpr,
        HEAD_SIZE_PADDED: tl.constexpr,
        QUANT_MAX: tl.constexpr,
        QUANT_MIN: tl.constexpr,
        IS_INT_QUANT: tl.constexpr,
    ):
        # 1:1 移植宿主 _reshape_cache_per_token_head：每 (token, head)
        # 一个程序；布局已由校验器保证连续，直接算偏移。clamp 用
        # minimum/maximum 组合——kivi 打包内核在 triton-ascend 3.5 上
        # 验证过的写法（tl.clamp 未上机验过）。
        tok = tl.program_id(0)
        head = tl.program_id(1)

        slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
        if slot < 0:
            return

        blk = slot // block_size
        slot_in_blk = slot % block_size
        dim_offs = tl.arange(0, HEAD_SIZE_PADDED)
        dim_mask = dim_offs < head_size

        k_h = tl.load(
            key_ptr + (tok * num_kv_heads + head) * head_size + dim_offs,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        k_scale = tl.maximum(tl.max(tl.abs(k_h)) / QUANT_MAX, 1e-6)
        tl.store(
            k_scale_ptr + ((blk * block_size + slot_in_blk) * num_kv_heads + head),
            k_scale,
        )
        k_q = k_h * (1.0 / k_scale)
        if IS_INT_QUANT:
            k_q = tl.where(k_q >= 0, k_q + 0.5, k_q - 0.5)
        k_q = tl.minimum(tl.maximum(k_q, QUANT_MIN), QUANT_MAX)
        tl.store(
            k_data_ptr
            + ((blk * block_size + slot_in_blk) * num_kv_heads + head) * head_size
            + dim_offs,
            k_q,
            mask=dim_mask,
        )

        v_h = tl.load(
            value_ptr + (tok * num_kv_heads + head) * head_size + dim_offs,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        v_scale = tl.maximum(tl.max(tl.abs(v_h)) / QUANT_MAX, 1e-6)
        tl.store(
            v_scale_ptr + ((blk * block_size + slot_in_blk) * num_kv_heads + head),
            v_scale,
        )
        v_q = v_h * (1.0 / v_scale)
        if IS_INT_QUANT:
            v_q = tl.where(v_q >= 0, v_q + 0.5, v_q - 0.5)
        v_q = tl.minimum(tl.maximum(v_q, QUANT_MIN), QUANT_MAX)
        tl.store(
            v_data_ptr
            + ((blk * block_size + slot_in_blk) * num_kv_heads + head) * head_size
            + dim_offs,
            v_q,
            mask=dim_mask,
        )

    return _per_token_head_store_kernel


def _launch_kernel(
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_data_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    v_data_fp8: torch.Tensor,
    v_scale: torch.Tensor,
    quant_max: float,
    quant_min: float,
    is_int_quant: bool,
) -> None:
    """实际内核启动（模块级函数：CPU 测试 monkeypatch 这里）。"""
    global _STORE_KERNEL
    if _STORE_KERNEL is None:
        _STORE_KERNEL = _build_store_kernel()
    _, triton = _load_triton()

    num_tokens, num_kv_heads, head_size = key.shape
    head_padded = triton.next_power_of_2(head_size)
    block_size = k_data_fp8.shape[1]
    _STORE_KERNEL[(num_tokens, num_kv_heads)](
        key,
        value,
        slot_mapping,
        k_data_fp8,
        k_scale,
        v_data_fp8,
        v_scale,
        num_kv_heads,
        block_size,
        head_size,
        head_padded,
        quant_max,
        quant_min,
        is_int_quant,
        num_warps=min(16, max(1, head_padded // 32)),
    )


def fp8_per_token_head_store(
    key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size] 高精度输入
    value: torch.Tensor,  # 同上
    slot_mapping: torch.Tensor,  # [num_tokens] 绝对槽位 id
    k_data: torch.Tensor,  # [nb, block, kvh, head] uint8（布局视图）
    k_scale: torch.Tensor,  # [nb, block, kvh, 1] fp32
    v_data: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """把一批全精度 K/V 按 per-token-head E4M3 写进分页缓存。

    数据区按 uint8 布局存储；启动前以 ``view(torch.float8_e4m3fn)``
    取得 fp8 指针（宿主内核同构）。见模块 docstring 的设备风险注记。
    """
    if key.numel() == 0:
        return
    validate_store_args(key, value, slot_mapping, k_data, k_scale, v_data, v_scale)

    quant_max, quant_min = QUANT_PARAMS[torch.float8_e4m3fn]
    _launch_kernel(
        key,
        value,
        slot_mapping,
        k_data.view(torch.float8_e4m3fn),
        k_scale,
        v_data.view(torch.float8_e4m3fn),
        v_scale,
        quant_max,
        quant_min,
        is_int_quant=False,
    )


__all__ = ["QUANT_PARAMS", "fp8_per_token_head_store", "validate_store_args"]
