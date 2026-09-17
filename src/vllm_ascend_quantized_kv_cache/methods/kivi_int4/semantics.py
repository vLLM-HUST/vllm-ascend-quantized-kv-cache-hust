# SPDX-License-Identifier: Apache-2.0
"""KIVI INT4 方案的纯语义（纯 torch，CPU 可测）。

挖掘自 legacy ascend PR #116 提交 0003-0009（最终状态）。KIVI 把 KV
缓存分成两个区域：
  * 历史区（paged int4 cache）：键按 token 组（每 group_size 个连续
    token 一组，组内逐 (head, dim) 求 min/max）、值按 head 维组
    （每个 token 的 head_dim 按 group_size 分组）分别量化打包；
  * 残差区（residual window）：每个请求最近 residual_length 个 token
    保持全精度；键窗口写满后整组 flush 进历史区。

设备侧打包/聚集在 triton-ascend 内核里（``ops.triton.kivi_cache`` 的
pack 内核）；本模块全部是普通 torch 运算，在 CPU 上即可单测，同时也是
NPU 内核的数值参考实现。
"""

from __future__ import annotations

from typing import Any

import torch


def validate_kivi_geometry(
    *,
    head_size: int,
    group_size: int,
    residual_length: int,
    block_size: int | None = None,
) -> None:
    """复刻 legacy 实现强制执行的全部布局不变量。

    ``block_size`` 可选：impl 对象在绑定缓存后才能知道块大小，且
    ``_write_kivi_key_quant_cache`` 在 flush 时会再查一次
    （block_size % group_size）。
    """
    if group_size <= 0 or group_size % 8:
        # int32 打包要求组大小是 8 的倍数（1 word = 8 个 int4 lane）
        raise ValueError(
            f"kivi_int4 requires group_size divisible by 8, got {group_size}"
        )
    if residual_length % group_size:
        # 残差窗口按整组 flush，必须能被组大小整除
        raise ValueError(
            f"kivi_int4 requires residual_length ({residual_length}) "
            f"to be divisible by group_size ({group_size})"
        )
    if head_size % 8:
        raise ValueError(
            f"kivi_int4 requires head_size divisible by 8, got {head_size}"
        )
    if head_size % group_size:
        # 值按 head 维分组，head_dim 必须能被组大小整除
        raise ValueError(
            f"kivi_int4 requires head_size ({head_size}) to be "
            f"divisible by group_size ({group_size})"
        )
    if block_size is not None and block_size % group_size:
        raise ValueError(
            f"kivi_int4 requires block_size ({block_size}) to be "
            f"divisible by group_size ({group_size})"
        )


def validate_kivi_config(config: Any) -> None:
    """校验入口：接受 MethodConfig，也接受活的 impl 对象
    （后者的属性名带 kivi_ 前缀，如 kivi_group_size）。"""
    group_size = getattr(config, "group_size", None)
    if group_size is None:
        group_size = getattr(config, "kivi_group_size", None)
    residual_length = getattr(config, "residual_length", None)
    if residual_length is None:
        residual_length = getattr(config, "kivi_residual_length", None)
    if group_size is None or residual_length is None:
        raise ValueError(
            "kivi_int4 config must define group_size/residual_length "
            "(or kivi_group_size/kivi_residual_length)"
        )
    validate_kivi_geometry(
        head_size=config.head_size,
        group_size=group_size,
        residual_length=residual_length,
        block_size=getattr(config, "block_size", None),
    )


def _normalize_config(config: Any) -> Any:
    """把 impl 对象（kivi_* 属性名）归一成统一字段视图。"""
    if hasattr(config, "group_size"):
        return config
    from types import SimpleNamespace

    return SimpleNamespace(
        head_size=config.head_size,
        group_size=config.kivi_group_size,
        residual_length=config.kivi_residual_length,
        block_size=getattr(config, "block_size", None),
    )


class KiviInt4Semantics:
    """KIVI INT4 的纯语义对象：打包数学 + 残差窗口簿记（纯 list/torch 逻辑）。"""

    def __init__(self, config: Any) -> None:
        validate_kivi_config(config)
        self.config = _normalize_config(config)

    # -- 配置几何 ------------------------------------------------------------

    @property
    def group_size(self) -> int:
        return self.config.group_size

    @property
    def bits(self) -> int:
        return 4

    def packed_key_last_dim(self) -> int:
        """键每 token 行的 int32 word 数：head_size / 8。"""
        return self.config.head_size // 8

    # -- 打包数学 ------------------------------------------------------------

    @staticmethod
    def unpack_int4(packed: Any) -> Any:
        """int32 word -> 8 个 int4 lane（新增最后一维），float 表示。

        写成 8 次标量移位而不是广播移位：torch_npu 的 aclnnRightShift
        适配器不支持广播（且把 self 操作数当作输出形状）。
        """
        packed_i32 = packed.to(torch.int32)
        unpacked = torch.stack(
            [(packed_i32 >> (4 * lane)) & 0xF for lane in range(8)], dim=-1
        )
        return unpacked.to(torch.float32)

    @staticmethod
    def pack_int4(quant: Any) -> Any:
        """8 个 int4 lane（最后一维）打包进一个 int32 word（lane L 在 bit 4L）。"""
        quant = quant.to(torch.int32)
        packed = torch.zeros_like(quant[..., 0], dtype=torch.int32)
        for lane in range(8):
            packed = packed | (quant[..., lane] << (lane * 4))
        return packed

    def fake_quant_key(self, key: Any) -> Any:
        """键的 token 组假量化 [T, H, D]：整组 min/max 对称量化后再反量化。

        这是 NPU 打包内核的数值参考实现——测试用它对拍真机结果。
        """
        if key.numel() == 0:
            return key
        group_size = self.group_size
        num_tokens = key.shape[0]
        # token 数不是组的整数倍时，复制末 token 补齐（补齐部分再裁掉）
        pad_tokens = (group_size - num_tokens % group_size) % group_size
        work = key.to(torch.float32)
        if pad_tokens:
            pad = work[-1:].expand(pad_tokens, -1, -1)
            work = torch.cat([work, pad], dim=0)
        grouped = work.view(-1, group_size, *key.shape[1:])
        mn = grouped.amin(dim=1, keepdim=True)
        mx = grouped.amax(dim=1, keepdim=True)
        scale = (mx - mn).clamp(min=1e-6) / (2**self.bits - 1)
        quant = torch.clamp(torch.round((grouped - mn) / scale), 0, 2**self.bits - 1)
        dequant = (quant * scale + mn).view(-1, *key.shape[1:])
        return dequant[:num_tokens].to(key.dtype)

    def fake_quant_value(self, value: Any) -> Any:
        """值的 head 维组假量化 [T, H, D]：每个 token 的 head 维按组量化。"""
        if value.numel() == 0:
            return value
        group_size = self.group_size
        head_size = value.shape[-1]
        pad_dim = (group_size - head_size % group_size) % group_size
        work = value.to(torch.float32)
        if pad_dim:
            pad = work[..., -1:].expand(*work.shape[:-1], pad_dim)
            work = torch.cat([work, pad], dim=-1)
        grouped = work.view(*work.shape[:-1], -1, group_size)
        mn = grouped.amin(dim=-1, keepdim=True)
        mx = grouped.amax(dim=-1, keepdim=True)
        scale = (mx - mn).clamp(min=1e-6) / (2**self.bits - 1)
        quant = torch.clamp(torch.round((grouped - mn) / scale), 0, 2**self.bits - 1)
        dequant = (quant * scale + mn).view(*work.shape)
        return dequant[..., :head_size].to(value.dtype)

    def dequant_key_blocks(
        self, k_quant: Any, k_scale: Any, k_mn: Any, target_dtype: Any
    ) -> Any:
        """键块反量化：[B, blocks, KVH, D, B/8] -> [B, blocks, B, KVH, D]。

        scale/mn 每组一份（最后一维 block/group 个），沿 token 维展开。
        """
        q = self.unpack_int4(k_quant).flatten(-2)
        block_size = q.shape[-1]
        scale = k_scale.repeat_interleave(self.group_size, dim=-1)[..., :block_size]
        mn = k_mn.repeat_interleave(self.group_size, dim=-1)[..., :block_size]
        deq = q.to(scale.dtype) * scale + mn
        # [B, blocks, KVH, D, B] -> [B, blocks, B, KVH, D]
        return deq.permute(0, 1, 4, 2, 3).contiguous().to(target_dtype)

    def dequant_value_blocks(
        self, v_quant: Any, v_scale: Any, v_mn: Any, target_dtype: Any
    ) -> Any:
        """值块反量化：[B, blocks, B, KVH, H/8] -> [B, blocks, B, KVH, D]。"""
        q = self.unpack_int4(v_quant).flatten(-2)
        head_size = q.shape[-1]
        scale = v_scale.repeat_interleave(self.group_size, dim=-1)[..., :head_size]
        mn = v_mn.repeat_interleave(self.group_size, dim=-1)[..., :head_size]
        deq = q.to(scale.dtype) * scale + mn
        return deq.contiguous().to(target_dtype)

    @staticmethod
    def normalize_block_layout(
        blocks: Any,
        batch_size: int,
        max_blocks: int,
        cache_block_size: int,
        name: str,
    ) -> Any:
        """接受 [B, blocks, block_size, ...] 或其转置，归一成前者。"""
        if blocks.ndim != 5:
            raise RuntimeError(
                f"KIVI {name} cache must be 5D after dequant, got "
                f"shape={tuple(blocks.shape)}."
            )
        if blocks.shape[0] != batch_size:
            raise RuntimeError(
                f"KIVI {name} batch mismatch after dequant: expected "
                f"{batch_size}, got {blocks.shape[0]}."
            )
        if blocks.shape[1] == max_blocks and blocks.shape[2] == cache_block_size:
            return blocks
        if blocks.shape[1] == cache_block_size and blocks.shape[2] == max_blocks:
            return blocks.transpose(1, 2).contiguous()
        raise RuntimeError(
            f"KIVI {name} cache layout mismatch: shape={tuple(blocks.shape)}, "
            f"expected (*, {max_blocks}, {cache_block_size}, ...)."
        )

    # -- 窗口簿记（纯 list 逻辑，无设备依赖）---------------------------------

    @staticmethod
    def cu_seqlens_to_seq_lens(cu_seqlens: Any) -> list[int]:
        """累积序列长度 cu_seqlens -> 各请求长度（差分）。"""
        if isinstance(cu_seqlens, torch.Tensor):
            cu_seqlens = cu_seqlens.tolist()
        prev = 0
        seq_lens = []
        for end in cu_seqlens:
            seq_lens.append(int(end) - prev)
            prev = int(end)
        return seq_lens

    @staticmethod
    def ordered_slots(
        block_table: Any, seq_lens: list[int], block_size: int
    ) -> list[list[int]]:
        """把 (block_id, 块内偏移) 展开成每个请求的绝对槽位 id 列表。

        绝对槽位 = block_id * block_size + 块内偏移；这是残差窗口与
        分页缓存之间的"地址簿"。
        """
        block_table = block_table.to(torch.long)
        ordered_slots: list[list[int]] = []
        for req_idx, seq_len in enumerate(seq_lens):
            req_slots: list[int] = []
            for pos in range(int(seq_len)):
                block_pos = pos // block_size
                block_offset = pos % block_size
                block_id = int(block_table[req_idx, block_pos].item())
                if block_id >= 0:
                    req_slots.append(block_id * block_size + block_offset)
            ordered_slots.append(req_slots)
        return ordered_slots

    def is_aligned_key_window(self, window_slots: list[int], block_size: int) -> bool:
        """键 flush 的对齐检查：每个组必须连续、块对齐、且完整落在同一个
        缓存块内（打包内核要求一个 int32 word 的 8 个 lane 全在同一块）。"""
        if not window_slots:
            return False
        group_size = self.group_size
        if len(window_slots) % group_size:
            return False

        for start in range(0, len(window_slots), group_size):
            group_slots = window_slots[start : start + group_size]
            if len(group_slots) != group_size:
                return False

            first_slot = int(group_slots[0])
            last_slot = int(group_slots[-1])
            block_idx = first_slot // block_size
            block_offset = first_slot % block_size

            # 组起点必须是 block 内 group 对齐的位置
            if block_offset % group_size != 0:
                return False
            # 组不能跨缓存块
            if last_slot // block_size != block_idx:
                return False

            expected = list(range(first_slot, first_slot + group_size))
            if group_slots != expected:
                return False
        return True

    def build_causal_mask(
        self, q_len: int, kv_seq_len: int, dtype: Any, device: Any
    ) -> Any:
        """纯 torch 稠密注意力路径的加性因果掩码（kv_pos > q_pos 处屏蔽）。"""
        q_pos = torch.arange(
            kv_seq_len - q_len, kv_seq_len, dtype=torch.long, device=device
        )
        kv_pos = torch.arange(kv_seq_len, dtype=torch.long, device=device)
        mask = kv_pos.unsqueeze(0) > q_pos.unsqueeze(1)
        neg_inf = torch.finfo(dtype).min
        return torch.where(
            mask,
            torch.full((), neg_inf, dtype=dtype, device=device),
            torch.zeros((), dtype=dtype, device=device),
        ).unsqueeze(0)

    def describe(self) -> dict[str, Any]:
        """方案语义的自描述。"""
        return {
            "scheme": "kivi_int4",
            "bits": self.bits,
            "group_size": self.group_size,
            "residual_length": self.config.residual_length,
            "key_grouping": "token groups (residual window flushed whole)",
            "value_grouping": "head-dim groups (per token)",
        }


__all__ = ["KiviInt4Semantics", "validate_kivi_config", "validate_kivi_geometry"]
