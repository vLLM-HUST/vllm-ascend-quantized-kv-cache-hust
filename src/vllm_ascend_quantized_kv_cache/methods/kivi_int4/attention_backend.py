# SPDX-License-Identifier: Apache-2.0
"""Plugin-owned port of the KIVI INT4 attention implementation.

移植自 legacy ascend PR #116 提交 0003-0009（AscendAttentionBackendImpl
KIVI 分支的最终状态）。设计总览：

* 分页缓存是 6 元组 ``(k_quant, k_scale, k_mn, v_quant, v_scale, v_mn)``。
  键：残差窗口按 token 组量化（组内逐 (head,dim) min/max）；
  值：每个 token 按 head 维组量化。
* 每个请求最近 ``kivi_residual_length`` 个 token 保持全精度，存在
  "残差行"里（环形缓冲，行数 = max_num_seqs，一行一个请求）。键窗口
  写满后整组 flush 进 int4 历史区；值则每挤掉一个最老槽位。
* 注意力计算：把 int4 历史区 gather + 反量化成稠密张量，交 NPU
  fused-inference 算子（TND 布局），再把全精度残差尾覆盖回最后几个
  token —— 即"历史区量化省显存，残差区保精度"。

实现体留在本仓库：适配器把它 mix 到活的宿主 ``AscendAttentionBackendImpl``
之上（见 adapters/vllm_ascend_hust/backend.py），插件持有 INT4 策略，
宿主提供 num_heads/head_size/scale 等构造期属性。
triton 内核 / torch_npu 在用到的方法里惰性导入；注意力状态按枚举成员
名字匹配，模块在任何宿主之外都能干净导入（stub 测试依赖这一点）。
"""

from __future__ import annotations

from typing import Any

import torch

from ...core.runtime import import_torch_npu
from .byte_cache import kivi_byte_cache_layout, kivi_caches_from_byte_tensors
from .geometry import validate_kivi_geometry
from .semantics import KiviInt4Semantics


def _attn_state_name(attn_metadata: Any) -> str:
    state = getattr(attn_metadata, "attn_state", None)
    if state is None:
        return "UNKNOWN"
    return getattr(state, "name", None) or str(state)


class AscendKiviInt4AttentionBackendMixin:
    """KIVI INT4 (int4 history + full-precision residual window) attention path."""

    # ------------------------------------------------------------------
    # 状态：方案属性 + 布局校验 + 6 元组缓存绑定
    # ------------------------------------------------------------------

    def _init_kivi_state(
        self, kv_cache_dtype: str | None, vllm_config: Any = None
    ) -> None:
        """建立 KIVI 全部实例状态。

        group_size / residual_length 优先读宿主 cache_config 的
        kivi_group_size / kivi_residual_length（默认 128/128）；
        残差行数取 scheduler 的 max_num_seqs（一行服务一个请求）。
        """
        self.enable_kivi = kv_cache_dtype in ("kivi_int4", "kivi")
        cache_config = getattr(vllm_config, "cache_config", None)
        group_size = getattr(cache_config, "kivi_group_size", 128)
        residual_length = getattr(cache_config, "kivi_residual_length", 128)
        self.kivi_group_size = group_size if isinstance(group_size, int) else 128
        self.kivi_residual_length = (
            residual_length if isinstance(residual_length, int) else 128
        )
        if self.enable_kivi:
            self._validate_kivi_geometry()
        self.kivi_bits = 4
        self.k_quant_cache = None
        self.k_scale_cache = None
        self.k_mn_cache = None
        self.v_quant_cache = None
        self.v_scale_cache = None
        self.v_mn_cache = None
        # Full-precision residual window, one row per live request.
        self.kivi_residual_key_cache: torch.Tensor | None = None
        self.kivi_residual_value_cache: torch.Tensor | None = None
        self.kivi_residual_key_slot_ids: torch.Tensor | None = None
        self.kivi_residual_value_slot_ids: torch.Tensor | None = None
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        max_num_seqs = getattr(scheduler_config, "max_num_seqs", 128)
        self.kivi_max_num_seqs = (
            max_num_seqs if isinstance(max_num_seqs, int) and max_num_seqs > 0 else 128
        )
        self.kivi_residual_req_to_row: dict[str, int] = {}
        self.kivi_residual_row_to_req: list[str | None] = []
        self.kivi_residual_free_rows: list[int] = []
        self.kivi_residual_key_start: list[int] = []
        self.kivi_residual_key_len: list[int] = []
        self.kivi_residual_value_start: list[int] = []
        self.kivi_residual_value_len: list[int] = []

    def _validate_kivi_geometry(self) -> None:
        """Run the layout invariants against live impl attributes."""
        validate_kivi_geometry(
            head_size=self.head_size,
            group_size=self.kivi_group_size,
            residual_length=self.kivi_residual_length,
        )

    def _kivi_sem(self) -> KiviInt4Semantics:
        return KiviInt4Semantics(self)

    def _check_kivi_cache_bound(self) -> None:
        if (
            self.k_quant_cache is None
            or self.k_scale_cache is None
            or self.k_mn_cache is None
            or self.v_quant_cache is None
            or self.v_scale_cache is None
            or self.v_mn_cache is None
        ):
            raise RuntimeError("KIVI cache tensors are not bound.")

    def _get_kivi_block_size(self) -> int:
        self._check_kivi_cache_bound()
        return self.k_quant_cache.shape[-1] * 8

    # ------------------------------------------------------------------
    # 缓存绑定：6 元组（k/v 各 quant+scale+mn），或宿主给的两张字节缓冲
    # ------------------------------------------------------------------

    def _bind_kivi_cache(self, kv_cache: Any) -> None:
        """绑定分页缓存；首次绑定时顺带分配残差缓冲。

        vLLM 只会把两张张量交给 impl，所以两元素形态是常规入口：按 INT4
        的区域布局切成 6 个视图（不复制数据）。六元素形态保留给已经在树里
        按 6 张分配的宿主。
        """
        if kv_cache is None:
            return
        if not isinstance(kv_cache, list | tuple):
            raise RuntimeError(
                "KIVI INT4 kv_cache must be a 2-tuple of byte buffers or a "
                f"6-tuple, got {type(kv_cache)}."
            )
        if len(kv_cache) == 0:
            return
        if len(kv_cache) == 2:
            kv_cache = kivi_caches_from_byte_tensors(
                *kv_cache,
                layout=kivi_byte_cache_layout(
                    kv_cache[0],
                    kv_cache[1],
                    num_kv_heads=self.num_kv_heads,
                    head_size=self.head_size,
                    group_size=self.kivi_group_size,
                ),
            )
        if len(kv_cache) != 6:
            raise RuntimeError(
                "KIVI INT4 kv_cache must be a 2-tuple of byte buffers or a "
                "6-tuple: (k_quant, k_scale, k_mn, v_quant, v_scale, v_mn)."
            )
        (
            self.k_quant_cache,
            self.k_scale_cache,
            self.k_mn_cache,
            self.v_quant_cache,
            self.v_scale_cache,
            self.v_mn_cache,
        ) = kv_cache
        self._ensure_kivi_residual_buffers()

    # ------------------------------------------------------------------
    # 残差行：每个请求占一行（行数 = max_num_seqs），请求结束回收
    # ------------------------------------------------------------------

    def _ensure_kivi_residual_buffers(self) -> None:
        """惰性分配残差环形缓冲（每请求一行 × residual_length 槽）。

        slot_ids 记录每个 lane 存的是哪个绝对槽位（-1 = 空闲）；
        start/len 构成环形队列的读出顺序。
        """
        if self.kivi_residual_key_cache is not None:
            return

        self._check_kivi_cache_bound()
        num_rows = self.kivi_max_num_seqs
        residual_slots = self.kivi_residual_length
        device = self.k_quant_cache.device

        self.kivi_residual_key_cache = torch.empty(
            (num_rows, residual_slots, self.num_kv_heads, self.head_size),
            dtype=self.k_scale_cache.dtype,
            device=device,
        )
        self.kivi_residual_value_cache = torch.empty(
            (num_rows, residual_slots, self.num_kv_heads, self.head_size),
            dtype=self.v_scale_cache.dtype,
            device=device,
        )
        self.kivi_residual_key_slot_ids = torch.full(
            (num_rows, residual_slots), -1, dtype=torch.long, device=device
        )
        self.kivi_residual_value_slot_ids = torch.full(
            (num_rows, residual_slots), -1, dtype=torch.long, device=device
        )
        self.kivi_residual_req_to_row = {}
        self.kivi_residual_row_to_req = [None] * num_rows
        self.kivi_residual_free_rows = list(range(num_rows - 1, -1, -1))
        self.kivi_residual_key_start = [0] * num_rows
        self.kivi_residual_key_len = [0] * num_rows
        self.kivi_residual_value_start = [0] * num_rows
        self.kivi_residual_value_len = [0] * num_rows

    def _get_kivi_req_keys(self, req_ids: list[str] | None, num_reqs: int) -> list[str]:
        if req_ids is not None and len(req_ids) >= num_reqs:
            return [str(req_id) for req_id in req_ids[:num_reqs]]
        return [f"__kivi_batch_row_{idx}" for idx in range(num_reqs)]

    def _get_kivi_residual_row(self, req_key: str, *, create: bool) -> int | None:
        self._ensure_kivi_residual_buffers()
        row_idx = self.kivi_residual_req_to_row.get(req_key)
        if row_idx is not None or not create:
            return row_idx

        if not self.kivi_residual_free_rows:
            raise RuntimeError(
                "KIVI residual rows are exhausted. "
                f"max_num_seqs={self.kivi_max_num_seqs}, req_id={req_key}."
            )

        row_idx = self.kivi_residual_free_rows.pop()
        self.kivi_residual_req_to_row[req_key] = row_idx
        self.kivi_residual_row_to_req[row_idx] = req_key
        return row_idx

    def _release_kivi_residual_row(self, req_key: str) -> None:
        row_idx = self.kivi_residual_req_to_row.pop(req_key, None)
        if row_idx is None:
            return

        self.kivi_residual_row_to_req[row_idx] = None
        if self.kivi_residual_key_slot_ids is not None:
            self.kivi_residual_key_slot_ids[row_idx].fill_(-1)
        if self.kivi_residual_value_slot_ids is not None:
            self.kivi_residual_value_slot_ids[row_idx].fill_(-1)
        self.kivi_residual_key_start[row_idx] = 0
        self.kivi_residual_key_len[row_idx] = 0
        self.kivi_residual_value_start[row_idx] = 0
        self.kivi_residual_value_len[row_idx] = 0
        self.kivi_residual_free_rows.append(row_idx)

    def _release_finished_kivi_residual_rows(self, finished_req_keys: set[str]) -> None:
        for req_key in finished_req_keys:
            self._release_kivi_residual_row(req_key)

    # ------------------------------------------------------------------
    # 残差环形缓冲：行内是 start/len 环形队列，写满按策略挤前缀
    # ------------------------------------------------------------------

    def _get_kivi_residual_buffers(
        self, *, is_key: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_kivi_residual_buffers()
        slot_ids = (
            self.kivi_residual_key_slot_ids
            if is_key
            else self.kivi_residual_value_slot_ids
        )
        cache = (
            self.kivi_residual_key_cache if is_key else self.kivi_residual_value_cache
        )
        assert slot_ids is not None and cache is not None
        return slot_ids, cache

    def _get_kivi_residual_state(self, *, is_key: bool) -> tuple[list[int], list[int]]:
        self._ensure_kivi_residual_buffers()
        if is_key:
            return self.kivi_residual_key_start, self.kivi_residual_key_len
        return self.kivi_residual_value_start, self.kivi_residual_value_len

    def _get_kivi_residual_lanes(self, row_idx: int, *, is_key: bool) -> list[int]:
        """按环形（FIFO）顺序返回该行已占用 lane 的下标列表。

        start+len 越过容量时回绕：range(start, cap) + range(0, ...)。
        """
        starts, lengths = self._get_kivi_residual_state(is_key=is_key)
        length = lengths[row_idx]
        if length <= 0:
            return []
        start = starts[row_idx]
        cap = self.kivi_residual_length
        end = start + length
        if end <= cap:
            return list(range(start, end))
        return list(range(start, cap)) + list(range(0, end % cap))

    def _collect_kivi_residual_window(
        self,
        req_key: str,
        *,
        is_key: bool,
        target_dtype: Any | None = None,
        target_device: Any | None = None,
    ) -> tuple[list[int], torch.Tensor | None]:
        slot_ids, cache = self._get_kivi_residual_buffers(is_key=is_key)
        row_idx = self._get_kivi_residual_row(req_key, create=False)
        if row_idx is None:
            return [], None
        lanes = self._get_kivi_residual_lanes(row_idx, is_key=is_key)
        if not lanes:
            return [], None
        lane_tensor = torch.tensor(lanes, dtype=torch.long, device=cache.device)
        slot_tensor = slot_ids[row_idx].index_select(0, lane_tensor)
        tensors = cache[row_idx].index_select(0, lane_tensor)
        if target_dtype is not None or target_device is not None:
            if target_dtype is None:
                target_dtype = tensors.dtype
            if target_device is None:
                target_device = tensors.device
            tensors = tensors.to(device=target_device, dtype=target_dtype)
        return [int(slot) for slot in slot_tensor.tolist()], tensors

    def _set_kivi_residual_window(
        self,
        req_key: str,
        slots: list[int],
        tensors: torch.Tensor | None,
        *,
        is_key: bool,
    ) -> None:
        row_idx = self._get_kivi_residual_row(req_key, create=bool(slots))
        if row_idx is None:
            return
        slot_ids, cache = self._get_kivi_residual_buffers(is_key=is_key)
        starts, lengths = self._get_kivi_residual_state(is_key=is_key)
        slot_ids[row_idx].fill_(-1)
        starts[row_idx] = 0
        lengths[row_idx] = len(slots)
        if not slots:
            return
        if len(slots) > self.kivi_residual_length:
            raise RuntimeError(
                "KIVI residual window exceeds capacity: "
                f"{len(slots)} > {self.kivi_residual_length}."
            )
        if tensors is None:
            raise RuntimeError(
                "KIVI residual tensors are required when resetting a non-empty window."
            )
        slot_tensor = torch.tensor(slots, dtype=torch.long, device=slot_ids.device)
        slot_ids[row_idx, : len(slots)] = slot_tensor
        cache[row_idx, : len(slots)] = tensors.to(
            device=cache.device, dtype=cache.dtype
        ).contiguous()

    def _get_kivi_residual_tail(
        self,
        req_key: str,
        *,
        is_key: bool,
        target_dtype: Any,
        target_device: Any,
    ) -> torch.Tensor | None:
        _, tensors = self._collect_kivi_residual_window(
            req_key,
            is_key=is_key,
            target_dtype=target_dtype,
            target_device=target_device,
        )
        return tensors

    def _flush_kivi_residual_prefix(
        self, req_key: str, *, is_key: bool, count: int
    ) -> None:
        """把残差窗口最老的 count 个槽位 flush 进 int4 历史区。

        键窗口满时 count = 整个窗口（整组 flush）；值则 count = 1
        （每次挤掉一个最老槽位）——这是 legacy 的非对称策略。
        """
        row_idx = self._get_kivi_residual_row(req_key, create=False)
        if row_idx is None:
            return
        slot_ids, cache = self._get_kivi_residual_buffers(is_key=is_key)
        starts, lengths = self._get_kivi_residual_state(is_key=is_key)
        length = lengths[row_idx]
        if length <= 0:
            return
        count = min(count, length)
        lanes = self._get_kivi_residual_lanes(row_idx, is_key=is_key)[:count]
        if not lanes:
            return
        lane_tensor = torch.tensor(lanes, dtype=torch.long, device=cache.device)
        slot_tensor = slot_ids[row_idx].index_select(0, lane_tensor)
        tensors = cache[row_idx].index_select(0, lane_tensor).contiguous()
        if is_key:
            self._write_kivi_key_quant_cache(tensors, slot_tensor)
        else:
            self._write_kivi_value_quant_cache(tensors, slot_tensor)
        slot_ids[row_idx, lane_tensor] = -1
        lengths[row_idx] = length - count
        if lengths[row_idx] == 0:
            starts[row_idx] = 0
        else:
            starts[row_idx] = (starts[row_idx] + count) % self.kivi_residual_length

    def _store_kivi_residual_entries(
        self, values: torch.Tensor, slots: torch.Tensor, *, req_key: str, is_key: bool
    ) -> None:
        """向请求的残差环形行追加 token；写满则先 flush 前缀再写入。

        逐槽处理：命中已有 lane 就原地覆盖；否则走环形尾写入，
        写入前 while 循环保证有空间（键挤整窗、值挤一个）。
        """
        if values.numel() == 0 or slots.numel() == 0:
            return

        slot_ids, cache = self._get_kivi_residual_buffers(is_key=is_key)
        starts, lengths = self._get_kivi_residual_state(is_key=is_key)
        values = values.to(cache.dtype).contiguous()
        slots = slots.to(torch.long).contiguous()
        row_idx = self._get_kivi_residual_row(req_key, create=True)
        assert row_idx is not None

        for idx in range(int(slots.shape[0])):
            slot = int(slots[idx].item())
            match = (slot_ids[row_idx] == slot).nonzero(as_tuple=False)
            if match.numel() > 0:
                lane = int(match[0].item())
                cache[row_idx, lane] = values[idx]
                continue

            while lengths[row_idx] >= self.kivi_residual_length:
                self._flush_kivi_residual_prefix(
                    req_key,
                    is_key=is_key,
                    count=self.kivi_residual_length if is_key else 1,
                )

            tail = (starts[row_idx] + lengths[row_idx]) % self.kivi_residual_length
            slot_ids[row_idx, tail] = slot
            cache[row_idx, tail] = values[idx]
            lengths[row_idx] += 1

    def _lookup_kivi_residual_tensor(
        self,
        slot: int,
        *,
        req_key: str,
        is_key: bool,
        target_dtype: Any,
        target_device: Any | None = None,
    ) -> torch.Tensor | None:
        slot_ids, cache = self._get_kivi_residual_buffers(is_key=is_key)
        row_idx = self._get_kivi_residual_row(req_key, create=False)
        if row_idx is None:
            return None
        slot_row = slot_ids[row_idx]
        match = (slot_row == slot).nonzero(as_tuple=False)
        if match.numel() == 0:
            return None

        lane = int(match[0].item())
        tensor = cache[row_idx, lane]
        if target_device is None:
            target_device = tensor.device
        return tensor.to(device=target_device, dtype=target_dtype)

    def _gather_kivi_residual_tensors(
        self, slots: list[int], *, req_key: str, is_key: bool
    ) -> torch.Tensor:
        if not slots:
            raise RuntimeError("KIVI residual gather expects a non-empty slot list.")

        target_dtype = self.k_scale_cache.dtype if is_key else self.v_scale_cache.dtype
        tensors = []
        cache_name = "key" if is_key else "value"
        for slot in slots:
            tensor = self._lookup_kivi_residual_tensor(
                int(slot),
                req_key=req_key,
                is_key=is_key,
                target_dtype=target_dtype,
            )
            if tensor is None:
                raise RuntimeError(
                    f"KIVI {cache_name} residual tensor for slot {slot} is "
                    "missing before flush."
                )
            tensors.append(tensor)
        return torch.stack(tensors, dim=0)

    def _has_kivi_residual_entry(
        self, slot: int, *, req_key: str, is_key: bool
    ) -> bool:
        slot_ids, _ = self._get_kivi_residual_buffers(is_key=is_key)
        row_idx = self._get_kivi_residual_row(req_key, create=False)
        if row_idx is None:
            return False
        return bool((slot_ids[row_idx] == slot).any().item())

    def _clear_kivi_residual_entries(
        self, slots: list[int], *, req_key: str, is_key: bool
    ) -> None:
        if not slots:
            return
        current_slots, current_tensors = self._collect_kivi_residual_window(
            req_key, is_key=is_key
        )
        if not current_slots or current_tensors is None:
            return
        drop = {int(slot) for slot in slots}
        keep_indices = [
            idx for idx, slot in enumerate(current_slots) if slot not in drop
        ]
        if len(keep_indices) == len(current_slots):
            return
        if not keep_indices:
            self._set_kivi_residual_window(req_key, [], None, is_key=is_key)
            return
        keep_tensor = current_tensors.index_select(
            0,
            torch.tensor(keep_indices, dtype=torch.long, device=current_tensors.device),
        )
        keep_slots = [current_slots[idx] for idx in keep_indices]
        self._set_kivi_residual_window(req_key, keep_slots, keep_tensor, is_key=is_key)

    def _clear_stale_kivi_residual_entries(
        self, req_key: str, live_slots: set[int], *, is_key: bool
    ) -> None:
        current_slots, current_tensors = self._collect_kivi_residual_window(
            req_key, is_key=is_key
        )
        if not current_slots or current_tensors is None:
            return
        keep_indices = [
            idx for idx, slot in enumerate(current_slots) if int(slot) in live_slots
        ]
        if len(keep_indices) == len(current_slots):
            return
        if not keep_indices:
            self._set_kivi_residual_window(req_key, [], None, is_key=is_key)
            return
        keep_tensor = current_tensors.index_select(
            0,
            torch.tensor(keep_indices, dtype=torch.long, device=current_tensors.device),
        )
        keep_slots = [current_slots[idx] for idx in keep_indices]
        self._set_kivi_residual_window(req_key, keep_slots, keep_tensor, is_key=is_key)

    # ------------------------------------------------------------------
    # 写入 / flush：残差 -> 分页 int4 历史区（经 triton 打包内核）
    # ------------------------------------------------------------------

    def _write_kivi_cache(
        self,
        key: Any,
        value: Any,
        slot_mapping: Any,
        req_ids: list[str] | None,
        actual_seq_qlen: list[int] | Any,
    ) -> None:
        if key is None or value is None or slot_mapping is None:
            return

        if isinstance(actual_seq_qlen, torch.Tensor):
            actual_seq_qlen = actual_seq_qlen.tolist()

        if not actual_seq_qlen:
            return

        req_keys = self._get_kivi_req_keys(req_ids, len(actual_seq_qlen))
        max_tokens = int(slot_mapping.shape[0])
        prev_q_end = 0
        for req_idx, q_end in enumerate(actual_seq_qlen):
            q_end = min(int(q_end), max_tokens)
            if prev_q_end >= max_tokens:
                break
            if q_end <= prev_q_end:
                prev_q_end = q_end
                continue

            req_slots = slot_mapping[prev_q_end:q_end]
            valid = req_slots >= 0
            if bool(valid.any()):
                req_key = req_keys[req_idx]
                self._store_kivi_residual_entries(
                    key[prev_q_end:q_end][valid].detach(),
                    req_slots[valid].to(torch.long),
                    req_key=req_key,
                    is_key=True,
                )
                self._store_kivi_residual_entries(
                    value[prev_q_end:q_end][valid].detach(),
                    req_slots[valid].to(torch.long),
                    req_key=req_key,
                    is_key=False,
                )
            prev_q_end = q_end

    def _write_kivi_prefill_cache(
        self,
        key: Any,
        value: Any,
        slot_mapping: Any,
        req_ids: list[str] | None,
        actual_seq_qlen: list[int] | Any,
    ) -> None:
        if key is None or value is None or slot_mapping is None:
            return

        if isinstance(actual_seq_qlen, torch.Tensor):
            actual_seq_qlen = actual_seq_qlen.tolist()

        if not actual_seq_qlen:
            return

        req_keys = self._get_kivi_req_keys(req_ids, len(actual_seq_qlen))
        max_tokens = int(slot_mapping.shape[0])
        prev_q_end = 0
        for req_idx, q_end in enumerate(actual_seq_qlen):
            q_end = min(int(q_end), max_tokens)
            if prev_q_end >= max_tokens:
                break
            if q_end <= prev_q_end:
                prev_q_end = q_end
                continue

            req_slots = slot_mapping[prev_q_end:q_end]
            valid = req_slots >= 0
            if not bool(valid.any()):
                prev_q_end = q_end
                continue

            req_key = req_keys[req_idx]
            req_key_states = key[prev_q_end:q_end][valid].detach()
            req_value_states = value[prev_q_end:q_end][valid].detach()
            req_slots = req_slots[valid].to(torch.long)

            num_valid_tokens = int(req_slots.shape[0])
            flush_tokens = (
                num_valid_tokens // self.kivi_residual_length
            ) * self.kivi_residual_length

            if flush_tokens > 0:
                flush_slots = req_slots[:flush_tokens].contiguous()
                self._write_kivi_key_quant_cache(
                    req_key_states[:flush_tokens].contiguous(), flush_slots
                )
                self._write_kivi_value_quant_cache(
                    req_value_states[:flush_tokens].contiguous(), flush_slots
                )

            if flush_tokens < num_valid_tokens:
                self._store_kivi_residual_entries(
                    req_key_states[flush_tokens:],
                    req_slots[flush_tokens:],
                    req_key=req_key,
                    is_key=True,
                )
                self._store_kivi_residual_entries(
                    req_value_states[flush_tokens:],
                    req_slots[flush_tokens:],
                    req_key=req_key,
                    is_key=False,
                )
            prev_q_end = q_end

    def _write_kivi_key_quant_cache(self, key: Any, slot_mapping: Any) -> None:
        """键整组 flush 入历史区：先做全部布局校验，再发 triton 打包内核。

        校验清单：head 对齐、整组 token、word 对齐、组大小、块大小、
        槽位连续且块对齐——任何不满足都在发内核前 fail-closed。
        """
        self._check_kivi_cache_bound()
        if key is None or slot_mapping is None:
            return

        valid = slot_mapping >= 0
        if not bool(valid.any()):
            return

        key = key[valid].to(self.k_scale_cache.dtype).contiguous()
        slots = slot_mapping[valid].to(torch.long).contiguous()

        if key.shape[-1] != self.head_size:
            raise RuntimeError(
                "KIVI INT4 key head_size must match attention head_size "
                f"({self.head_size}), got {key.shape[-1]}."
            )
        if key.shape[0] % self.kivi_group_size != 0:
            raise RuntimeError(
                "KIVI key flush must contain whole token groups, got "
                f"{key.shape[0]} tokens for group_size={self.kivi_group_size}."
            )

        if self.head_size % 8 != 0:
            raise RuntimeError(
                f"KIVI INT4 int32 packing requires head_size ({self.head_size}) "
                "to be divisible by 8."
            )
        if self.kivi_group_size % 8 != 0:
            raise RuntimeError(
                "KIVI INT4 key packing requires kivi_group_size "
                f"({self.kivi_group_size}) to be divisible by 8."
            )

        block_size = self._get_kivi_block_size()
        if block_size % self.kivi_group_size != 0:
            raise RuntimeError(
                f"KIVI INT4 key cache requires block_size ({block_size}) to be "
                f"divisible by kivi_group_size ({self.kivi_group_size})."
            )
        if not self._kivi_sem().is_aligned_key_window(slots.tolist(), block_size):
            raise RuntimeError(
                "KIVI key flush requires contiguous aligned token groups."
            )

        self._launch_kivi_key_pack(key, slots)

    def _launch_kivi_key_pack(self, key: Any, slots: Any) -> None:
        """Kernel launch hook (isolated so tests can stub it)."""
        from ...ops.triton.kivi_pack import kivi_pack_key_cache

        kivi_pack_key_cache(
            key,
            slots,
            self.k_quant_cache,
            self.k_scale_cache,
            self.k_mn_cache,
            self.kivi_group_size,
        )

    def _write_kivi_value_quant_cache(self, value: Any, slot_mapping: Any) -> None:
        self._check_kivi_cache_bound()
        if value is None or slot_mapping is None:
            return

        valid = slot_mapping >= 0
        if not bool(valid.any()):
            return

        value = value[valid].to(self.v_scale_cache.dtype).contiguous()
        slots = slot_mapping[valid].to(torch.long).contiguous()
        head_size = value.shape[-1]

        if head_size != self.head_size:
            raise RuntimeError(
                "KIVI INT4 value head_size must match attention head_size "
                f"({self.head_size}), got {head_size}."
            )

        if head_size % self.kivi_group_size != 0:
            raise RuntimeError(
                f"KIVI INT4 value head_size ({head_size}) must be divisible by "
                f"kivi_group_size ({self.kivi_group_size})."
            )

        if head_size % 8 != 0:
            raise RuntimeError(
                f"KIVI INT4 int32 packing requires head_size ({head_size}) "
                "to be divisible by 8."
            )
        if self.kivi_group_size % 8 != 0:
            raise RuntimeError(
                "KIVI INT4 value packing requires kivi_group_size "
                f"({self.kivi_group_size}) to be divisible by 8."
            )

        self._launch_kivi_value_pack(value, slots)

    def _launch_kivi_value_pack(self, value: Any, slots: Any) -> None:
        """Kernel launch hook (isolated so tests can stub it)."""
        from ...ops.triton.kivi_pack import kivi_pack_value_cache

        kivi_pack_value_cache(
            value,
            slots,
            self.v_quant_cache,
            self.v_scale_cache,
            self.v_mn_cache,
            self.kivi_group_size,
        )

    def _is_aligned_kivi_key_window(self, window_slots: list[int]) -> bool:
        return self._kivi_sem().is_aligned_key_window(
            window_slots, self._get_kivi_block_size()
        )

    def _flush_kivi_slots(
        self, slots: list[int], *, req_key: str, is_key: bool
    ) -> None:
        if not slots:
            return

        tensors = self._gather_kivi_residual_tensors(
            slots, req_key=req_key, is_key=is_key
        )
        slot_tensor = torch.tensor(slots, dtype=torch.long, device=tensors.device)
        if is_key:
            self._write_kivi_key_quant_cache(tensors, slot_tensor)
        else:
            self._write_kivi_value_quant_cache(tensors, slot_tensor)
        self._clear_kivi_residual_entries(slots, req_key=req_key, is_key=is_key)

    def _flush_kivi_key_batches(self, req_key: str, window_slots: list[int]) -> None:
        while len(window_slots) >= self.kivi_residual_length:
            flush_slots = window_slots[: self.kivi_residual_length]
            if not self._is_aligned_kivi_key_window(flush_slots):
                break
            self._flush_kivi_slots(flush_slots, req_key=req_key, is_key=True)
            del window_slots[: self.kivi_residual_length]

    def _flush_kivi_value_batches(self, req_key: str, window_slots: list[int]) -> None:
        if len(window_slots) < self.kivi_residual_length:
            return

        flush_slots = window_slots[:1]
        self._flush_kivi_slots(flush_slots, req_key=req_key, is_key=False)
        del window_slots[:1]

    def _sync_kivi_residual_windows(
        self,
        block_table: Any,
        seq_lens: list[int],
        req_ids: list[str] | None,
        finished_req_ids: set[str] | None = None,
    ) -> None:
        """每步同步：回收 finished 请求的残差行 + 清理各请求的过期槽位。

        请求被抢占/换块后，残差行里可能残留不再属于它的槽位，
        全部清掉，防止读到陈旧的全精度值。
        """
        req_keys = self._get_kivi_req_keys(req_ids, len(seq_lens))
        ordered_slots = self._get_kivi_ordered_slots(block_table, seq_lens)
        if finished_req_ids:
            self._release_finished_kivi_residual_rows(
                {str(req_id) for req_id in finished_req_ids}
            )

        for req_idx, req_slots in enumerate(ordered_slots):
            req_key = req_keys[req_idx]
            live_slots = set(req_slots)
            self._clear_stale_kivi_residual_entries(req_key, live_slots, is_key=True)
            self._clear_stale_kivi_residual_entries(req_key, live_slots, is_key=False)

    # ------------------------------------------------------------------
    # gather + 反量化：int4 历史区 -> 稠密张量（残差尾覆盖回尾部）
    # ------------------------------------------------------------------

    def _get_kivi_ordered_slots(
        self, block_table: Any, seq_lens: list[int]
    ) -> list[list[int]]:
        self._check_kivi_cache_bound()
        block_size = self._get_kivi_block_size()
        return KiviInt4Semantics.ordered_slots(block_table, seq_lens, block_size)

    def _gather_dequant_kivi_paged_cache(
        self,
        block_table: Any,
        seq_lens: list[int],
        target_dtype: Any,
        req_ids: list[str] | None = None,
    ) -> tuple[Any, Any]:
        """int4 历史区 -> 稠密 K/V，并把全精度残差尾覆盖回每请求尾部。

        triton dequant-gather（torch 路径，见 ops.kivi_gather）产出历史
        区的稠密张量；随后每请求取残差窗口里的张量按尾部对齐覆盖——
        历史区里这些槽位还是旧的量化值，覆盖后才与真实 KV 一致。
        """
        from ...ops.kivi_gather import kivi_dequant_gather_cache

        cache_device = self.k_quant_cache.device
        batch_size = len(seq_lens)
        self._ensure_kivi_residual_buffers()
        block_table = (
            block_table[:batch_size]
            .to(device=cache_device, dtype=torch.long)
            .contiguous()
        )
        seq_lens_t = torch.tensor(
            seq_lens, dtype=torch.long, device=cache_device
        ).contiguous()
        req_keys = self._get_kivi_req_keys(req_ids, batch_size)
        dense_k, dense_v = kivi_dequant_gather_cache(
            self.k_quant_cache,
            self.k_scale_cache,
            self.k_mn_cache,
            self.v_quant_cache,
            self.v_scale_cache,
            self.v_mn_cache,
            block_table,
            seq_lens_t,
            target_dtype,
            self.kivi_group_size,
        )

        dense_k_parts: list[torch.Tensor] = []
        dense_v_parts: list[torch.Tensor] = []
        kv_start = 0
        for req_idx, seq_len in enumerate(seq_lens):
            req_len = int(seq_len)
            req_k = dense_k[kv_start : kv_start + req_len]
            req_v = dense_v[kv_start : kv_start + req_len]
            kv_start += req_len

            req_key = req_keys[req_idx]
            residual_key = self._get_kivi_residual_tail(
                req_key,
                is_key=True,
                target_dtype=target_dtype,
                target_device=req_k.device,
            )
            if residual_key is not None and residual_key.shape[0] > 0 and req_len > 0:
                tail_len = min(req_len, int(residual_key.shape[0]))
                req_k = req_k.clone()
                req_k[-tail_len:] = residual_key[-tail_len:]

            residual_value = self._get_kivi_residual_tail(
                req_key,
                is_key=False,
                target_dtype=target_dtype,
                target_device=req_v.device,
            )
            if (
                residual_value is not None
                and residual_value.shape[0] > 0
                and req_len > 0
            ):
                tail_len = min(req_len, int(residual_value.shape[0]))
                req_v = req_v.clone()
                req_v[-tail_len:] = residual_value[-tail_len:]

            dense_k_parts.append(req_k)
            dense_v_parts.append(req_v)

        if not dense_k_parts:
            empty = dense_k.new_empty((0, self.num_kv_heads, self.head_size))
            return empty, empty

        return (
            torch.cat(dense_k_parts, dim=0).contiguous(),
            torch.cat(dense_v_parts, dim=0).contiguous(),
        )

    # ------------------------------------------------------------------
    # 注意力路径：按注意力状态分派（prefill/decode/chunked/兜底稠密）
    # ------------------------------------------------------------------

    def _repeat_kv(self, tensor: Any) -> Any:
        if self.num_queries_per_kv == 1:
            return tensor
        return tensor.repeat_interleave(self.num_queries_per_kv, dim=1)

    def _build_kivi_causal_mask(
        self, q_len: int, kv_seq_len: int, dtype: Any, device: Any
    ) -> Any:
        return self._kivi_sem().build_causal_mask(q_len, kv_seq_len, dtype, device)

    def _unpack_int4(self, packed: Any) -> Any:
        return KiviInt4Semantics.unpack_int4(packed)

    def _pack_int4(self, quant: Any) -> Any:
        return KiviInt4Semantics.pack_int4(quant)

    @staticmethod
    def _cu_seqlens_to_seq_lens(cu_seqlens: Any) -> list[int]:
        return KiviInt4Semantics.cu_seqlens_to_seq_lens(cu_seqlens)

    def _is_kivi_chunked_prefill_all_new(self, attn_metadata: Any) -> bool:
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_decodes = attn_metadata.num_decodes

        for req_idx in range(num_decodes, len(attn_metadata.seq_lens_list)):
            q_start = actual_seq_qlen[req_idx - 1] if req_idx > 0 else 0
            qlen_i = actual_seq_qlen[req_idx] - q_start
            if attn_metadata.seq_lens_list[req_idx] > qlen_i:
                return False
        return True

    def forward(
        self,
        layer: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Any,
        attn_metadata: Any,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """INT4 forward entry owned by the plugin.

        KIVI writes the paged cache through its own packing kernels, so this
        path deliberately bypasses the host ``reshape_and_cache``.
        """
        assert output is not None, "Output tensor must be provided."
        if not getattr(self, "enable_kivi", False):
            raise RuntimeError(
                "the KIVI INT4 attention impl was instantiated without the "
                "kivi_int4 cache dtype; check the host get_impl_cls dispatch"
            )
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for "
                "AscendKiviInt4AttentionBackendImpl"
            )
        if attn_metadata is None:
            return output.fill_(0)
        return self._forward_kivi_attention(
            query, key, value, attn_metadata, output, kv_cache
        )

    def _forward_kivi_attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        attn_metadata: Any,
        output: Any,
        kv_cache: Any = None,
    ) -> Any:
        """KIVI 的 forward 总入口：回收 finished 行 -> 写入 -> 按状态分派。

        分派规则：
          * PrefillNoCache：整段直接量化入历史 + FIA prefill（不进残差）；
          * DecodeOnly：写残差 -> 同步窗口 -> gather+反量化 -> TND FIA；
          * ChunkedPrefill：只支持"decode 行 + 全新 prefill"组合，
            prefill 带历史的场景直接 RuntimeError（fail-closed）；
          * 其他状态（PrefillCacheHit 等）：写 + 同步 + 纯 torch 稠密兜底。
        """
        self._bind_kivi_cache(kv_cache)

        finished_req_ids = getattr(attn_metadata, "finished_req_ids", None)
        if finished_req_ids:
            # Reclaim per-request residual rows before this step writes new K/V.
            self._release_finished_kivi_residual_rows(
                {str(req_id) for req_id in finished_req_ids}
            )

        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        if _attn_state_name(attn_metadata) == "DecodeOnly":
            actual_seq_qlen = list(range(1, len(attn_metadata.seq_lens_list) + 1))
        num_tokens = int(actual_seq_qlen[-1])
        query = query[:num_tokens]

        if _attn_state_name(attn_metadata) == "PrefillNoCache":
            if key is None or value is None:
                raise RuntimeError("KIVI PrefillNoCache requires dense key/value.")
            num_actual_tokens = min(
                getattr(attn_metadata, "num_actual_tokens", key.shape[0]),
                key.shape[0],
            )
            self._write_kivi_prefill_cache(
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                attn_metadata.slot_mapping[:num_actual_tokens],
                getattr(attn_metadata, "req_ids", None),
                attn_metadata.actual_seq_lengths_q,
            )
            return self._forward_kivi_prefill_fia(
                query=query,
                key=key,
                value=value,
                attn_metadata=attn_metadata,
                output=output,
            )

        state = _attn_state_name(attn_metadata)
        if state == "ChunkedPrefill" and not self._is_kivi_chunked_prefill_all_new(
            attn_metadata
        ):
            raise RuntimeError(
                "KIVI ChunkedPrefill does not support prefill requests with "
                "historical KV cache. Only decode rows plus all-new prefill "
                "prompts are supported."
            )

        seq_lens = attn_metadata.seq_lens_list
        if state == "DecodeOnly":
            if key is not None and value is not None:
                num_actual_tokens = min(
                    getattr(attn_metadata, "num_actual_tokens", key.shape[0]),
                    key.shape[0],
                )
                if num_actual_tokens > 0:
                    self._write_kivi_cache(
                        key[:num_actual_tokens],
                        value[:num_actual_tokens],
                        attn_metadata.slot_mapping[:num_actual_tokens],
                        getattr(attn_metadata, "req_ids", None),
                        list(range(1, len(seq_lens) + 1)),
                    )
            self._sync_kivi_residual_windows(
                attn_metadata.block_tables,
                seq_lens,
                getattr(attn_metadata, "req_ids", None),
                finished_req_ids,
            )
            return self._forward_kivi_paged_decode_attention(
                query,
                attn_metadata.block_tables,
                seq_lens,
                getattr(attn_metadata, "req_ids", None),
                output,
            )
        if state == "ChunkedPrefill":
            if key is not None and value is not None:
                num_actual_tokens = min(
                    getattr(attn_metadata, "num_actual_tokens", key.shape[0]),
                    key.shape[0],
                )
                if num_actual_tokens > 0:
                    slot_mapping = attn_metadata.slot_mapping[:num_actual_tokens]
                    key_to_write = key[:num_actual_tokens]
                    value_to_write = value[:num_actual_tokens]
                    req_ids = getattr(attn_metadata, "req_ids", None)
                    num_decode = min(attn_metadata.num_decode_tokens, num_actual_tokens)
                    num_decodes = attn_metadata.num_decodes

                    if num_decode > 0:
                        decode_req_ids = (
                            None if req_ids is None else req_ids[:num_decodes]
                        )
                        self._write_kivi_cache(
                            key_to_write[:num_decode],
                            value_to_write[:num_decode],
                            slot_mapping[:num_decode],
                            decode_req_ids,
                            list(range(1, num_decodes + 1)),
                        )

                    if num_decode < num_actual_tokens:
                        prefill_req_ids = (
                            None if req_ids is None else req_ids[num_decodes:]
                        )
                        prefill_seq_qlen = [
                            int(attn_metadata.actual_seq_lengths_q[i]) - num_decode
                            for i in range(
                                num_decodes,
                                len(attn_metadata.actual_seq_lengths_q),
                            )
                        ]
                        self._write_kivi_prefill_cache(
                            key_to_write[num_decode:num_actual_tokens],
                            value_to_write[num_decode:num_actual_tokens],
                            slot_mapping[num_decode:num_actual_tokens],
                            prefill_req_ids,
                            prefill_seq_qlen,
                        )
            self._sync_kivi_residual_windows(
                attn_metadata.block_tables,
                seq_lens,
                getattr(attn_metadata, "req_ids", None),
                finished_req_ids,
            )
            return self._forward_kivi_chunked_prefill(
                query=query,
                key=key,
                value=value,
                attn_metadata=attn_metadata,
                output=output,
            )

        if key is not None and value is not None:
            num_actual_tokens = min(
                getattr(attn_metadata, "num_actual_tokens", key.shape[0]),
                key.shape[0],
            )
            if num_actual_tokens > 0:
                self._write_kivi_cache(
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    attn_metadata.slot_mapping[:num_actual_tokens],
                    getattr(attn_metadata, "req_ids", None),
                    attn_metadata.actual_seq_lengths_q,
                )
        self._sync_kivi_residual_windows(
            attn_metadata.block_tables,
            seq_lens,
            getattr(attn_metadata, "req_ids", None),
            finished_req_ids,
        )
        dense_key, dense_value = self._gather_dequant_kivi_paged_cache(
            attn_metadata.block_tables,
            seq_lens,
            query.dtype,
            getattr(attn_metadata, "req_ids", None),
        )

        return self._forward_kivi_dense_attention(
            query=query,
            key=dense_key,
            value=dense_value,
            seq_lens=seq_lens,
            actual_seq_qlen=actual_seq_qlen,
            causal=attn_metadata.causal,
            output=output,
        )

    def _forward_kivi_paged_decode_attention(
        self,
        query: Any,
        block_table: Any,
        seq_lens: list[int],
        req_ids: list[str] | None,
        output: Any,
    ) -> Any:
        torch_npu = import_torch_npu("_forward_kivi_paged_decode_attention")
        batch_size = len(seq_lens)

        dense_key, dense_value = self._gather_dequant_kivi_paged_cache(
            block_table, seq_lens, query.dtype, req_ids
        )
        actual_seq_lengths_q = list(range(1, batch_size + 1))
        actual_seq_lengths_kv = (
            torch.tensor(seq_lens, dtype=torch.int32, device="cpu")
            .cumsum(dim=0)
            .tolist()
        )

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query[:batch_size],
            key=dense_key,
            value=dense_value,
            block_table=None,
            input_layout="TND",
            sparse_mode=0,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
        )
        output[:batch_size] = attn_output.view(
            batch_size, self.num_heads, self.head_size
        )
        return output

    def _forward_kivi_prefill_fia(
        self,
        query: Any,
        key: Any,
        value: Any,
        attn_metadata: Any,
        output: Any,
    ) -> Any:
        torch_npu = import_torch_npu("_forward_kivi_prefill_fia")
        num_tokens = int(attn_metadata.actual_seq_lengths_q[-1])

        query = query[:num_tokens]
        key = key[:num_tokens]
        value = value[:num_tokens]
        sparse_mode = 3 if attn_metadata.causal else 0

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask if attn_metadata.causal else None,
            block_table=None,
            input_layout="TND",
            block_size=128,
            actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
            actual_seq_lengths_kv=attn_metadata.actual_seq_lengths_q,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=sparse_mode,
        )
        output[:num_tokens] = attn_output.view(
            num_tokens, self.num_heads, self.head_size
        )
        return output

    def _forward_kivi_chunked_prefill(
        self,
        query: Any,
        key: Any,
        value: Any,
        attn_metadata: Any,
        output: Any,
    ) -> Any:
        torch_npu = import_torch_npu("_forward_kivi_chunked_prefill")
        num_decode = attn_metadata.num_decode_tokens
        num_decodes = attn_metadata.num_decodes
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])
        block_size = self._get_kivi_block_size()

        if num_decode > 0:
            self._forward_kivi_paged_decode_attention(
                query[:num_decode],
                attn_metadata.block_tables[:num_decodes],
                attn_metadata.seq_lens_list[:num_decodes],
                None
                if getattr(attn_metadata, "req_ids", None) is None
                else attn_metadata.req_ids[:num_decodes],
                output,
            )

        if attn_metadata.num_prefills <= 0:
            return output

        prefill_q = query[num_decode:num_tokens]
        prefill_seq_qlen = [
            actual_seq_qlen[i] - num_decode
            for i in range(num_decodes, len(actual_seq_qlen))
        ]

        if key is None or value is None:
            raise RuntimeError(
                "KIVI ChunkedPrefill requires dense key/value for all-new "
                "prefill prompts."
            )

        if not self._is_kivi_chunked_prefill_all_new(attn_metadata):
            raise RuntimeError(
                "KIVI ChunkedPrefill prefill-with-history is disabled. Only "
                "all-new prompt prefills are supported."
            )

        prefill_k = key[num_decode:num_tokens]
        prefill_v = value[num_decode:num_tokens]
        prefill_seq_kvlen = prefill_seq_qlen

        sparse_mode = 3 if attn_metadata.causal else 0
        attn_out, _ = torch_npu.npu_fused_infer_attention_score(
            query=prefill_q,
            key=prefill_k,
            value=prefill_v,
            atten_mask=attn_metadata.attn_mask if attn_metadata.causal else None,
            block_table=None,
            input_layout="TND",
            sparse_mode=sparse_mode,
            block_size=block_size,
            actual_seq_lengths=prefill_seq_qlen,
            actual_seq_lengths_kv=prefill_seq_kvlen,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
        )
        n_prefill = num_tokens - num_decode
        output[num_decode:num_tokens] = attn_out.view(
            n_prefill, self.num_heads, self.head_size
        )[:n_prefill]
        return output

    def _forward_kivi_dense_attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        seq_lens: list[int],
        actual_seq_qlen: Any,
        causal: bool,
        output: Any,
    ) -> Any:
        """纯 torch 稠密注意力兜底路径（逐请求 softmax，无需 NPU 算子）。"""
        prev_q_end = 0
        kv_start = 0
        outputs = []

        if isinstance(actual_seq_qlen, torch.Tensor):
            actual_seq_qlen = actual_seq_qlen.tolist()

        for req_idx, kv_len in enumerate(seq_lens):
            q_end = int(actual_seq_qlen[req_idx])
            q_seq = query[prev_q_end:q_end]
            k_seq = key[kv_start : kv_start + int(kv_len)]
            v_seq = value[kv_start : kv_start + int(kv_len)]

            q_len = q_seq.shape[0]
            if q_len > 0:
                k_seq = self._repeat_kv(k_seq)
                v_seq = self._repeat_kv(v_seq)

                q_states = q_seq.transpose(0, 1)
                attn = torch.matmul(q_states, k_seq.permute(1, 2, 0)) * self.scale

                if causal:
                    attn = attn + self._build_kivi_causal_mask(
                        q_len=q_len,
                        kv_seq_len=int(kv_len),
                        dtype=attn.dtype,
                        device=attn.device,
                    )

                attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
                out = torch.matmul(attn, v_seq.transpose(0, 1))
                outputs.append(out.transpose(0, 1).contiguous())

            prev_q_end = q_end
            kv_start += int(kv_len)

        if outputs:
            attn_output = torch.cat(outputs, dim=0)
            output[: attn_output.shape[0]] = attn_output

        return output


__all__ = ["AscendKiviInt4AttentionBackendMixin"]
