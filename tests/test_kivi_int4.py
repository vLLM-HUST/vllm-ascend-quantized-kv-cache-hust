"""KIVI INT4 solution: semantics math and residual-window state machine.

The triton-ascend kernels themselves require an Ascend NPU; here the
mixin's write paths are exercised against stub kernels, and the pure
semantics run on CPU tensors.
"""

import pytest
import torch

from vllm_ascend_quantized_kv_cache.core.spec import SolutionConfig
from vllm_ascend_quantized_kv_cache.solutions.kivi_int4 import (
    attention_mixin as am,
)
from vllm_ascend_quantized_kv_cache.solutions.kivi_int4.semantics import (
    KiviInt4Semantics,
    validate_kivi_config,
)

HEAD_SIZE = 32
NUM_KV_HEADS = 2
GROUP = 8
BLOCK = 16


# ---------------------------------------------------------------------------
# config invariants
# ---------------------------------------------------------------------------


def test_validate_config_accepts_aligned_geometry() -> None:
    cfg = SolutionConfig(
        head_size=HEAD_SIZE,
        group_size=GROUP,
        residual_length=2 * GROUP,
        block_size=BLOCK,
    )
    validate_kivi_config(cfg)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"group_size": 7}, "divisible by 8"),
        ({"residual_length": 4, "group_size": 8}, "residual_length"),
        ({"head_size": 12}, "head_size"),
        ({"head_size": 8, "group_size": 16}, "head_size"),
        ({"block_size": 8, "group_size": 16}, "block_size"),
    ],
)
def test_validate_config_rejects_bad_geometry(overrides: dict, match: str) -> None:
    kwargs = dict(
        head_size=HEAD_SIZE, group_size=GROUP, residual_length=16, block_size=BLOCK
    )
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        validate_kivi_config(SolutionConfig(**kwargs))


# ---------------------------------------------------------------------------
# packing math
# ---------------------------------------------------------------------------


def test_pack_unpack_int4_roundtrip() -> None:
    torch.manual_seed(0)
    quant = torch.randint(0, 16, (3, NUM_KV_HEADS, 4, 8), dtype=torch.int32)
    packed = KiviInt4Semantics.pack_int4(quant)
    unpacked = KiviInt4Semantics.unpack_int4(packed)
    assert torch.equal(unpacked, quant.to(torch.float32))


def test_fake_quant_error_bounded_by_group_range() -> None:
    torch.manual_seed(1)
    key = torch.randn(32, NUM_KV_HEADS, HEAD_SIZE)
    sem = KiviInt4Semantics(
        SolutionConfig(
            head_size=HEAD_SIZE, group_size=GROUP, residual_length=16, block_size=BLOCK
        )
    )
    out = sem.fake_quant_key(key)
    assert out.shape == key.shape
    assert torch.max(torch.abs(out - key)) <= key.abs().max() / 7 + 1e-5

    value = torch.randn(4, NUM_KV_HEADS, HEAD_SIZE)
    vout = sem.fake_quant_value(value)
    assert vout.shape == value.shape


def test_dequant_key_blocks_roundtrip_shape() -> None:
    torch.manual_seed(2)
    batch, blocks, block_size = 1, 3, BLOCK
    sem = KiviInt4Semantics(
        SolutionConfig(
            head_size=HEAD_SIZE, group_size=GROUP, residual_length=16, block_size=BLOCK
        )
    )
    # legacy cache layout: [B, blocks, H, D, block/G] scales, quantised words
    # [B, blocks, H, D, block/8]
    k_quant = torch.randint(
        0,
        256,
        (batch, blocks, NUM_KV_HEADS, HEAD_SIZE, block_size // 8),
        dtype=torch.int32,
    )
    k_scale = (
        torch.rand(batch, blocks, NUM_KV_HEADS, HEAD_SIZE, block_size // GROUP) + 0.5
    )
    k_mn = torch.full(
        (batch, blocks, NUM_KV_HEADS, HEAD_SIZE, block_size // GROUP), -1.0
    )
    out = sem.dequant_key_blocks(k_quant, k_scale, k_mn, torch.float32)
    assert out.shape == (batch, blocks, block_size, NUM_KV_HEADS, HEAD_SIZE)


def test_is_aligned_key_window() -> None:
    sem = KiviInt4Semantics(
        SolutionConfig(
            head_size=HEAD_SIZE, group_size=GROUP, residual_length=16, block_size=BLOCK
        )
    )
    block_size = BLOCK
    good = list(range(block_size, block_size + GROUP)) + list(
        range(2 * block_size, 2 * block_size + GROUP)
    )
    assert sem.is_aligned_key_window(good, block_size)
    assert not sem.is_aligned_key_window([], block_size)
    assert not sem.is_aligned_key_window(list(range(GROUP - 1)), block_size)
    # misaligned group start (offset 1 inside its block)
    assert not sem.is_aligned_key_window(list(range(1, 1 + GROUP)), block_size)
    # group spanning a block boundary
    crossing = list(range(block_size - 4, block_size + 4))
    assert not sem.is_aligned_key_window(crossing, block_size)


def test_build_causal_mask_shape_and_values() -> None:
    sem = KiviInt4Semantics(
        SolutionConfig(
            head_size=HEAD_SIZE, group_size=GROUP, residual_length=16, block_size=BLOCK
        )
    )
    mask = sem.build_causal_mask(2, 4, torch.float32, torch.device("cpu"))
    assert mask.shape == (1, 2, 4)
    neg_inf = torch.finfo(torch.float32).min
    # q_pos = [2, 3]; a kv position is masked only when kv_pos > q_pos.
    # Row 0 (q=2) masks kv position 3; row 1 (q=3) masks nothing.
    assert mask[0, 0, :3].equal(torch.zeros(3))
    assert mask[0, 0, 3].item() == neg_inf
    assert mask[0, 1].equal(torch.zeros(4))


# ---------------------------------------------------------------------------
# mixin residual-window state machine (stubs instead of NPU kernels)
# ---------------------------------------------------------------------------


class _StubBase:
    num_kv_heads = NUM_KV_HEADS
    head_size = HEAD_SIZE
    num_queries_per_kv = 1
    scale = 0.125
    vllm_config = None

    def __init__(self) -> None:
        self.num_queries_per_kv = 1


def _make_impl(residual_length: int = 2 * GROUP, max_seqs: int = 2):
    class Impl(am.KiviInt4AttentionMixin, _StubBase):
        pass

    impl = Impl()
    # initialise the full attribute set with validation deferred (the stub
    # geometry below is intentionally smaller than the defaults)
    impl._init_kivi_state(None)
    impl.enable_kivi = True
    impl.kivi_group_size = GROUP
    impl.kivi_residual_length = residual_length
    impl.kivi_max_num_seqs = max_seqs
    block_words = BLOCK // 8
    impl.k_quant_cache = torch.zeros(
        4, NUM_KV_HEADS, HEAD_SIZE, block_words, dtype=torch.int32
    )
    impl.k_scale_cache = torch.ones(4, NUM_KV_HEADS, HEAD_SIZE, BLOCK // GROUP)
    impl.k_mn_cache = torch.zeros(4, NUM_KV_HEADS, HEAD_SIZE, BLOCK // GROUP)
    impl.v_quant_cache = torch.zeros(
        4, BLOCK, NUM_KV_HEADS, HEAD_SIZE // 8, dtype=torch.int32
    )
    impl.v_scale_cache = torch.ones(4, BLOCK, NUM_KV_HEADS, HEAD_SIZE // GROUP)
    impl.v_mn_cache = torch.zeros(4, BLOCK, NUM_KV_HEADS, HEAD_SIZE // GROUP)

    # Stub only the kernel launches; the real write validations stay active.
    impl._kivi_key_launches: list = []
    impl._kivi_value_launches: list = []
    impl._launch_kivi_key_pack = lambda key, slots: impl._kivi_key_launches.append(
        (key.clone(), slots.clone())
    )
    impl._launch_kivi_value_pack = lambda val, slots: impl._kivi_value_launches.append(
        (val.clone(), slots.clone())
    )
    return impl


def test_residual_row_lifecycle() -> None:
    impl = _make_impl()
    assert impl._get_kivi_residual_row("req-a", create=False) is None
    row = impl._get_kivi_residual_row("req-a", create=True)
    assert row == 0
    assert impl._get_kivi_residual_row("req-a", create=False) == row
    impl._release_kivi_residual_row("req-a")
    assert impl._get_kivi_residual_row("req-a", create=False) is None
    assert impl.kivi_residual_free_rows == [1, 0]


def test_residual_rows_exhaustion_fails_closed() -> None:
    impl = _make_impl(max_seqs=1)
    impl._get_kivi_residual_row("r0", create=True)
    with pytest.raises(RuntimeError, match="exhausted"):
        impl._get_kivi_residual_row("r1", create=True)


def test_store_entries_and_key_flush_whole_window() -> None:
    impl = _make_impl(residual_length=2 * GROUP)
    # 2*GROUP entries exactly fill the window: no flush yet.
    key = torch.randn(2 * GROUP, NUM_KV_HEADS, HEAD_SIZE)
    slots = torch.arange(2 * GROUP, dtype=torch.long)  # block 1, aligned
    impl._store_kivi_residual_entries(key, slots, req_key="r", is_key=True)
    assert len(impl._kivi_key_launches) == 0
    window_slots, _ = impl._collect_kivi_residual_window("r", is_key=True)
    assert window_slots == slots.tolist()

    # one more entry evicts the whole window as a single aligned flush
    key2 = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
    impl._store_kivi_residual_entries(
        key2, torch.tensor([32]), req_key="r", is_key=True
    )
    assert len(impl._kivi_key_launches) == 1
    flushed, flush_slots = impl._kivi_key_launches[0]
    assert flushed.shape[0] == 2 * GROUP
    assert torch.equal(flush_slots, slots)
    window_slots, tensors = impl._collect_kivi_residual_window("r", is_key=True)
    assert window_slots == [32] and tensors is not None


def test_value_flush_is_slot_at_a_time() -> None:
    impl = _make_impl(residual_length=2 * GROUP)
    value = torch.randn(2 * GROUP + 1, NUM_KV_HEADS, HEAD_SIZE)
    slots = torch.arange(2 * GROUP + 1, dtype=torch.long)
    impl._store_kivi_residual_entries(value, slots, req_key="r", is_key=False)
    # the 17th store evicts exactly one (the oldest) value slot
    assert len(impl._kivi_value_launches) == 1
    flushed, flush_slots = impl._kivi_value_launches[0]
    assert flushed.shape[0] == 1
    assert flush_slots.tolist() == [0]
    window_slots, _ = impl._collect_kivi_residual_window("r", is_key=False)
    assert window_slots == slots[1:].tolist()


def test_misaligned_key_flush_fails_closed_before_kernel() -> None:
    impl = _make_impl()
    # Real validation path: misaligned slots must be rejected before any
    # triton import/launch happens.
    key = torch.randn(GROUP, NUM_KV_HEADS, HEAD_SIZE)
    slots = torch.arange(1, 1 + GROUP, dtype=torch.long)  # offset 1 -> misaligned
    with pytest.raises(RuntimeError, match="aligned token groups"):
        impl._write_kivi_key_quant_cache(key, slots)
    # partial token count must be rejected too
    with pytest.raises(RuntimeError, match="whole token groups"):
        impl._write_kivi_key_quant_cache(key[:-1], torch.arange(GROUP - 1))


def test_aligned_key_flush_reaches_kernel_hook() -> None:
    impl = _make_impl()
    launches: list = []
    impl._launch_kivi_key_pack = lambda key, slots: launches.append(slots.clone())
    key = torch.randn(GROUP, NUM_KV_HEADS, HEAD_SIZE)
    slots = torch.arange(GROUP, dtype=torch.long)  # block 0 group 0: aligned
    impl._write_kivi_key_quant_cache(key, slots)
    assert len(launches) == 1
    assert torch.equal(launches[0], slots)


def test_release_finished_rows_clears_windows() -> None:
    impl = _make_impl()
    value = torch.randn(2, NUM_KV_HEADS, HEAD_SIZE)
    slots = torch.tensor([0, 1], dtype=torch.long)
    impl._store_kivi_residual_entries(value, slots, req_key="r", is_key=False)
    assert impl._has_kivi_residual_entry(0, req_key="r", is_key=False)
    impl._release_finished_kivi_residual_rows({"r"})
    assert not impl._has_kivi_residual_entry(0, req_key="r", is_key=False)
    assert impl._get_kivi_residual_row("r", create=False) is None


def test_sync_windows_releases_finished_and_keeps_live() -> None:
    impl = _make_impl()
    impl._get_kivi_residual_row("live", create=True)
    impl._get_kivi_residual_row("dead", create=True)
    block_table = torch.tensor([[0], [1]])
    impl._sync_kivi_residual_windows(
        block_table, [1, 1], ["live"], finished_req_ids={"dead"}
    )
    assert impl._get_kivi_residual_row("dead", create=False) is None
    assert impl._get_kivi_residual_row("live", create=False) is not None


def test_bind_kivi_cache_rejects_bad_layouts() -> None:
    impl = _make_impl()
    with pytest.raises(RuntimeError, match="6-tuple"):
        impl._bind_kivi_cache((impl.k_quant_cache,))
    impl._bind_kivi_cache(
        [
            impl.k_quant_cache,
            impl.k_scale_cache,
            impl.k_mn_cache,
            impl.v_quant_cache,
            impl.v_scale_cache,
            impl.v_mn_cache,
        ]
    )
    assert impl.kivi_residual_key_cache is not None


def test_chunked_prefill_with_history_fails_closed() -> None:
    from types import SimpleNamespace

    impl = _make_impl()
    md = SimpleNamespace(
        actual_seq_lengths_q=[8, 16],
        seq_lens_list=[32, 16],
        num_decodes=1,
        num_decode_tokens=8,
        num_prefills=1,
    )
    assert impl._is_kivi_chunked_prefill_all_new(md) is False
