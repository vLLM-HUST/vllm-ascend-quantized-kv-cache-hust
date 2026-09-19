"""KIVI INT4 method: semantics math, residual-window state machine, attention paths.

The triton-ascend kernels require an Ascend NPU, so the mixin's write paths
run against CPU reference packers, and the fused-attention branches are
checked at the ``torch_npu`` operator boundary with a recording stub.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend_quantized_kv_cache.methods.base import MethodConfig
from vllm_ascend_quantized_kv_cache.methods.kivi_int4 import (
    attention_backend as ab,
)
from vllm_ascend_quantized_kv_cache.methods.kivi_int4.geometry import (
    validate_kivi_config,
)
from vllm_ascend_quantized_kv_cache.methods.kivi_int4.semantics import (
    KiviInt4Semantics,
)

HEAD_SIZE = 32
NUM_KV_HEADS = 2
GROUP = 8
BLOCK = 16


# ---------------------------------------------------------------------------
# config invariants
# ---------------------------------------------------------------------------


def test_validate_config_accepts_aligned_geometry() -> None:
    cfg = MethodConfig(
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
        validate_kivi_config(MethodConfig(**kwargs))


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
        MethodConfig(
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
        MethodConfig(
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
        MethodConfig(
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
        MethodConfig(
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


def _make_impl(
    residual_length: int = 2 * GROUP,
    max_seqs: int = 2,
    *,
    head_size: int = HEAD_SIZE,
    num_kv_heads: int = NUM_KV_HEADS,
    group_size: int = GROUP,
    block_size: int = BLOCK,
    num_blocks: int = 4,
):
    class Impl(ab.AscendKiviInt4AttentionBackendMixin, _StubBase):
        pass

    impl = Impl()
    # initialise the full attribute set with validation deferred (the stub
    # geometry below is intentionally smaller than the defaults)
    impl._init_kivi_state(None)
    impl.enable_kivi = True
    impl.head_size = head_size
    impl.num_kv_heads = num_kv_heads
    impl.kivi_group_size = group_size
    impl.kivi_residual_length = residual_length
    impl.kivi_max_num_seqs = max_seqs
    impl.k_quant_cache = torch.zeros(
        num_blocks, num_kv_heads, head_size, block_size // 8, dtype=torch.int32
    )
    impl.k_scale_cache = torch.ones(
        num_blocks, num_kv_heads, head_size, block_size // group_size
    )
    impl.k_mn_cache = torch.zeros(
        num_blocks, num_kv_heads, head_size, block_size // group_size
    )
    impl.v_quant_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, head_size // 8, dtype=torch.int32
    )
    impl.v_scale_cache = torch.ones(
        num_blocks, block_size, num_kv_heads, head_size // group_size
    )
    impl.v_mn_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, head_size // group_size
    )

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
    impl = _make_impl()
    md = SimpleNamespace(
        actual_seq_lengths_q=[8, 16],
        seq_lens_list=[32, 16],
        num_decodes=1,
        num_decode_tokens=8,
        num_prefills=1,
    )
    assert impl._is_kivi_chunked_prefill_all_new(md) is False


# ---------------------------------------------------------------------------
# torch-op gather path (routed by kivi_dequant_gather_cache) — CPU roundtrip
# ---------------------------------------------------------------------------


def _build_packed_key_cache(
    key, *, num_blocks, block_size, group_size, num_kv_heads, head_size
):
    """Emulate the (NPU-only) pack kernel layout on CPU via semantics math."""
    torch.manual_seed(3)
    kq = torch.zeros(
        num_blocks, num_kv_heads, head_size, block_size // 8, dtype=torch.int32
    )
    ks = torch.zeros(num_blocks, num_kv_heads, head_size, block_size // group_size)
    km = torch.zeros_like(ks)
    for b in range(num_blocks):
        block_key = key[b * block_size : (b + 1) * block_size]  # [B, KVH, H]
        grouped = block_key.view(
            block_size // group_size, group_size, num_kv_heads, head_size
        )
        mn = grouped.amin(dim=1, keepdim=True)
        mx = grouped.amax(dim=1, keepdim=True)
        scale = KiviInt4Semantics.group_scale(mn, mx)
        quant = KiviInt4Semantics.quantize_group(grouped, mn, scale)
        # [groups, group, KVH, H] -> [KVH, H, block/8, 8]
        words = quant.permute(2, 3, 0, 1).reshape(
            num_kv_heads, head_size, block_size // 8, 8
        )
        kq[b] = KiviInt4Semantics.pack_int4(words)
        ks[b] = scale[:, 0].permute(1, 2, 0)  # [KVH, H, groups]
        km[b] = mn[:, 0].permute(1, 2, 0)
    return kq, ks, km


def _build_packed_value_cache(
    value, *, num_blocks, block_size, group_size, num_kv_heads, head_size
):
    """值的 head 维组量化打包成 ``[blocks, block, KVH, head/8]`` int32 word。"""
    total = num_blocks * block_size
    grouped = value.view(total, num_kv_heads, head_size // group_size, group_size)
    vmn = grouped.amin(-1, keepdim=True)
    vscale = KiviInt4Semantics.group_scale(vmn, grouped.amax(-1, keepdim=True))
    vquant = KiviInt4Semantics.quantize_group(grouped, vmn, vscale).view(
        total, num_kv_heads, head_size
    )
    pad = torch.zeros(total, num_kv_heads, (8 - head_size % 8) % 8, dtype=torch.int32)
    vquant = torch.cat([vquant, pad], dim=-1)
    vq_words = KiviInt4Semantics.pack_int4(
        vquant.view(num_blocks, block_size, num_kv_heads, head_size // 8, 8)
    )
    return (
        vq_words,
        vscale.view(num_blocks, block_size, num_kv_heads, head_size // group_size),
        vmn.view(num_blocks, block_size, num_kv_heads, head_size // group_size),
    )


def test_dequant_gather_cache_torch_roundtrip_cpu() -> None:
    from vllm_ascend_quantized_kv_cache.ops.kivi_gather import (
        kivi_dequant_gather_cache,
    )

    torch.manual_seed(4)
    nb, block, group, kvh, head = 3, 16, 8, 2, 32
    total = nb * block
    key = torch.randn(total, kvh, head)
    value = torch.randn(total, kvh, head)
    sem = KiviInt4Semantics(
        MethodConfig(
            head_size=head,
            num_kv_heads=kvh,
            group_size=group,
            residual_length=group,
            block_size=block,
        )
    )
    kq, ks, km = _build_packed_key_cache(
        key,
        num_blocks=nb,
        block_size=block,
        group_size=group,
        num_kv_heads=kvh,
        head_size=head,
    )
    # values: per-token per-head-dim-group quantization
    vq_words, v_scale, v_mn = _build_packed_value_cache(
        value,
        num_blocks=nb,
        block_size=block,
        group_size=group,
        num_kv_heads=kvh,
        head_size=head,
    )

    block_table = torch.arange(nb).view(1, nb)
    seq_lens = torch.tensor([total])
    k_out, v_out = kivi_dequant_gather_cache(
        kq,
        ks,
        km,
        vq_words,
        v_scale,
        v_mn,
        block_table,
        seq_lens,
        torch.float32,
        group,
    )
    ref_k = sem.fake_quant_key(key)
    ref_v = sem.fake_quant_value(value)
    assert torch.allclose(k_out, ref_k, atol=1e-5)
    assert torch.allclose(v_out, ref_v, atol=1e-5)

    # partial sequence: only live tokens come back
    k_out2, v_out2 = kivi_dequant_gather_cache(
        kq,
        ks,
        km,
        vq_words,
        v_scale,
        v_mn,
        block_table,
        torch.tensor([total - 3]),
        torch.float32,
        group,
    )
    assert k_out2.shape[0] == total - 3
    assert torch.equal(k_out2, k_out[: total - 3])


def test_semantics_block_dequant_matches_gather_op() -> None:
    """The CPU reference and the routed gather op encode the same dequant math.

    They are two implementations, so drift has to fail loudly: the semantics
    side is what the NPU kernels are bit-checked against on 910B2.
    """
    from vllm_ascend_quantized_kv_cache.ops.kivi_gather import (
        kivi_dequant_gather_cache,
    )

    torch.manual_seed(11)
    nb, block, group, kvh, head = 2, 16, 8, 2, 32
    total = nb * block
    key = torch.randn(total, kvh, head)
    value = torch.randn(total, kvh, head)
    sem = KiviInt4Semantics(
        MethodConfig(
            head_size=head,
            num_kv_heads=kvh,
            group_size=group,
            residual_length=group,
            block_size=block,
        )
    )
    kq, ks, km = _build_packed_key_cache(
        key,
        num_blocks=nb,
        block_size=block,
        group_size=group,
        num_kv_heads=kvh,
        head_size=head,
    )
    vq, vs, vm = _build_packed_value_cache(
        value,
        num_blocks=nb,
        block_size=block,
        group_size=group,
        num_kv_heads=kvh,
        head_size=head,
    )

    gathered_k, gathered_v = kivi_dequant_gather_cache(
        kq,
        ks,
        km,
        vq,
        vs,
        vm,
        torch.arange(nb).view(1, nb),
        torch.tensor([total]),
        torch.float32,
        group,
    )
    ref_k = sem.dequant_key_blocks(
        kq.unsqueeze(0), ks.unsqueeze(0), km.unsqueeze(0), torch.float32
    )
    ref_v = sem.dequant_value_blocks(
        vq.unsqueeze(0), vs.unsqueeze(0), vm.unsqueeze(0), torch.float32
    )
    assert torch.equal(ref_k.reshape(total, kvh, head), gathered_k)
    assert torch.equal(ref_v.reshape(total, kvh, head), gathered_v)


# ---------------------------------------------------------------------------
# forward(): CPU reference packers stand in for the triton launches, so the
# whole write -> flush -> gather -> attention pipeline runs without an NPU.
# ---------------------------------------------------------------------------


def _install_cpu_packers(impl) -> None:
    """Slot-driven int4 packers writing the exact layout the kernels define."""
    block = impl.k_quant_cache.shape[-1] * 8

    def pack_key(key, slots):
        group = impl.kivi_group_size
        for gi in range(key.shape[0] // group):
            grp = key[gi * group : (gi + 1) * group].to(torch.float32)
            blk, off = int(slots[gi * group]) // block, int(slots[gi * group]) % block
            mn = grp.amin(dim=0, keepdim=True)
            scale = KiviInt4Semantics.group_scale(mn, grp.amax(dim=0, keepdim=True))
            quant = KiviInt4Semantics.quantize_group(grp, mn, scale)
            words = quant.permute(1, 2, 0).reshape(
                impl.num_kv_heads, impl.head_size, group // 8, 8
            )
            first_word = off // 8
            impl.k_quant_cache[blk, :, :, first_word : first_word + group // 8] = (
                KiviInt4Semantics.pack_int4(words)
            )
            impl.k_scale_cache[blk, :, :, off // group] = scale[0]
            impl.k_mn_cache[blk, :, :, off // group] = mn[0]

    def pack_value(value, slots):
        group = impl.kivi_group_size
        heads = impl.head_size // group
        for i in range(value.shape[0]):
            v = value[i].to(torch.float32)
            blk, off = int(slots[i]) // block, int(slots[i]) % block
            grouped = v.reshape(impl.num_kv_heads, heads, group)
            mn = grouped.amin(dim=-1, keepdim=True)
            scale = KiviInt4Semantics.group_scale(
                mn, grouped.amax(dim=-1, keepdim=True)
            )
            quant = KiviInt4Semantics.quantize_group(grouped, mn, scale)
            flat = quant.reshape(impl.num_kv_heads, impl.head_size)
            impl.v_quant_cache[blk, off] = KiviInt4Semantics.pack_int4(
                flat.reshape(impl.num_kv_heads, impl.head_size // 8, 8)
            )
            impl.v_scale_cache[blk, off] = scale[..., 0]
            impl.v_mn_cache[blk, off] = mn[..., 0]

    impl._launch_kivi_key_pack = pack_key
    impl._launch_kivi_value_pack = pack_value


def _six_tuple(impl):
    return (
        impl.k_quant_cache,
        impl.k_scale_cache,
        impl.k_mn_cache,
        impl.v_quant_cache,
        impl.v_scale_cache,
        impl.v_mn_cache,
    )


def _metadata(tokens: int, **overrides):
    class _State:
        name = "PrefillCacheHit"

    md = SimpleNamespace(
        attn_state=_State(),
        actual_seq_lengths_q=[tokens],
        seq_lens_list=[tokens],
        slot_mapping=torch.arange(tokens, dtype=torch.long),
        block_tables=torch.tensor([[0, 1]], dtype=torch.long),
        req_ids=["r"],
        num_actual_tokens=tokens,
        causal=True,
        finished_req_ids=None,
    )
    for key, value in overrides.items():
        setattr(md, key, value)
    return md


def test_forward_guards_fail_closed() -> None:
    impl = _make_impl()
    q = k = v = torch.randn(1, NUM_KV_HEADS, HEAD_SIZE)
    out = torch.zeros(1, NUM_KV_HEADS, HEAD_SIZE)

    assert impl.forward(None, q, k, v, _six_tuple(impl), None, out) is out
    assert torch.equal(out, torch.zeros_like(out))

    with pytest.raises(NotImplementedError, match="fused output quantization"):
        impl.forward(
            None, q, k, v, _six_tuple(impl), _metadata(1), out, output_scale=out.clone()
        )
    with pytest.raises(RuntimeError, match="6-tuple"):
        impl.forward(None, q, k, v, (impl.k_quant_cache,), _metadata(1), out)

    impl.enable_kivi = False
    with pytest.raises(RuntimeError, match="kivi_int4 cache dtype"):
        impl.forward(None, q, k, v, _six_tuple(impl), _metadata(1), out)


def test_forward_history_plus_residual_matches_quantized_reference() -> None:
    """Keys flush whole groups into int4 history; the residual tail stays exact."""
    residual = 2 * GROUP
    tokens = residual + 1
    torch.manual_seed(7)
    impl = _make_impl(residual_length=residual)
    _install_cpu_packers(impl)
    sem = KiviInt4Semantics(
        MethodConfig(
            head_size=HEAD_SIZE,
            num_kv_heads=NUM_KV_HEADS,
            group_size=GROUP,
            residual_length=residual,
            block_size=BLOCK,
        )
    )

    query = torch.randn(tokens, NUM_KV_HEADS, HEAD_SIZE)
    key = torch.randn(tokens, NUM_KV_HEADS, HEAD_SIZE)
    value = torch.randn(tokens, NUM_KV_HEADS, HEAD_SIZE)
    output = torch.zeros(tokens, NUM_KV_HEADS, HEAD_SIZE)

    got = impl.forward(
        None, query, key, value, _six_tuple(impl), _metadata(tokens), output
    )

    # history: the first `residual` keys were flushed as two aligned groups;
    # values only evict one oldest slot at a time.
    expected_k = torch.cat([sem.fake_quant_key(key[:residual]), key[residual:]])
    expected_v = torch.cat([sem.fake_quant_value(value[:1]), value[1:]])
    gathered_k, gathered_v = impl._gather_dequant_kivi_paged_cache(
        torch.tensor([[0, 1]], dtype=torch.long), [tokens], torch.float32, ["r"]
    )
    assert torch.allclose(gathered_k, expected_k, atol=1e-5)
    assert torch.allclose(gathered_v, expected_v, atol=1e-5)

    scores = torch.einsum("qhd,khd->hqk", query, expected_k) * impl.scale
    mask = torch.triu(torch.full((tokens, tokens), float("-inf")), diagonal=1)
    reference = torch.einsum(
        "hqk,khd->qhd", torch.softmax(scores + mask, dim=-1), expected_v
    )
    assert torch.allclose(got, reference, atol=1e-4)

    # Non-vacuity: the same attention over unquantized KV must differ, or the
    # int4 history would not be in the compute loop at all.
    exact_scores = torch.einsum("qhd,khd->hqk", query, key) * impl.scale
    exact = torch.einsum(
        "hqk,khd->qhd", torch.softmax(exact_scores + mask, dim=-1), value
    )
    assert (got - exact).abs().max() > 1e-3
    assert impl.k_quant_cache.any() and impl.v_quant_cache.any()


@pytest.mark.parametrize(
    "break_it, match",
    [
        (lambda c: c.__setitem__("kq", c["kq"].transpose(1, 2)), "must be contiguous"),
        (
            lambda c: c.__setitem__("ks", torch.zeros(2, 2, 32, 3)),
            "k_scale_cache layout",
        ),
        (
            lambda c: c.__setitem__("vq", torch.zeros(2, 8, 2, 4, dtype=torch.int32)),
            "layout mismatch",
        ),
        (
            lambda c: c.__setitem__("block_table", c["block_table"].float()),
            "block_table must be int32/int64",
        ),
        (lambda c: c.__setitem__("seq_lens", torch.tensor([16, 16])), "must match"),
    ],
)
def test_gather_op_rejects_broken_cache_layout(break_it, match) -> None:
    """Every pack-kernel layout promise is enforced before any gather compute."""
    from vllm_ascend_quantized_kv_cache.ops.kivi_gather import (
        kivi_dequant_gather_cache,
    )

    nb, block, group, kvh, head = 2, 16, 8, 2, 32
    total = nb * block
    key = torch.randn(total, kvh, head)
    value = torch.randn(total, kvh, head)
    kq, ks, km = _build_packed_key_cache(
        key,
        num_blocks=nb,
        block_size=block,
        group_size=group,
        num_kv_heads=kvh,
        head_size=head,
    )
    vq, vs, vm = _build_packed_value_cache(
        value,
        num_blocks=nb,
        block_size=block,
        group_size=group,
        num_kv_heads=kvh,
        head_size=head,
    )
    caches = {
        "kq": kq,
        "ks": ks,
        "km": km,
        "vq": vq,
        "vs": vs,
        "vm": vm,
        "block_table": torch.arange(nb).view(1, nb),
        "seq_lens": torch.tensor([total]),
    }
    break_it(caches)
    with pytest.raises(RuntimeError, match=match):
        kivi_dequant_gather_cache(
            caches["kq"],
            caches["ks"],
            caches["km"],
            caches["vq"],
            caches["vs"],
            caches["vm"],
            caches["block_table"],
            caches["seq_lens"],
            torch.float32,
            group,
        )


def test_gather_op_requires_eight_lane_groups() -> None:
    from vllm_ascend_quantized_kv_cache.ops.kivi_gather import (
        kivi_dequant_gather_cache,
    )

    nb, block, kvh, head = 2, 16, 2, 32
    total = nb * block
    key = torch.randn(total, kvh, head)
    value = torch.randn(total, kvh, head)
    kq, ks, km = _build_packed_key_cache(
        key,
        num_blocks=nb,
        block_size=block,
        group_size=8,
        num_kv_heads=kvh,
        head_size=head,
    )
    vq, vs, vm = _build_packed_value_cache(
        value,
        num_blocks=nb,
        block_size=block,
        group_size=8,
        num_kv_heads=kvh,
        head_size=head,
    )
    with pytest.raises(RuntimeError, match="divisible by 8"):
        kivi_dequant_gather_cache(
            kq,
            ks,
            km,
            vq,
            vs,
            vm,
            torch.arange(nb).view(1, nb),
            torch.tensor([total]),
            torch.float32,
            4,
        )


# ---------------------------------------------------------------------------
# NPU operator boundary: the fused-attention branches are checked against a
# recording torch_npu stub (port fidelity), which is the same standard the
# pack/gather kernels are held to before a 910B2 run.
# ---------------------------------------------------------------------------


def _fake_torch_npu(monkeypatch):
    """Install a recording ``torch_npu`` and return the captured call list."""
    import sys
    import types

    calls: list[dict] = []

    def npu_fused_infer_attention_score(**kwargs):
        query = kwargs["query"]
        heads = kwargs["num_heads"]
        head_size = kwargs["key"].shape[-1]
        # a per-call sentinel, so a wrong output slice is detectable
        out = torch.full(
            (query.shape[0], heads, head_size), float(len(calls) + 1), dtype=query.dtype
        )
        calls.append(kwargs)
        return out, None

    module = types.ModuleType("torch_npu")
    module.npu_fused_infer_attention_score = npu_fused_infer_attention_score
    monkeypatch.setitem(sys.modules, "torch_npu", module)
    return calls


def _attention_impl(residual_length: int = 2 * GROUP):
    impl = _make_impl(residual_length=residual_length)
    _install_cpu_packers(impl)
    impl.num_heads = NUM_KV_HEADS
    return impl


def test_prefill_nocache_quantizes_then_calls_dense_tnd_fia(monkeypatch) -> None:
    tokens = 2 * GROUP + 1
    impl = _attention_impl()
    calls = _fake_torch_npu(monkeypatch)

    query = torch.randn(tokens, NUM_KV_HEADS, HEAD_SIZE)
    key = torch.randn(tokens, NUM_KV_HEADS, HEAD_SIZE)
    value = torch.randn(tokens, NUM_KV_HEADS, HEAD_SIZE)
    out = torch.zeros(tokens, NUM_KV_HEADS, HEAD_SIZE)
    mask = torch.zeros(1, tokens, tokens)

    md = _metadata(
        tokens,
        attn_state=SimpleNamespace(name="PrefillNoCache"),
        attn_mask=mask,
    )

    impl.forward(
        None,
        query,
        key,
        value,
        _six_tuple(impl),
        md,
        out,
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["input_layout"] == "TND"
    # causal prefill uses the additive mask, and the operator sees the raw
    # prompt K/V while the packed copy lands in history + residual
    assert call["sparse_mode"] == 3
    assert call["atten_mask"] is mask
    assert torch.equal(call["key"], key)
    assert call["block_table"] is None
    assert call["actual_seq_lengths"] == [tokens]
    assert call["num_heads"] == NUM_KV_HEADS
    assert call["num_key_value_heads"] == NUM_KV_HEADS
    assert torch.equal(out, torch.ones_like(out))
    assert impl.k_scale_cache.abs().sum() > 0
    assert impl._get_kivi_residual_row("r", create=False) is not None


def test_decode_only_gathers_history_and_calls_paged_fia(monkeypatch) -> None:
    impl = _attention_impl()
    history = 2 * GROUP
    # two requests, each with a fully flushed 16-token history
    prefill_key = torch.randn(2 * history, NUM_KV_HEADS, HEAD_SIZE)
    prefill_value = torch.randn(2 * history, NUM_KV_HEADS, HEAD_SIZE)
    impl._write_kivi_cache(
        prefill_key,
        prefill_value,
        torch.arange(2 * history, dtype=torch.long),
        ["a", "b"],
        [history, 2 * history],
    )

    calls = _fake_torch_npu(monkeypatch)
    # one decode token per request; request a continues into block 2
    query = torch.randn(2, NUM_KV_HEADS, HEAD_SIZE)
    key = torch.randn(2, NUM_KV_HEADS, HEAD_SIZE)
    value = torch.randn(2, NUM_KV_HEADS, HEAD_SIZE)
    out = torch.zeros(2, NUM_KV_HEADS, HEAD_SIZE)
    seq_len = history + 1
    md = _metadata(
        2,
        attn_state=SimpleNamespace(name="DecodeOnly"),
        seq_lens_list=[seq_len, seq_len],
        slot_mapping=torch.tensor([2 * history, 2 * history + 1], dtype=torch.long),
        block_tables=torch.tensor([[0, 2], [1, 2]], dtype=torch.long),
        req_ids=["a", "b"],
    )

    impl.forward(None, query, key, value, _six_tuple(impl), md, out)

    assert len(calls) == 1
    call = calls[0]
    assert call["input_layout"] == "TND"
    assert call["sparse_mode"] == 0
    assert call["block_table"] is None
    assert "atten_mask" not in call
    # one query token per request; kv lengths are cumulative across the batch
    assert call["actual_seq_lengths"] == [1, 2]
    assert call["actual_seq_lengths_kv"] == [seq_len, 2 * seq_len]
    assert call["query"].shape[0] == 2
    assert call["key"].shape == (2 * seq_len, NUM_KV_HEADS, HEAD_SIZE)
    assert call["num_key_value_heads"] == NUM_KV_HEADS
    assert torch.equal(out, torch.ones_like(out))
    # the overflowed windows left for int4 history; this step's K/V stayed exact
    assert impl._has_kivi_residual_entry(2 * history, req_key="a", is_key=True)
    assert not impl._has_kivi_residual_entry(0, req_key="a", is_key=True)
    assert impl.k_scale_cache.abs().sum() > 0


def test_chunked_prefill_dispatches_decode_rows_then_prompt_rows(monkeypatch) -> None:
    impl = _attention_impl()
    calls = _fake_torch_npu(monkeypatch)

    num_decode, prompt = 1, 4
    tokens = num_decode + prompt
    query = torch.randn(tokens, NUM_KV_HEADS, HEAD_SIZE)
    key = torch.randn(tokens, NUM_KV_HEADS, HEAD_SIZE)
    value = torch.randn(tokens, NUM_KV_HEADS, HEAD_SIZE)
    out = torch.zeros(tokens, NUM_KV_HEADS, HEAD_SIZE)
    md = _metadata(
        tokens,
        attn_state=SimpleNamespace(name="ChunkedPrefill"),
        actual_seq_lengths_q=[num_decode, tokens],
        seq_lens_list=[num_decode, prompt],
        num_decodes=num_decode,
        num_decode_tokens=num_decode,
        num_prefills=1,
        req_ids=["d", "p"],
        block_tables=torch.tensor([[0, 0], [0, 1]], dtype=torch.long),
        attn_mask=torch.zeros(1, prompt, prompt),
    )

    impl.forward(None, query, key, value, _six_tuple(impl), md, out)

    assert len(calls) == 2
    decode_call, prefill_call = calls
    assert decode_call["query"].shape[0] == num_decode
    assert decode_call["actual_seq_lengths"] == [1]
    assert decode_call["actual_seq_lengths_kv"] == [num_decode]
    assert prefill_call["query"] is not None
    assert prefill_call["query"].shape[0] == prompt
    # all-new prompt rows attend to their own dense K/V, not the paged cache
    assert torch.equal(prefill_call["key"], key[num_decode:tokens])
    assert prefill_call["sparse_mode"] == 3
    assert prefill_call["actual_seq_lengths"] == [prompt]
    assert prefill_call["block_size"] == BLOCK
    # decode rows took the first FIA result, prefill rows the second one
    assert torch.equal(out[:num_decode], torch.ones_like(out[:num_decode]))
    assert torch.equal(
        out[num_decode:tokens], torch.full_like(out[num_decode:tokens], 2.0)
    )


def test_reference_arithmetic_matches_pack_kernel_contract() -> None:
    """The CPU reference must round and floor exactly like the triton kernel.

    Both choices were measured against ``ops/triton/kivi_pack.py``: the scale
    floor applies to the scale, and ties round up (``floor(x + 0.5)``), while
    ``torch.round`` is half-to-even.
    """
    kernel_source = (
        Path(__file__).parents[1]
        / "src/vllm_ascend_quantized_kv_cache/ops/triton/kivi_pack.py"
    ).read_text()
    assert "tl.maximum((mx - mn) / 15.0, 1.0e-6)" in kernel_source
    assert "tl.floor((pack_values - mn) / scale + 0.5)" in kernel_source

    mn = torch.zeros(1, 1, 1)
    # a near-constant group: flooring the range first would yield 2e-8 scale
    assert KiviInt4Semantics.group_scale(mn, torch.full_like(mn, 3e-7)).item() == (
        pytest.approx(1e-6, rel=1e-6)
    )
    assert KiviInt4Semantics.group_scale(mn, torch.full_like(mn, 1e-9)).item() == (
        pytest.approx(1e-6, rel=1e-6)
    )

    scale = torch.full((1, 1, 3), 1.0)
    values = torch.full((1, 1, 3), 2.5)
    quantized = KiviInt4Semantics.quantize_group(values, mn, scale)
    assert quantized.tolist() == [[[3, 3, 3]]]
    assert quantized.dtype == torch.int32
    clipped = KiviInt4Semantics.quantize_group(
        torch.full((1, 1, 1), 99.0), torch.zeros(1, 1, 1), torch.ones(1, 1, 1)
    )
    assert clipped.item() == 15


def test_chunked_prefill_with_prompt_history_fails_closed(monkeypatch) -> None:
    impl = _attention_impl()
    _fake_torch_npu(monkeypatch)
    tokens = 5
    md = _metadata(
        tokens,
        attn_state=SimpleNamespace(name="ChunkedPrefill"),
        actual_seq_lengths_q=[1, tokens],
        seq_lens_list=[1, tokens - 1 + 3],
        num_decodes=1,
        num_decode_tokens=1,
        num_prefills=1,
    )
    with pytest.raises(RuntimeError, match="historical KV cache"):
        impl.forward(
            None,
            torch.randn(tokens, NUM_KV_HEADS, HEAD_SIZE),
            torch.randn(tokens, NUM_KV_HEADS, HEAD_SIZE),
            torch.randn(tokens, NUM_KV_HEADS, HEAD_SIZE),
            _six_tuple(impl),
            md,
            torch.zeros(tokens, NUM_KV_HEADS, HEAD_SIZE),
        )


# ---------------------------------------------------------------------------
# byte-region caches: vLLM hands the impl two tensors, so the six KIVI views
# have to be views over those two buffers.
# ---------------------------------------------------------------------------


def _byte_caches(num_blocks: int = 4, block_size: int = BLOCK):
    from vllm_ascend_quantized_kv_cache.methods.kivi_int4.byte_cache import (
        KiviByteCacheLayout,
    )

    layout = KiviByteCacheLayout(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        group_size=GROUP,
    )
    key = torch.zeros(
        num_blocks,
        block_size,
        NUM_KV_HEADS,
        layout.bytes_per_token_head,
        dtype=torch.uint8,
    )
    value = torch.zeros_like(key)
    return key, value, layout


def test_byte_cache_budget_matches_view_sizes() -> None:
    from vllm_ascend_quantized_kv_cache.methods.kivi_int4.byte_cache import (
        KiviByteCacheLayout,
    )

    key, value, layout = _byte_caches()
    # independent arithmetic: int4 data + one fp32 scale and min per group
    assert layout.bytes_per_token_head == HEAD_SIZE // 2 + 8 * HEAD_SIZE // GROUP
    assert key.numel() * key.element_size() == layout.region_bytes
    assert layout.key_bytes == layout.value_bytes == layout.region_bytes
    assert layout.bytes_per_token == 2 * NUM_KV_HEADS * layout.bytes_per_token_head

    # the default production geometry must beat fp16 by ~3.5x, not a vague 4x
    default = KiviByteCacheLayout(
        num_blocks=1, block_size=128, num_kv_heads=8, head_size=128, group_size=128
    )
    assert default.compression_vs_fp16() == pytest.approx(256 / 72, rel=1e-6)
    assert (
        layout.compression_vs_fp16() < 2.0
    )  # head_size 32 / group 8 is overhead-heavy


def test_byte_cache_views_alias_the_host_buffers() -> None:
    from vllm_ascend_quantized_kv_cache.methods.kivi_int4.byte_cache import (
        kivi_caches_from_byte_tensors,
    )

    key, value, layout = _byte_caches()
    k_quant, k_scale, k_mn, v_quant, v_scale, v_mn = kivi_caches_from_byte_tensors(
        key, value, layout
    )
    assert k_quant.shape == (4, NUM_KV_HEADS, HEAD_SIZE, BLOCK // 8)
    assert k_quant.dtype == torch.int32
    assert k_scale.shape == k_mn.shape == (4, NUM_KV_HEADS, HEAD_SIZE, BLOCK // GROUP)
    assert k_scale.dtype == torch.float32
    assert v_quant.shape == (4, BLOCK, NUM_KV_HEADS, HEAD_SIZE // 8)
    assert v_scale.shape == v_mn.shape == (4, BLOCK, NUM_KV_HEADS, HEAD_SIZE // GROUP)

    k_quant[0, 0, 0, 0] = 0x04030201
    assert key.reshape(-1)[:4].tolist() == [1, 2, 3, 4]  # little-endian, same memory
    v_scale[0, 0, 0, 0] = 2.5
    # the value region is [v_quant | v_scale | v_mn]; fp32 2.5 is 00 00 20 40
    scale_offset = v_quant.numel() * v_quant.element_size()
    flat_value = value.reshape(-1)
    assert not flat_value[:scale_offset].any()
    assert flat_value[scale_offset : scale_offset + 4].tolist() == [0, 0, 0x20, 0x40]
    assert key.reshape(-1)[:4].tolist() == [1, 2, 3, 4]  # value side never touches K


@pytest.mark.parametrize(
    "break_it, match",
    [
        (
            lambda kv: (kv[0], kv[1][:1]),
            "equal regions",
        ),
        (
            lambda kv: (kv[0].transpose(0, 1), kv[1]),
            "must be contiguous",
        ),
    ],
)
def test_byte_cache_layout_rejects_bad_buffers(break_it, match) -> None:
    from vllm_ascend_quantized_kv_cache.methods.kivi_int4.byte_cache import (
        kivi_byte_cache_layout,
    )

    key, value, _ = _byte_caches()
    key, value = break_it((key, value))
    with pytest.raises(RuntimeError, match=match):
        kivi_byte_cache_layout(
            key, value, num_kv_heads=NUM_KV_HEADS, head_size=HEAD_SIZE, group_size=GROUP
        )


def test_forward_on_two_byte_buffers_matches_six_tuple_run(monkeypatch) -> None:
    """The host-visible 2-tensor form must behave exactly like the 6-tuple one."""
    from vllm_ascend_quantized_kv_cache.methods.kivi_int4.byte_cache import (
        kivi_caches_from_byte_tensors,
    )

    residual = 2 * GROUP
    tokens = residual + 1
    torch.manual_seed(21)
    query = torch.randn(tokens, NUM_KV_HEADS, HEAD_SIZE)
    key = torch.randn(tokens, NUM_KV_HEADS, HEAD_SIZE)
    value = torch.randn(tokens, NUM_KV_HEADS, HEAD_SIZE)
    md = _metadata(tokens)

    six = _attention_impl(residual_length=residual)
    _install_cpu_packers(six)
    six_out = torch.zeros(tokens, NUM_KV_HEADS, HEAD_SIZE)
    six.forward(None, query, key, value, _six_tuple(six), md, six_out)

    buffers, value_buffers, layout = _byte_caches(num_blocks=six.k_quant_cache.shape[0])
    two = _attention_impl(residual_length=residual)
    _install_cpu_packers(two)
    two_out = torch.zeros(tokens, NUM_KV_HEADS, HEAD_SIZE)
    two.forward(None, query, key, value, (buffers, value_buffers), md, two_out)

    assert torch.equal(two_out, six_out)
    derived = kivi_caches_from_byte_tensors(buffers, value_buffers, layout)
    assert all(
        torch.equal(new, old) for new, old in zip(derived, _six_tuple(two), strict=True)
    )
    assert buffers.any(), "int4 history must have been written into the host buffer"


def test_forward_at_production_geometry_on_byte_buffers() -> None:
    """head 128 / kv 8 / block 128 / group 128 -- the shipped defaults.

    Every other test runs at a small toy geometry; the word math, the
    one-group-per-block flush and the region budget all depend on the real
    sizes, so they are exercised here over the host's two byte buffers.
    """
    from vllm_ascend_quantized_kv_cache.methods.kivi_int4.byte_cache import (
        KiviByteCacheLayout,
        kivi_caches_from_byte_tensors,
    )

    head, kvh, block, group = 128, 8, 128, 128
    residual = 128
    num_blocks = 3
    tokens = residual + 1

    layout = KiviByteCacheLayout(
        num_blocks=num_blocks,
        block_size=block,
        num_kv_heads=kvh,
        head_size=head,
        group_size=group,
    )
    key_buf = torch.zeros(
        num_blocks, block, kvh, layout.bytes_per_token_head, dtype=torch.uint8
    )
    value_buf = torch.zeros_like(key_buf)
    assert key_buf.numel() == layout.region_bytes

    impl = _make_impl(
        residual_length=residual,
        head_size=head,
        num_kv_heads=kvh,
        group_size=group,
        block_size=block,
        num_blocks=num_blocks,
    )
    impl.num_heads = kvh
    _install_cpu_packers(impl)

    sem = KiviInt4Semantics(
        MethodConfig(
            head_size=head,
            num_kv_heads=kvh,
            group_size=group,
            residual_length=residual,
            block_size=block,
        )
    )
    torch.manual_seed(31)
    query = torch.randn(tokens, kvh, head)
    key = torch.randn(tokens, kvh, head)
    value = torch.randn(tokens, kvh, head)
    out = torch.zeros(tokens, kvh, head)

    md = _metadata(
        tokens,
        block_tables=torch.tensor([[0, 1]], dtype=torch.long),
        req_ids=["r"],
    )
    impl.forward(None, query, key, value, (key_buf, value_buf), md, out)

    k_quant, k_scale, k_mn, v_quant, v_scale, v_mn = kivi_caches_from_byte_tensors(
        key_buf, value_buf, layout
    )
    assert k_quant.shape == (num_blocks, kvh, head, block // 8)
    assert k_scale.shape == (num_blocks, kvh, head, 1)  # one group per block
    assert v_quant.shape == (num_blocks, block, kvh, head // 8)

    # 128 keys flushed as a single aligned group into block 0, token 128 stayed
    # exact; values evict only the oldest slot.
    expected_k = torch.cat([sem.fake_quant_key(key[:residual]), key[residual:]])
    expected_v = torch.cat([sem.fake_quant_value(value[:1]), value[1:]])
    gathered_k, gathered_v = impl._gather_dequant_kivi_paged_cache(
        torch.tensor([[0, 1]], dtype=torch.long), [tokens], torch.float32, ["r"]
    )
    assert torch.allclose(gathered_k, expected_k, atol=1e-5)
    assert torch.allclose(gathered_v, expected_v, atol=1e-5)
    assert key_buf.any() and value_buf.any()

    # memory claim, measured rather than quoted: ~3.6x vs fp16 at this geometry
    assert layout.compression_vs_fp16() == pytest.approx(
        2 * head / layout.bytes_per_token_head, rel=1e-9
    )
    assert 3.5 < layout.compression_vs_fp16() < 3.7
