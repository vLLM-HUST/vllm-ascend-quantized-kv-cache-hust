# SPDX-License-Identifier: Apache-2.0
"""910B2 probe: does the KIVI INT4 attention path call fused attention right?

``tests/test_kivi_int4.py`` pins the fused-attention branches against a
recording ``torch_npu`` stub; this drives the same branches against the real
``npu_fused_infer_attention_score``.  The reference is the operator called
directly on the plugin's own dequant-gathered K/V, so a mismatch points at the
plugin's parameter marshalling (layout, seq lengths, slicing, output write),
not at the operator's numerics.

Causal prefill is run with the additive mask the host builder supplies
(``float16 [1, T, T]``, ``-inf`` above the diagonal): aclnn rejects
``sparse_mode=3`` without a mask (error 561002), which is why the metadata
contract requires ``attn_mask`` whenever ``causal`` is set.

    python scripts/npu_probe_kivi_attention.py
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, "src")

import torch  # noqa: E402
import torch_npu  # noqa: E402

from vllm_ascend_quantized_kv_cache.methods.base import MethodConfig  # noqa: E402
from vllm_ascend_quantized_kv_cache.methods.kivi_int4.attention_backend import (  # noqa: E402
    AscendKiviInt4AttentionBackendMixin,
)
from vllm_ascend_quantized_kv_cache.methods.kivi_int4.byte_cache import (  # noqa: E402
    KiviByteCacheLayout,
)
from vllm_ascend_quantized_kv_cache.methods.kivi_int4.semantics import (  # noqa: E402
    KiviInt4Semantics,
)

DEV = os.environ.get("KIVI_PROBE_DEVICE", "npu:0")
# defaults are a small probe; override to the shipped geometry with
# KIVI_PROBE_HEAD=128 KIVI_PROBE_KV_HEADS=8 KIVI_PROBE_GROUP=128
# KIVI_PROBE_BLOCK=128 KIVI_PROBE_RESIDUAL=128
HEAD = int(os.environ.get("KIVI_PROBE_HEAD", 64))
KVH = int(os.environ.get("KIVI_PROBE_KV_HEADS", 2))
GROUP = int(os.environ.get("KIVI_PROBE_GROUP", 32))
BLOCK = int(os.environ.get("KIVI_PROBE_BLOCK", 32))
RESIDUAL = int(os.environ.get("KIVI_PROBE_RESIDUAL", 32))
# real models are grouped-query: every kv head serves several query heads.
# KIVI_PROBE_GQA=7 gives the Llama-style 28Q/4KV split.
GQA = int(os.environ.get("KIVI_PROBE_GQA", 1))
NUM_HEADS = KVH * GQA
SCALE = HEAD**-0.5
MAX_SEQS = int(os.environ.get("KIVI_PROBE_SEQS", 2))
PREFILL_TOKENS = RESIDUAL + 8  # one full key window flushes into int4 history
CHUNK_PROMPT = GROUP  # the all-new prompt row added by the chunked step
# block bookkeeping has to follow the geometry: with residual_length >
# block_size one request spans several blocks, and block_size !=
# residual_length is exactly where the window/slot arithmetic can disagree
HISTORY_TOKENS = PREFILL_TOKENS + 2
HISTORY_BLOCKS = -(-HISTORY_TOKENS // BLOCK)
PROMPT_BLOCKS = -(-CHUNK_PROMPT // BLOCK)
PROMPT_BASE = HISTORY_BLOCKS  # the prompt row gets blocks of its own
WIDTH = max(HISTORY_BLOCKS, PROMPT_BLOCKS)
NUM_BLOCKS = max(
    int(os.environ.get("KIVI_PROBE_BLOCKS", 4)), PROMPT_BASE + PROMPT_BLOCKS
)


def table(first_block: int, tokens: int) -> list[int]:
    """Block ids for ``tokens`` consecutive tokens, padded to a common width."""
    ids = [first_block + i for i in range(-(-tokens // BLOCK))]
    return ids + [ids[0]] * (WIDTH - len(ids))


def slot(row: list[int], token: int) -> int:
    return row[token // BLOCK] * BLOCK + token % BLOCK


class _HostImplShim:
    """The attributes the mixin expects from the live host impl."""

    def __init__(self, vllm_config):
        self.num_heads = NUM_HEADS
        self.num_kv_heads = KVH
        self.num_queries_per_kv = GQA
        self.head_size = HEAD
        self.scale = SCALE
        self.vllm_config = vllm_config


def build_impl(max_seqs: int | None = None):
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            kivi_group_size=GROUP, kivi_residual_length=RESIDUAL
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=MAX_SEQS if max_seqs is None else max_seqs
        ),
    )

    class Impl(AscendKiviInt4AttentionBackendMixin, _HostImplShim):
        def __init__(self, config):
            _HostImplShim.__init__(self, config)
            self._init_kivi_state("kivi_int4", config)

    return Impl(vllm_config)


def byte_caches(num_blocks: int | None = None):
    blocks = NUM_BLOCKS if num_blocks is None else num_blocks
    layout = KiviByteCacheLayout(
        num_blocks=blocks,
        block_size=BLOCK,
        num_kv_heads=KVH,
        head_size=HEAD,
        group_size=GROUP,
    )
    key = torch.zeros(
        blocks,
        BLOCK,
        KVH,
        layout.bytes_per_token_head,
        dtype=torch.uint8,
        device=DEV,
    )
    value = torch.zeros_like(key)
    return key, value, layout


def host_causal_mask():
    """The mask the Ascend metadata builder hands a causal prefill.

    Not a T x T additive mask: aclnn's split-fuse path wants the builder's
    fixed ``triu(2048, 2048)`` int8 mask, so the probe takes it from the host.
    """
    try:
        from vllm_ascend.attention.attention_mask import AttentionMaskBuilder

        builder = AttentionMaskBuilder(torch.device(DEV))
        mask = builder.get_attention_mask(
            True, SimpleNamespace(runner_type="generate", dtype=torch.float16)
        )
        source = "vllm_ascend.AttentionMaskBuilder"
    except Exception as exc:  # pragma: no cover - host stack optional
        print(f"host mask builder unavailable ({type(exc).__name__}: {exc})")
        mask = torch.triu(
            torch.ones(2048, 2048, dtype=torch.int8, device=DEV), diagonal=1
        )
        source = "local replica"
    print(f"causal mask: {tuple(mask.shape)} {mask.dtype} from {source}", flush=True)
    return mask


def metadata(tokens, seq_lens, state, slots, block_tables, **overrides):
    md = SimpleNamespace(
        attn_state=SimpleNamespace(name=state),
        actual_seq_lengths_q=[tokens],
        seq_lens_list=seq_lens,
        slot_mapping=slots,
        block_tables=block_tables,
        req_ids=["r"],
        num_actual_tokens=tokens,
        causal=True,
        finished_req_ids=None,
        attn_mask=None,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
    )
    for key, value in overrides.items():
        setattr(md, key, value)
    return md


def direct_fia(query, key, value, qlen, kvlen, atten_mask=None, sparse_mode=0):
    kwargs = {}
    if atten_mask is not None:
        kwargs["atten_mask"] = atten_mask
    out, _ = torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=key,
        value=value,
        input_layout="TND",
        block_table=None,
        actual_seq_lengths=qlen,
        actual_seq_lengths_kv=kvlen,
        num_key_value_heads=KVH,
        num_heads=NUM_HEADS,
        scale=SCALE,
        sparse_mode=sparse_mode,
        **kwargs,
    )
    return out


def main() -> int:
    torch.manual_seed(0)
    sem = KiviInt4Semantics(
        MethodConfig(
            head_size=HEAD,
            num_kv_heads=KVH,
            group_size=GROUP,
            residual_length=RESIDUAL,
            block_size=BLOCK,
        )
    )
    impl = build_impl()
    key_buf, value_buf, _ = byte_caches()
    impl._bind_kivi_cache((key_buf, value_buf))

    failures: list[str] = []

    # ---- prefill (PrefillNoCache): packs via the triton kernels, attends dense
    pre_key = torch.randn(PREFILL_TOKENS, KVH, HEAD, device=DEV, dtype=torch.float16)
    pre_value = torch.randn_like(pre_key)
    pre_query = torch.randn(
        PREFILL_TOKENS, NUM_HEADS, HEAD, device=DEV, dtype=torch.float16
    )
    slots = torch.arange(PREFILL_TOKENS, dtype=torch.long, device=DEV)
    out = torch.zeros(PREFILL_TOKENS, NUM_HEADS, HEAD, device=DEV, dtype=torch.float16)
    mask = host_causal_mask()
    md = metadata(
        PREFILL_TOKENS,
        [PREFILL_TOKENS],
        "PrefillNoCache",
        slots,
        torch.tensor([table(0, PREFILL_TOKENS)], dtype=torch.long, device=DEV),
        attn_mask=mask,
    )
    got = impl.forward(
        None, pre_query, pre_key, pre_value, (key_buf, value_buf), md, out
    )
    torch.npu.synchronize()
    if not bool(torch.isfinite(got).all()):
        failures.append("prefill output has NaN/Inf")
    if int(key_buf.abs().sum()) == 0:
        failures.append("prefill wrote nothing into the int4 history")
    print(
        f"prefill: {PREFILL_TOKENS} tokens, history bytes written="
        f"{int(key_buf.abs().sum())}, finite={bool(torch.isfinite(got).all())}",
        flush=True,
    )

    # ---- decode: the plugin gathers history + residual, then calls FIA once
    dec_key = torch.randn(1, KVH, HEAD, device=DEV, dtype=torch.float16)
    dec_value = torch.randn_like(dec_key)
    dec_query = torch.randn(1, NUM_HEADS, HEAD, device=DEV, dtype=torch.float16)
    seq_len = PREFILL_TOKENS + 1
    hist_row = table(0, HISTORY_TOKENS)
    dec_slots = torch.tensor(
        [slot(hist_row, PREFILL_TOKENS)], dtype=torch.long, device=DEV
    )
    block_tables = torch.tensor([hist_row], dtype=torch.long, device=DEV)
    dec_out = torch.zeros(1, NUM_HEADS, HEAD, device=DEV, dtype=torch.float16)
    dec_md = metadata(
        1,
        [seq_len],
        "DecodeOnly",
        dec_slots,
        block_tables,
        actual_seq_lengths_q=[1],
    )
    got_dec = impl.forward(
        None, dec_query, dec_key, dec_value, (key_buf, value_buf), dec_md, dec_out
    )
    torch.npu.synchronize()

    gathered_k, gathered_v = impl._gather_dequant_kivi_paged_cache(
        block_tables, [seq_len], torch.float16, ["r"]
    )
    reference = direct_fia(
        dec_query[:1], gathered_k, gathered_v, [1], [int(gathered_k.shape[0])]
    )
    diff = (got_dec.float() - reference.float()).abs().max().item()
    print(
        f"decode: gathered {tuple(gathered_k.shape)}, "
        f"max|diff| vs direct FIA = {diff:.6f}",
        flush=True,
    )
    if not bool(torch.isfinite(got_dec).all()):
        failures.append("decode output has NaN/Inf")
    if diff > 1e-3:
        failures.append(f"decode attention deviates from the operator ({diff})")

    # the gathered history must really be int4-dequantized, not the raw prompt
    expected_k = torch.cat(
        [
            sem.fake_quant_key(pre_key[:RESIDUAL].float()),
            pre_key[RESIDUAL:].float(),
            dec_key.float(),
        ]
    ).half()
    k_diff = (gathered_k - expected_k).abs().max().item()
    print(
        f"decode: gathered keys vs int4 reference max|diff| = {k_diff:.6f}", flush=True
    )
    if k_diff > 1e-2:
        failures.append(f"gathered keys deviate from the int4 reference ({k_diff})")

    # the prefill entry flushes whole windows for keys *and* values, so only the
    # remainder plus this decode step stays full precision (the slot-at-a-time
    # value eviction of the incremental write path is pinned by the CPU tests)
    flushed = (PREFILL_TOKENS // RESIDUAL) * RESIDUAL
    expected_residual = PREFILL_TOKENS - flushed + 1
    for name, length in (
        ("key", impl.kivi_residual_key_len[0]),
        ("value", impl.kivi_residual_value_len[0]),
    ):
        if int(length) != expected_residual:
            failures.append(
                f"{name} residual rows hold {length}, expected {expected_residual}"
            )

    print(
        f"geometry: head={HEAD} kv_heads={KVH} q_heads={NUM_HEADS} group={GROUP} "
        f"block={BLOCK} residual={RESIDUAL} tokens={PREFILL_TOKENS} "
        f"history_blocks={HISTORY_BLOCKS} prompt_base={PROMPT_BASE} "
        f"blocks={NUM_BLOCKS}",
        flush=True,
    )
    # ---- chunked prefill: a decode row plus an all-new prompt row ============
    prompt = GROUP
    rows = 1 + prompt
    seq_r = PREFILL_TOKENS + 2
    prompt_row = table(PROMPT_BASE, prompt)
    slots = torch.cat(
        [
            torch.tensor(
                [slot(hist_row, PREFILL_TOKENS + 1)], dtype=torch.long, device=DEV
            ),
            torch.tensor(
                [slot(prompt_row, t) for t in range(prompt)],
                dtype=torch.long,
                device=DEV,
            ),
        ]
    )
    tables = torch.tensor([hist_row, prompt_row], dtype=torch.long, device=DEV)
    ck_query = torch.randn(rows, NUM_HEADS, HEAD, device=DEV, dtype=torch.float16)
    ck_key = torch.randn(rows, KVH, HEAD, device=DEV, dtype=torch.float16)
    ck_value = torch.randn_like(ck_key)
    ck_md = metadata(
        rows,
        [seq_r, prompt],
        "ChunkedPrefill",
        slots,
        tables,
        actual_seq_lengths_q=[1, rows],
        req_ids=["r", "p"],
        num_decodes=1,
        num_decode_tokens=1,
        num_prefills=1,
        attn_mask=mask,
    )
    ck_out = torch.zeros(rows, NUM_HEADS, HEAD, device=DEV, dtype=torch.float16)
    got_ck = impl.forward(
        None, ck_query, ck_key, ck_value, (key_buf, value_buf), ck_md, ck_out
    )
    torch.npu.synchronize()

    history_k, history_v = impl._gather_dequant_kivi_paged_cache(
        tables[:1], [seq_r], torch.float16, ["r"]
    )
    ref_decode = direct_fia(
        ck_query[:1], history_k, history_v, [1], [int(history_k.shape[0])]
    )
    ref_prefill = direct_fia(
        ck_query[1:rows],
        ck_key[1:rows],
        ck_value[1:rows],
        [prompt],
        [prompt],
        atten_mask=mask,
        sparse_mode=3,
    )
    decode_diff = (got_ck[:1].float() - ref_decode.float()).abs().max().item()
    prefill_diff = (got_ck[1:rows].float() - ref_prefill.float()).abs().max().item()
    print(
        f"chunked: rows={rows} (1 decode + {prompt} prompt), "
        f"decode max|diff|={decode_diff:.6f}, prefill max|diff|={prefill_diff:.6f}",
        flush=True,
    )
    if not bool(torch.isfinite(got_ck).all()):
        failures.append("chunked output has NaN/Inf")
    if decode_diff > 1e-3:
        failures.append(
            f"chunked decode rows deviate from the operator ({decode_diff})"
        )
    if prefill_diff > 1e-3:
        failures.append(
            f"chunked prefill rows deviate from the operator ({prefill_diff})"
        )

    print("RESULT:", "PASS" if not failures else f"FAIL {failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
