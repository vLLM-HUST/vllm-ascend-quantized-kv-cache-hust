# SPDX-License-Identifier: Apache-2.0
"""910B2 probe: one chunked step carrying several decodes *and* several prompts.

Chunked prefill is the branch with the most index arithmetic in the plugin: the
batch is decode rows first, then prompt rows, and every per-request length has
to be recovered from a cumulative ``actual_seq_lengths_q`` minus the decode
count, then written back into a slice of the shared output.  Until now the only
chunked steps ever executed on hardware had one decode row and one prompt row,
where all of that reduces to a no-op -- so nothing tested it with two decodes
and three prompts of *different* lengths.

Three things are checked:

1. each decode row equals a direct ``npu_fused_infer_attention_score`` over that
   request's own gathered history;
2. each prompt row equals a direct call over that request's own new K/V with the
   host's causal mask -- a wrong ``actual_seq_lengths`` split shows up as rows of
   one request attending over another's;
3. after the step, each prompt request's cache matches the bulk-write schedule
   (whole windows quantised, remainder exact) with the shared level comparison,
   and each decode request's newest token is still full precision.

    python scripts/npu_probe_kivi_chunked_batch.py
    KIVI_PROBE_HEAD=128 KIVI_PROBE_KV_HEADS=8 KIVI_PROBE_GROUP=128 \
    KIVI_PROBE_BLOCK=128 KIVI_PROBE_RESIDUAL=128 \
      python scripts/npu_probe_kivi_chunked_batch.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "scripts")

import kivi_probe_reference as ref  # noqa: E402
import torch  # noqa: E402
from npu_probe_kivi_attention import (  # noqa: E402
    BLOCK,
    DEV,
    GROUP,
    HEAD,
    KVH,
    NUM_HEADS,
    RESIDUAL,
    build_impl,
    byte_caches,
    direct_fia,
    host_causal_mask,
    metadata,
)

from vllm_ascend_quantized_kv_cache.methods.base import MethodConfig  # noqa: E402
from vllm_ascend_quantized_kv_cache.methods.kivi_int4.semantics import (  # noqa: E402
    KiviInt4Semantics,
)

D = int(os.environ.get("KIVI_PROBE_DECODES", 2))
P = int(os.environ.get("KIVI_PROBE_PROMPTS", 3))
TAIL = 8
DECODE_LENS = [RESIDUAL * (i + 1) + TAIL for i in range(D)]
PROMPT_LENS = [GROUP * (j + 1) + TAIL for j in range(P)]
REQS = DECODE_LENS + PROMPT_LENS
PAGES = [-(-(length + 1) // BLOCK) for length in REQS]
WIDTH = max(PAGES)
NUM_BLOCKS = sum(PAGES)
REQ_IDS = [f"dec-{i}" for i in range(D)] + [f"pre-{j}" for j in range(P)]


def block_rows() -> list[list[int]]:
    rows, start = [], 0
    for count in PAGES:
        ids = list(range(start, start + count))
        rows.append(ids + [ids[0]] * (WIDTH - count))
        start += count
    return rows


def slot_at(row: list[int], token: int) -> int:
    return row[token // BLOCK] * BLOCK + token % BLOCK


def cumulative(values: list[int]) -> list[int]:
    total, out = 0, []
    for value in values:
        total += value
        out.append(total)
    return out


def expected_history(
    sem, hist_k: torch.Tensor, hist_v: torch.Tensor, split_k: int, split_v: int
):
    """(quantised prefix + exact remainder) per side, at its own boundary.

    Keys and values are quantised at different token counts on the incremental
    write path, and each expectation must be built from its own tensor --
    passing one for both silently compares the values against the keys.
    """
    expected_k = torch.cat(
        [sem.fake_quant_key(hist_k[:split_k].float()), hist_k[split_k:].float()]
    ).half()
    expected_v = torch.cat(
        [sem.fake_quant_value(hist_v[:split_v].float()), hist_v[split_v:].float()]
    ).half()
    return expected_k, expected_v


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
    impl = build_impl(max_seqs=D + P)
    key_buf, value_buf, _ = byte_caches(num_blocks=NUM_BLOCKS)
    impl._bind_kivi_cache((key_buf, value_buf))
    mask = host_causal_mask()
    rows = block_rows()
    failures: list[str] = []
    tie_flips = 0

    print(
        f"geometry: head={HEAD} kv_heads={KVH} q_heads={NUM_HEADS} group={GROUP} "
        f"block={BLOCK} residual={RESIDUAL} decodes={D} prompts={P} "
        f"prompt_lens={PROMPT_LENS} blocks={NUM_BLOCKS}",
        flush=True,
    )

    # ---- the decode requests each get a history first =========================
    dec_hist: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i, length in enumerate(DECODE_LENS):
        k = torch.randn(length, KVH, HEAD, device=DEV, dtype=torch.float16)
        v = torch.randn_like(k)
        q = torch.randn(length, NUM_HEADS, HEAD, device=DEV, dtype=torch.float16)
        step_md = metadata(
            length,
            [length],
            "PrefillNoCache",
            torch.tensor(
                [slot_at(rows[i], t) for t in range(length)],
                dtype=torch.long,
                device=DEV,
            ),
            torch.tensor([rows[i]], dtype=torch.long, device=DEV),
            actual_seq_lengths_q=[length],
            req_ids=[REQ_IDS[i]],
            attn_mask=mask,
        )
        out = torch.zeros(length, NUM_HEADS, HEAD, device=DEV, dtype=torch.float16)
        got = impl.forward(None, q, k, v, (key_buf, value_buf), step_md, out)
        torch.npu.synchronize()
        if not bool(torch.isfinite(got).all()):
            failures.append(f"{REQ_IDS[i]} history prefill produced NaN/Inf")
        dec_hist.append((k, v))

    # ---- one chunked step: D decode rows + P prompt rows =====================
    prompt_rows: list[tuple[torch.Tensor, torch.Tensor]] = []
    keys, values, queries, slots = [], [], [], []
    for i, length in enumerate(DECODE_LENS):
        keys.append(torch.randn(1, KVH, HEAD, device=DEV, dtype=torch.float16))
        values.append(torch.randn_like(keys[-1]))
        queries.append(torch.randn(1, NUM_HEADS, HEAD, device=DEV, dtype=torch.float16))
        # the new token's index is the count of tokens already stored
        slots.append(slot_at(rows[i], length))
    for j, length in enumerate(PROMPT_LENS):
        row = rows[D + j]
        k = torch.randn(length, KVH, HEAD, device=DEV, dtype=torch.float16)
        v = torch.randn_like(k)
        keys.append(k)
        values.append(v)
        queries.append(torch.randn(length, NUM_HEADS, HEAD, device=DEV).half())
        slots.extend(slot_at(row, t) for t in range(length))
        prompt_rows.append((k, v))

    key_cat = torch.cat(keys)
    value_cat = torch.cat(values)
    query_cat = torch.cat(queries)
    tokens = int(query_cat.shape[0])
    seq_lens = [length + 1 for length in DECODE_LENS] + PROMPT_LENS
    qlen = cumulative([1] * D + PROMPT_LENS)
    ck_md = metadata(
        tokens,
        seq_lens,
        "ChunkedPrefill",
        torch.tensor(slots, dtype=torch.long, device=DEV),
        torch.tensor(rows, dtype=torch.long, device=DEV),
        actual_seq_lengths_q=qlen,
        req_ids=REQ_IDS,
        num_decodes=D,
        num_decode_tokens=D,
        num_prefills=P,
        attn_mask=mask,
    )
    ck_out = torch.zeros(tokens, NUM_HEADS, HEAD, device=DEV, dtype=torch.float16)
    got_ck = impl.forward(
        None, query_cat, key_cat, value_cat, (key_buf, value_buf), ck_md, ck_out
    )
    torch.npu.synchronize()
    print(
        f"chunked batch: rows={tokens} ({D} decode + {PROMPT_LENS}), "
        f"qlen={qlen}, seq={seq_lens}",
        flush=True,
    )
    if not bool(torch.isfinite(got_ck).all()):
        failures.append("chunked batch output has NaN/Inf")

    # 1) decode rows against the operator on their own gathered history
    for i, length in enumerate(DECODE_LENS):
        hist_k, hist_v = dec_hist[i]
        dense_k, dense_v = impl._gather_dequant_kivi_paged_cache(
            torch.tensor([rows[i]], dtype=torch.long, device=DEV),
            [length + 1],
            torch.float16,
            [REQ_IDS[i]],
        )
        reference = direct_fia(
            query_cat[i : i + 1], dense_k, dense_v, [1], [int(dense_k.shape[0])]
        )
        diff = (got_ck[i : i + 1].float() - reference.float()).abs().max().item()
        print(
            f"  {REQ_IDS[i]}: decode row over {int(dense_k.shape[0])} tokens, "
            f"max|diff| = {diff:.6f}",
            flush=True,
        )
        if diff > 1e-3:
            failures.append(f"{REQ_IDS[i]} decode row deviates ({diff})")

    # 2) prompt rows against the operator on their own new K/V
    offset = D
    for j, length in enumerate(PROMPT_LENS):
        k, v = prompt_rows[j]
        reference = direct_fia(
            query_cat[offset : offset + length],
            k,
            v,
            [length],
            [length],
            atten_mask=mask,
            sparse_mode=3,
        )
        diff = (
            (got_ck[offset : offset + length].float() - reference.float())
            .abs()
            .max()
            .item()
        )
        print(
            f"  {REQ_IDS[D + j]}: prompt row of {length} tokens, "
            f"max|diff| = {diff:.6f}",
            flush=True,
        )
        if diff > 1e-3:
            failures.append(f"{REQ_IDS[D + j]} prompt rows deviate ({diff})")
        offset += length

    # 3) cache contents after the step, per request, level by level
    for idx, (length, name) in enumerate(zip(REQS, REQ_IDS, strict=True)):
        hist_k, hist_v = dec_hist[idx] if idx < D else prompt_rows[idx - D]
        # the decode requests wrote one more token in this step
        if idx < D:
            hist_k = torch.cat([hist_k, keys[idx]])
            hist_v = torch.cat([hist_v, values[idx]])
        total = length + 1 if idx < D else length
        if idx >= D:
            # one bulk write: the same whole-window boundary applies to both
            # sides (confirmed on 910B2: keys and values quantise together)
            split_k = split_v = (total // RESIDUAL) * RESIDUAL
        else:
            # the history went in as one bulk write, then one token per step:
            # keys spill whole windows on overflow, values evict their oldest
            # slot per overflow
            split_k = ((total - 1) // RESIDUAL) * RESIDUAL
            split_v = max((length // RESIDUAL) * RESIDUAL, total - RESIDUAL)
        dense_k, dense_v = impl._gather_dequant_kivi_paged_cache(
            torch.tensor([rows[idx]], dtype=torch.long, device=DEV),
            [total],
            torch.float16,
            [name],
        )
        expect_k, expect_v = expected_history(sem, hist_k, hist_v, split_k, split_v)
        _, k_broken, k_ties = ref.score(
            sem, GROUP, dense_k, expect_k, hist_k, split_k, along_tokens=True
        )
        _, v_broken, v_ties = ref.score(
            sem, GROUP, dense_v, expect_v, hist_v, split_v, along_tokens=False
        )
        tie_flips += k_ties + v_ties
        print(
            f"  {name}: cache {total} tokens, quantised {split_k}/{split_v}, "
            f"violations k={k_broken} v={v_broken}",
            flush=True,
        )
        if k_broken or v_broken:
            failures.append(f"{name} cache is off the flush schedule (k={k_broken})")

    print(
        f"chunked batch: {D + P} requests in one step, "
        f"one-level tie flips: {tie_flips}",
        flush=True,
    )
    print("RESULT:", "PASS" if not failures else f"FAIL {failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
