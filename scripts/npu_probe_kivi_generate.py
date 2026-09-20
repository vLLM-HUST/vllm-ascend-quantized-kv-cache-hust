# SPDX-License-Identifier: Apache-2.0
"""910B2 probe: INT4 across a generation, not across a single step.

Every other check writes once and reads once.  Serving is a loop: tokens arrive
one at a time, the residual window fills, whole key windows spill into the int4
history while values are evicted one oldest slot at a time -- repeatedly,
mid-generation, with the real triton packer attached.  The CPU suite pins that
schedule (`tests/test_kivi_int4.py::test_int4_pipeline_across_geometries`) but
only with pure-torch packers and one bulk write.

The probe prefills a short prompt per request, then runs ``KIVI_PROBE_STEPS``
batched decode steps, crossing several window flushes and block boundaries.
After *every* step it compares the dequant-gathered cache against the pinned
rule applied to the fp16 history accumulated alongside:

    flushed_keys   = ((total - 1) // residual_length) * residual_length
    evicted_values = max(0, total - residual_length)
    expected       = quantise(that prefix) + keep the rest exact

so a step that flushes early, late, or out of order shows up immediately.  The
last steps also retire a request (its residual row must be freed) and cross-check
the attention call against a direct ``npu_fused_infer_attention_score``.

    python scripts/npu_probe_kivi_generate.py
    KIVI_PROBE_HEAD=128 KIVI_PROBE_KV_HEADS=8 KIVI_PROBE_GROUP=128 \
    KIVI_PROBE_BLOCK=128 KIVI_PROBE_RESIDUAL=128 KIVI_PROBE_STEPS=64 \
      python scripts/npu_probe_kivi_generate.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "scripts")

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

R = int(os.environ.get("KIVI_PROBE_SEQS", 2))
# two key-window flushes plus a partial one, so the schedule repeats
STEPS = int(os.environ.get("KIVI_PROBE_STEPS", 2 * RESIDUAL + 3))
PROMPT = max(GROUP, RESIDUAL // 2)
PAGES_PER_REQ = -(-(PROMPT + STEPS + 1) // BLOCK) + 1
NUM_BLOCKS = R * PAGES_PER_REQ
# fp16 dequantisation vs the triton packer's own rounding, for tokens that must
# still be exact
TOLERANCE = 1e-2


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
    impl = build_impl(max_seqs=R)
    key_buf, value_buf, _ = byte_caches(num_blocks=NUM_BLOCKS)
    impl._bind_kivi_cache((key_buf, value_buf))
    mask = host_causal_mask()
    failures: list[str] = []
    tie_flips = [0]  # one-level kernel/reference disagreements at rounding ties

    req_ids = [f"req-{i}" for i in range(R)]
    tables = torch.tensor(
        [[i * PAGES_PER_REQ + b for b in range(PAGES_PER_REQ)] for i in range(R)],
        dtype=torch.long,
        device=DEV,
    )
    # fp16 ground truth written alongside the cache, one row per token
    hist_k = [
        torch.empty(0, KVH, HEAD, device=DEV, dtype=torch.float16) for _ in range(R)
    ]
    hist_v = [
        torch.empty(0, KVH, HEAD, device=DEV, dtype=torch.float16) for _ in range(R)
    ]

    def slot_at(req_idx: int, token: int) -> int:
        block = int(tables[req_idx, token // BLOCK])
        return block * BLOCK + token % BLOCK

    def group_stats(x: torch.Tensor, *, along_tokens: bool):
        """Per-element (min, step) of the reference grid, as semantics groups it.

        Keys group along the token dim and values along the head dim; mirroring
        that here is what makes a *level* comparison possible rather than a
        fuzzy magnitude one.
        """
        work = x.float()
        if along_tokens:
            pad = (GROUP - work.shape[0] % GROUP) % GROUP
            if pad:
                work = torch.cat([work, work[-1:].expand(pad, *work.shape[1:])], 0)
            grouped = work.view(-1, GROUP, *x.shape[1:])
            mn = grouped.amin(dim=1, keepdim=True)
            mx = grouped.amax(dim=1, keepdim=True)
            scale = sem.group_scale(mn, mx, 4)
            shape = tuple(x.shape)
            return (
                mn.expand_as(grouped).reshape(shape),
                scale.expand_as(grouped).reshape(shape),
            )
        pad = (GROUP - work.shape[-1] % GROUP) % GROUP
        if pad:
            work = torch.cat([work, work[..., -1:].expand(*work.shape[:-1], pad)], -1)
        grouped = work.view(*work.shape[:-1], -1, GROUP)
        mn = grouped.amin(dim=-1, keepdim=True)
        mx = grouped.amax(dim=-1, keepdim=True)
        scale = sem.group_scale(mn, mx, 4)
        shape = tuple(x.shape)
        return mn.expand_as(grouped).reshape(shape), scale.expand_as(grouped).reshape(
            shape
        )

    def prefix_check(
        got: torch.Tensor,
        expected: torch.Tensor,
        exact: torch.Tensor,
        *,
        along_tokens: bool,
    ) -> tuple[int, int]:
        """Violations and tie flips in the quantised region.

        An element may differ from the reference only by sitting on the
        *adjacent* level, which happens when its normalised value is an exact
        half -- the triton kernel and ``quantize_group`` break those in
        opposite directions.  Such a flip is recognisable because both candidate
        levels are equally close to the exact value.  Anything else counts,
        in particular a token left in full precision: that is strictly closer to
        the exact value than any level is, so the equidistance test rejects it.
        """
        got = got.float()
        expected = expected.float()
        exact = exact.float()
        _, scale = group_stats(exact, along_tokens=along_tokens)
        scale = scale.float()
        diff = (got - expected).abs()
        d_ref = (exact - expected).abs()
        d_got = (exact - got).abs()
        tie = (diff <= 1.05 * scale) & ((d_got - d_ref).abs() <= 0.05 * scale)
        allowed = (diff <= 1e-3) | tie
        return int((~allowed).sum()), int((tie & (diff > 1e-3)).sum())

    def score(
        got: torch.Tensor,
        expected: torch.Tensor,
        exact: torch.Tensor,
        split: int,
        *,
        along_tokens: bool,
    ):
        """Violation count and worst deviation for the two regions.

        The quantised prefix is checked level by level (see prefix_check); the
        exact tail has no such excuse and must match to fp16 rounding.
        """
        worst = 0.0
        broken = 0
        ties = 0
        if split:
            diff = (got[:split] - expected[:split]).abs().amax().item()
            worst = max(worst, diff)
            broken_here, ties_here = prefix_check(
                got[:split], expected[:split], exact[:split], along_tokens=along_tokens
            )
            broken += broken_here
            ties += ties_here
        if split < got.shape[0]:
            diff = (got[split:] - expected[split:]).abs().amax().item()
            worst = max(worst, diff)
            if diff > TOLERANCE:
                broken += 1
        return worst, broken, ties

    def verify(
        lengths: list[int], tag: str, *, bulk: bool, only: list[int] | None = None
    ) -> tuple[float, float]:
        """Compare the live cache against the flush schedule the code applies.

        The two write paths differ exactly at a full window: the bulk prefill
        flushes ``(tokens // residual) * residual`` eagerly, while the
        incremental path only flushes on overflow, i.e.
        ``((tokens - 1) // residual) * residual``.  Values follow the bulk
        window on the first write and then one oldest slot per overflow.
        ``tests/test_kivi_int4.py`` pins both halves of that asymmetry.
        """
        rows = list(range(len(lengths))) if only is None else only
        sub_lengths = [lengths[i] for i in rows]
        dense_k, dense_v = impl._gather_dequant_kivi_paged_cache(
            tables[rows], sub_lengths, torch.float16, [req_ids[i] for i in rows]
        )
        worst_k = worst_v = 0.0
        offset = 0
        for i, length in zip(rows, sub_lengths, strict=True):
            if bulk:
                flushed = (length // RESIDUAL) * RESIDUAL
            else:
                flushed = ((length - 1) // RESIDUAL) * RESIDUAL
            evicted = max((PROMPT // RESIDUAL) * RESIDUAL, length - RESIDUAL)
            expected_k = torch.cat(
                [
                    sem.fake_quant_key(hist_k[i][:flushed].float()),
                    hist_k[i][flushed:].float(),
                ]
            ).half()
            expected_v = torch.cat(
                [
                    sem.fake_quant_value(hist_v[i][:evicted].float()),
                    hist_v[i][evicted:].float(),
                ]
            ).half()
            got_k = dense_k[offset : offset + length]
            got_v = dense_v[offset : offset + length]
            offset += length
            k_worst, k_broken, k_ties = score(
                got_k, expected_k, hist_k[i][:length], flushed, along_tokens=True
            )
            v_worst, v_broken, v_ties = score(
                got_v, expected_v, hist_v[i][:length], evicted, along_tokens=False
            )  # noqa: E501
            worst_k = max(worst_k, k_worst)
            worst_v = max(worst_v, v_worst)
            tie_flips[0] += k_ties + v_ties
            for what, count in (("keys", k_broken), ("values", v_broken)):
                if count:
                    failures.append(f"{tag}: req-{i} {what} off-grid x{count}")
            if length > hist_k[i].shape[0]:
                failures.append(
                    f"{tag}: req-{i} gathered {length} > {hist_k[i].shape[0]}"
                )
        return worst_k, worst_v

    print(
        f"geometry: head={HEAD} kv_heads={KVH} q_heads={NUM_HEADS} group={GROUP} "
        f"block={BLOCK} residual={RESIDUAL} seqs={R} prompt={PROMPT} "
        f"steps={STEPS} pages/req={PAGES_PER_REQ} blocks={NUM_BLOCKS}",
        flush=True,
    )

    # ---- prefill: one bulk write per request =================================
    lengths = [PROMPT] * R
    slots = torch.tensor(
        [slot_at(i, t) for i in range(R) for t in range(PROMPT)],
        dtype=torch.long,
        device=DEV,
    )
    pre_keys = torch.randn(PROMPT * R, KVH, HEAD, device=DEV, dtype=torch.float16)
    pre_values = torch.randn_like(pre_keys)
    pre_queries = torch.randn(PROMPT * R, NUM_HEADS, HEAD, device=DEV).half()
    for i in range(R):
        hist_k[i] = pre_keys[i * PROMPT : (i + 1) * PROMPT].clone()
        hist_v[i] = pre_values[i * PROMPT : (i + 1) * PROMPT].clone()
    prefill_md = metadata(
        PROMPT * R,
        lengths,
        "PrefillNoCache",
        slots,
        tables,
        actual_seq_lengths_q=[PROMPT * (i + 1) for i in range(R)],
        req_ids=req_ids,
        num_prefills=R,
        attn_mask=mask,
    )
    prefill_out = torch.zeros(PROMPT * R, NUM_HEADS, HEAD, device=DEV).half()
    got_pre = impl.forward(
        None,
        pre_queries,
        pre_keys,
        pre_values,
        (key_buf, value_buf),
        prefill_md,
        prefill_out,
    )
    torch.npu.synchronize()
    if not bool(torch.isfinite(got_pre).all()):
        failures.append("prefill produced NaN/Inf")
    k_diff, v_diff = verify(lengths, "prefill", bulk=True)
    print(
        f"prefill: history={lengths} key diff={k_diff:.6f} value diff={v_diff:.6f}",
        flush=True,
    )

    # ---- decode loop: one token per request per step =========================
    worst_loop_k = worst_loop_v = 0.0
    for t in range(STEPS):
        keys = torch.randn(R, KVH, HEAD, device=DEV, dtype=torch.float16)
        kvs = torch.randn_like(keys)
        queries = torch.randn(R, NUM_HEADS, HEAD, device=DEV, dtype=torch.float16)
        lengths = [length + 1 for length in lengths]
        slots = torch.tensor(
            [slot_at(i, lengths[i] - 1) for i in range(R)], dtype=torch.long, device=DEV
        )
        for i in range(R):
            hist_k[i] = torch.cat([hist_k[i], keys[i : i + 1]])
            hist_v[i] = torch.cat([hist_v[i], kvs[i : i + 1]])
        md = metadata(
            R,
            lengths,
            "DecodeOnly",
            slots,
            tables,
            actual_seq_lengths_q=list(range(1, R + 1)),
            req_ids=req_ids,
            num_decodes=R,
            num_decode_tokens=R,
            num_prefills=0,
        )
        out = torch.zeros(R, NUM_HEADS, HEAD, device=DEV, dtype=torch.float16)
        got = impl.forward(None, queries, keys, kvs, (key_buf, value_buf), md, out)
        torch.npu.synchronize()
        if not bool(torch.isfinite(got).all()):
            failures.append(f"step {t + 1} (history {lengths[0]}) produced NaN/Inf")
            break
        k_diff, v_diff = verify(lengths, f"step {t + 1}", bulk=False)
        worst_loop_k = max(worst_loop_k, k_diff)
        worst_loop_v = max(worst_loop_v, v_diff)
        if t == 0 or (t + 1) % max(1, STEPS // 5) == 0 or t == STEPS - 1:
            print(
                f"step {t + 1}/{STEPS}: history={lengths[0]} "
                f"residual rows={int(impl.kivi_residual_key_len[0])}/"
                f"{int(impl.kivi_residual_value_len[0])} "
                f"key diff={k_diff:.6f} value diff={v_diff:.6f}",
                flush=True,
            )

    # ---- retiring a request must free its residual row =======================
    live = R - 1  # the retired request's history is no longer comparable: with
    # its window released, its tail reads back as int4 history on purpose.
    keys = torch.randn(R, KVH, HEAD, device=DEV, dtype=torch.float16)
    kvs = torch.randn_like(keys)
    queries = torch.randn(R, NUM_HEADS, HEAD, device=DEV, dtype=torch.float16)
    retiring = req_ids[0]
    lengths = [length + 1 for length in lengths]
    slots = torch.tensor(
        [slot_at(i, lengths[i] - 1) for i in range(R)], dtype=torch.long, device=DEV
    )
    for i in range(R):
        hist_k[i] = torch.cat([hist_k[i], keys[i : i + 1]])
        hist_v[i] = torch.cat([hist_v[i], kvs[i : i + 1]])
    md = metadata(
        R,
        lengths,
        "DecodeOnly",
        slots,
        tables,
        actual_seq_lengths_q=list(range(1, R + 1)),
        req_ids=req_ids,
        num_decodes=R,
        num_decode_tokens=R,
        num_prefills=0,
        finished_req_ids=[retiring],
    )
    out = torch.zeros(R, NUM_HEADS, HEAD, device=DEV, dtype=torch.float16)
    got = impl.forward(None, queries, keys, kvs, (key_buf, value_buf), md, out)
    torch.npu.synchronize()
    if impl._get_kivi_residual_row(retiring, create=False) is not None:
        failures.append(f"retired request {retiring} still holds a residual row")
    kept_k, kept_v = verify(lengths, "post-retirement", bulk=False, only=[live])
    print(
        f"retirement: {retiring} released; survivor {req_ids[live]} still exact "
        f"at history {lengths[live]} (keys {kept_k:.6f}, values {kept_v:.6f})",
        flush=True,
    )

    # ---- attention marshalling at the end of a long generation ===============
    # forward first, then gather, so the reference sees exactly the state the
    # plugin attended over.
    one_lengths = [lengths[live]]
    one_tables = tables[live : live + 1]
    one_slots = slots[live : live + 1]
    final_query = torch.randn(1, NUM_HEADS, HEAD, device=DEV, dtype=torch.float16)
    final_out = torch.zeros(1, NUM_HEADS, HEAD, device=DEV, dtype=torch.float16)
    final_md = metadata(
        1,
        one_lengths,
        "DecodeOnly",
        one_slots,
        one_tables,
        actual_seq_lengths_q=[1],
        req_ids=[req_ids[live]],
        num_decodes=1,
        num_decode_tokens=1,
        num_prefills=0,
    )
    got_final = impl.forward(
        None,
        final_query,
        keys[live : live + 1],
        kvs[live : live + 1],
        (key_buf, value_buf),
        final_md,
        final_out,
    )
    torch.npu.synchronize()
    dense_k, dense_v = impl._gather_dequant_kivi_paged_cache(
        one_tables, one_lengths, torch.float16, [req_ids[live]]
    )
    reference = direct_fia(final_query, dense_k, dense_v, [1], [int(dense_k.shape[0])])
    marshalling = (got_final.float() - reference.float()).abs().max().item()
    print(
        f"final step: {req_ids[live]} history={one_lengths} gathered "
        f"{tuple(dense_k.shape)}, max|diff| vs direct FIA = {marshalling:.6f}",
        flush=True,
    )
    if marshalling > 1e-3:
        failures.append(f"generation ended off the operator ({marshalling})")

    print(
        f"generation: {STEPS} steps, worst deviation over the loop "
        f"keys={worst_loop_k:.6f} values={worst_loop_v:.6f}, "
        f"one-level tie flips tolerated: {tie_flips[0]}",
        flush=True,
    )
    print("RESULT:", "PASS" if not failures else f"FAIL {failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
