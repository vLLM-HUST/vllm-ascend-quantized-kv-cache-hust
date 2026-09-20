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
    evicted_values = max(bulk_prefill_window, total - residual_length)
    expected       = quantise(that prefix) + keep the rest exact

so a step that flushes early, late, or out of order shows up immediately.  The
last steps also retire a request (its residual row must be freed), re-admit a
fresh one onto that row -- a recycled row must not leak the previous request's
full-precision tail -- and report how the int4 attention error tracks the
history length against the same attention on the unquantized cache.

    python scripts/npu_probe_kivi_generate.py
    KIVI_PROBE_HEAD=128 KIVI_PROBE_KV_HEADS=8 KIVI_PROBE_GROUP=128 \
    KIVI_PROBE_BLOCK=128 KIVI_PROBE_RESIDUAL=128 KIVI_PROBE_STEPS=64 \
      python scripts/npu_probe_kivi_generate.py
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

R = int(os.environ.get("KIVI_PROBE_SEQS", 2))
# two key-window flushes plus a partial one, so the schedule repeats
STEPS = int(os.environ.get("KIVI_PROBE_STEPS", 2 * RESIDUAL + 3))
PROMPT = max(GROUP, RESIDUAL // 2)
PAGES_PER_REQ = -(-(PROMPT + STEPS + 1) // BLOCK) + 1
NUM_BLOCKS = R * PAGES_PER_REQ  # a re-admitted request reuses freed blocks
TOLERANCE = ref.TOLERANCE
# int4 attention may not exceed this multiple of the K/V rms at any history length
ACCURACY_BOUND = 0.2


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
            k_worst, k_broken, k_ties = ref.score(
                sem,
                GROUP,
                got_k,
                expected_k,
                hist_k[i][:length],
                flushed,
                along_tokens=True,
            )
            v_worst, v_broken, v_ties = ref.score(
                sem,
                GROUP,
                got_v,
                expected_v,
                hist_v[i][:length],
                evicted,
                along_tokens=False,
            )
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
    live = R - 1
    worst_loop_k = worst_loop_v = 0.0
    trend: list[tuple[int, float]] = []
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
        if (t + 1) % max(1, STEPS // 4) == 0 or t == STEPS - 1:
            # int4 attention vs the same attention on the unquantized cache, as
            # the history grows: this is the device-side accuracy trend.
            exact_k = hist_k[live][: lengths[live]]
            exact_v = hist_v[live][: lengths[live]]
            fp16_ref = direct_fia(
                queries[live : live + 1], exact_k, exact_v, [1], [lengths[live]]
            )
            kv_rms = float(exact_v.float().pow(2).mean().sqrt()) or 1.0
            err = float((got[live : live + 1].float() - fp16_ref.float()).abs().max())
            trend.append((lengths[live], err / kv_rms))
        if t == 0 or (t + 1) % max(1, STEPS // 5) == 0 or t == STEPS - 1:
            print(
                f"step {t + 1}/{STEPS}: history={lengths[0]} "
                f"residual rows={int(impl.kivi_residual_key_len[0])}/"
                f"{int(impl.kivi_residual_value_len[0])} "
                f"key diff={k_diff:.6f} value diff={v_diff:.6f}",
                flush=True,
            )

    # ---- retiring a request must free its residual row =======================
    # the retired request's history is no longer comparable: with its window
    # released, its tail reads back as int4 history on purpose, so only the
    # survivor is verified from here on.
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
        "int4 vs fp16 by history length (x K/V rms): "
        + ", ".join(f"{length} -> {value:.4f}" for length, value in trend),
        flush=True,
    )
    if trend and trend[-1][1] > ACCURACY_BOUND:
        failures.append(f"long-context int4 error grew to {trend[-1][1]:.4f}")
    print(
        f"generation: {STEPS} steps, worst deviation over the loop "
        f"keys={worst_loop_k:.6f} values={worst_loop_v:.6f}, "
        f"one-level tie flips tolerated: {tie_flips[0]}",
        flush=True,
    )
    # ---- a new request must land on the freed row without stale reads =========
    # Continuous batching reuses rows immediately: the retired row still holds
    # req-0's slot ids and tensors, so a fresh request that gets that row must
    # not have them surface in its own gather.
    re_admitted = "req-new"
    # deliberately not a whole window: this request must keep entries in the
    # residual ring, so it has to be handed a row (a full-window prompt
    # quantises everything and needs none, which would skip the reuse path)
    re_prompt = RESIDUAL + GROUP // 2
    # the allocator hands the retired request's own blocks straight back, so the
    # newcomer writes into the slots the old request used -- exactly where a
    # residual row that was not cleared would read another request's KV back out
    new_tables = tables[:1].clone()
    new_keys = torch.randn(re_prompt, KVH, HEAD, device=DEV, dtype=torch.float16)
    new_values = torch.randn_like(new_keys)
    new_queries = torch.randn(re_prompt, NUM_HEADS, HEAD, device=DEV).half()
    new_slots = torch.tensor(
        [int(new_tables[0, t // BLOCK]) * BLOCK + t % BLOCK for t in range(re_prompt)],
        dtype=torch.long,
        device=DEV,
    )
    new_md = metadata(
        re_prompt,
        [re_prompt],
        "PrefillNoCache",
        new_slots,
        new_tables,
        actual_seq_lengths_q=[re_prompt],
        req_ids=[re_admitted],
        num_prefills=1,
        attn_mask=mask,
    )
    new_out = torch.zeros(re_prompt, NUM_HEADS, HEAD, device=DEV).half()
    got_new = impl.forward(
        None,
        new_queries,
        new_keys,
        new_values,
        (key_buf, value_buf),
        new_md,
        new_out,
    )
    torch.npu.synchronize()
    row_new = impl._get_kivi_residual_row(re_admitted, create=False)
    dense_k, dense_v = impl._gather_dequant_kivi_paged_cache(
        new_tables, [re_prompt], torch.float16, [re_admitted]
    )
    flushed = (re_prompt // RESIDUAL) * RESIDUAL
    evicted = max((re_prompt // RESIDUAL) * RESIDUAL, re_prompt - RESIDUAL)
    expect_k = torch.cat(
        [sem.fake_quant_key(new_keys[:flushed].float()), new_keys[flushed:].float()]
    ).half()
    expect_v = torch.cat(
        [
            sem.fake_quant_value(new_values[:evicted].float()),
            new_values[evicted:].float(),
        ]
    ).half()
    k_bad, k_ties = ref.prefix_check(
        sem,
        GROUP,
        dense_k[:flushed],
        expect_k[:flushed],
        new_keys[:flushed],
        along_tokens=True,
    )
    v_bad, v_ties = ref.prefix_check(
        sem,
        GROUP,
        dense_v[:evicted],
        expect_v[:evicted],
        new_values[:evicted],
        along_tokens=False,
    )
    tail_k = float((dense_k[flushed:] - expect_k[flushed:]).abs().max())
    tail_v = float((dense_v[evicted:] - expect_v[evicted:]).abs().max())
    print(
        f"re-admission: {re_admitted} (history {re_prompt}) holds residual row "
        f"{row_new}, retired {retiring} holds "
        f"{impl._get_kivi_residual_row(retiring, create=False)}; violations "
        f"k={k_bad} v={v_bad}, tie flips={k_ties + v_ties}, exact tails "
        f"{tail_k:.6f}/{tail_v:.6f}",
        flush=True,
    )
    if row_new is None:
        failures.append("re-admitted request got no residual row")
    if not bool(torch.isfinite(got_new).all()):
        failures.append("re-admitted request produced NaN/Inf")
    if k_bad or v_bad or tail_k > TOLERANCE or tail_v > TOLERANCE:
        failures.append(
            f"re-admitted request reads stale or wrong rows (k={k_bad}/{tail_k}, "
            f"v={v_bad}/{tail_v})"
        )

    print("RESULT:", "PASS" if not failures else f"FAIL {failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
