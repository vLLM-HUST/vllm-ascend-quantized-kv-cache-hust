# SPDX-License-Identifier: Apache-2.0
"""910B2 probe: INT4 decode with a ragged, multi-block batch.

``npu_probe_kivi_attention.py`` checks each attention branch with one request
and a 2-row batch.  Serving looks different: several requests decode together
with unrelated history lengths, each spanning several cache blocks, and the
batch is described to aclnn only through cumulative ``actual_seq_lengths_kv``.
That ragged layout is where a block table or a residual row leaking across
requests would show up, and no CPU test can prove it -- there the operator is
only *recorded*, not executed.

Three things are checked per batched decode step:

1. the plugin's output equals one direct ``npu_fused_infer_attention_score``
   call over the same gathered cache (isolates parameter marshalling);
2. each request's gathered keys equal that request's own int4 reference
   (quantised flushed window, exact residual tail) -- cross-request leakage
   fails here;
3. the int4 output stays close to the same attention computed on the
   *unquantized* fp16 cache, which is the device-side accuracy datapoint.

    python scripts/npu_probe_kivi_batched.py
    KIVI_PROBE_HEAD=128 KIVI_PROBE_KV_HEADS=8 KIVI_PROBE_GROUP=128 \
    KIVI_PROBE_BLOCK=128 KIVI_PROBE_RESIDUAL=128 KIVI_PROBE_SEQS=4 \
      python scripts/npu_probe_kivi_batched.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "scripts")

import torch  # noqa: E402

# Reuse the single-request probe's harness: same env geometry, same host mask.
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

R = int(os.environ.get("KIVI_PROBE_SEQS", 3))
TAIL = 8  # keeps every history length off the block boundary
# unrelated, strictly increasing histories: one, two, three ... flushed windows
# plus a tail
LENS = [RESIDUAL * (i + 1) + TAIL for i in range(R)]
BLOCKS_PER_REQ = [(length + 1 + BLOCK - 1) // BLOCK for length in LENS]
NUM_BLOCKS = sum(BLOCKS_PER_REQ)
# an int4 history may add at most this much error, as a fraction of the K/V
# rms, and must stay this correlated with the fp16 result (measured on 910B2 at
# both geometries: 0.068 rms ratio, cosine 0.990)
ACCURACY_BOUND = 0.2
COSINE_FLOOR = 0.98


def block_ids() -> list[list[int]]:
    ids, start = [], 0
    for count in BLOCKS_PER_REQ:
        ids.append(list(range(start, start + count)))
        start += count
    return ids


def slot_for(req_blocks: list[int], token: int) -> int:
    block = req_blocks[token // BLOCK]
    return block * BLOCK + token % BLOCK


def padded_table(req_blocks: list[int]) -> list[int]:
    width = max(BLOCKS_PER_REQ)
    return req_blocks + [req_blocks[0]] * (width - len(req_blocks))


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
    tables = block_ids()
    failures: list[str] = []

    print(
        f"geometry: head={HEAD} kv_heads={KVH} group={GROUP} block={BLOCK} "
        f"residual={RESIDUAL} seqs={R} histories={LENS} blocks={NUM_BLOCKS}",
        flush=True,
    )

    # ---- each request prefills on its own, into its own blocks ===============
    prompts = []
    for i, length in enumerate(LENS):
        pre_key = torch.randn(length, KVH, HEAD, device=DEV, dtype=torch.float16)
        pre_value = torch.randn_like(pre_key)
        pre_query = torch.randn_like(pre_key)
        slots = torch.tensor(
            [slot_for(tables[i], t) for t in range(length)],
            dtype=torch.long,
            device=DEV,
        )
        md = metadata(
            length,
            [length],
            "PrefillNoCache",
            slots,
            torch.tensor([padded_table(tables[i])], device=DEV),
            attn_mask=mask,
            req_ids=[f"req-{i}"],
        )
        out = torch.zeros(length, NUM_HEADS, HEAD, device=DEV, dtype=torch.float16)
        got = impl.forward(
            None, pre_query, pre_key, pre_value, (key_buf, value_buf), md, out
        )
        torch.npu.synchronize()
        if not bool(torch.isfinite(got).all()):
            failures.append(f"prefill of req-{i} produced NaN/Inf")
        prompts.append((pre_key, pre_value))

    # ---- one batched decode step over the whole ragged batch =================
    seq_lens = [length + 1 for length in LENS]
    dec_key = torch.randn(R, KVH, HEAD, device=DEV, dtype=torch.float16)
    dec_value = torch.randn_like(dec_key)
    dec_query = torch.randn_like(dec_key)
    slots = torch.tensor(
        [slot_for(tables[i], LENS[i]) for i in range(R)],
        dtype=torch.long,
        device=DEV,
    )
    md = metadata(
        R,
        seq_lens,
        "DecodeOnly",
        slots,
        torch.tensor([padded_table(row) for row in tables], device=DEV),
        actual_seq_lengths_q=list(range(1, R + 1)),
        req_ids=[f"req-{i}" for i in range(R)],
        num_decodes=R,
        num_decode_tokens=R,
        num_prefills=0,
    )
    out = torch.zeros(R, NUM_HEADS, HEAD, device=DEV, dtype=torch.float16)
    got = impl.forward(
        None, dec_query, dec_key, dec_value, (key_buf, value_buf), md, out
    )
    torch.npu.synchronize()
    if not bool(torch.isfinite(got).all()):
        failures.append("batched decode output has NaN/Inf")

    # 1) same gathered cache, direct operator call: isolates the marshalling
    dense_k, dense_v = impl._gather_dequant_kivi_paged_cache(
        md.block_tables, seq_lens, torch.float16, md.req_ids
    )
    kv_cumsum = torch.tensor(seq_lens, dtype=torch.int32).cumsum(dim=0).tolist()
    reference = direct_fia(
        dec_query, dense_k, dense_v, list(range(1, R + 1)), kv_cumsum
    )
    marshalling = (got.float() - reference.float()).abs().max().item()
    print(
        f"batched decode: gathered {tuple(dense_k.shape)}, kv lens {kv_cumsum}, "
        f"max|diff| vs direct FIA = {marshalling:.6f}",
        flush=True,
    )
    if marshalling > 1e-3:
        failures.append(f"batched decode deviates from the operator ({marshalling})")

    # 2) each request's own int4 history, no cross-request leakage
    offset = 0
    for i, length in enumerate(LENS):
        pre_key, _ = prompts[i]
        flushed = (length // RESIDUAL) * RESIDUAL
        expected = torch.cat(
            [
                sem.fake_quant_key(pre_key[:flushed].float()),
                pre_key[flushed:].float(),
                dec_key[i : i + 1].float(),
            ]
        ).half()
        got_k = dense_k[offset : offset + length + 1]
        diff = (got_k - expected).abs().max().item()
        print(
            f"  req-{i}: history {length + 1} over {BLOCKS_PER_REQ[i]} blocks, "
            f"keys vs int4 reference max|diff| = {diff:.6f}",
            flush=True,
        )
        if diff > 1e-2:
            failures.append(f"req-{i} gathered keys deviate ({diff})")
        offset += length + 1

    # 3) accuracy against the same attention on the unquantized fp16 cache.
    # The deviation is normalised by the K/V scale rather than by the attention
    # output: with random keys the softmax is diffuse, so the output sits at
    # ~1/sqrt(N) of the value scale while the quantization error does not, and
    # an output-relative ratio would look alarming for any quantized scheme.
    worst = 0.0
    worst_cosine = 1.0
    for i, length in enumerate(LENS):
        pre_key, pre_value = prompts[i]
        exact_k = torch.cat([pre_key, dec_key[i : i + 1]])
        exact_v = torch.cat([pre_value, dec_value[i : i + 1]])
        fp16_ref = direct_fia(dec_query[i : i + 1], exact_k, exact_v, [1], [length + 1])
        error = (got[i : i + 1].float() - fp16_ref.float()).abs().max().item()
        kv_scale = float(exact_v.float().pow(2).mean().sqrt()) or 1.0
        flat_got = got[i : i + 1].float().flatten()
        flat_ref = fp16_ref.float().flatten()
        cosine = float(
            torch.dot(flat_got, flat_ref) / (flat_got.norm() * flat_ref.norm() + 1e-12)
        )
        worst = max(worst, error / kv_scale)
        worst_cosine = min(worst_cosine, cosine)
        print(
            f"  req-{i}: int4 vs fp16 max|diff| = {error:.6f} "
            f"(= {error / kv_scale:.4f} x K/V rms, cosine {cosine:.6f})",
            flush=True,
        )
    print(
        f"int4 accuracy: worst |diff|/K-V rms = {worst:.6f}, "
        f"worst cosine = {worst_cosine:.6f}",
        flush=True,
    )
    if worst > ACCURACY_BOUND:
        failures.append(f"int4 attention deviates from fp16 by {worst:.4f} of K/V rms")
    if worst_cosine < COSINE_FLOOR:
        failures.append(f"int4 attention correlates with fp16 only {worst_cosine:.4f}")

    print("RESULT:", "PASS" if not failures else f"FAIL {failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
