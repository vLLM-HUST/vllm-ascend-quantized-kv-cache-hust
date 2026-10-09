# SPDX-License-Identifier: Apache-2.0
"""Offline INT4 (KIVI) generation through the real vLLM engine.

Run against a host tree carrying the ``scripts/host_int4_patch.py`` edits and
the plugin on ``PYTHONPATH``; the point is the *integration*, not model
quality:

1. ``kv_cache_dtype="kivi_int4"`` survives config validation and cache
   allocation (the host hands each layer two equal uint8 byte buffers),
2. the plugin binds the six views out of them instead of failing closed,
3. the engine runs prefill + decode + continuous batching and emits tokens.

Compare its output against the same run with ``KIVI_E2E_DTYPE=auto`` (dense
fp16 cache) to see what 4-bit KV costs in a real model.

    KIVI_E2E_DTYPE=kivi_int4 python scripts/npu_e2e_kivi_generate.py
"""

from __future__ import annotations

import os
import sys

from vllm import LLM, SamplingParams

DTYPE = os.environ.get("KIVI_E2E_DTYPE", "kivi_int4")
MODEL = os.environ.get("KIVI_E2E_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
BLOCK = int(os.environ.get("KIVI_E2E_BLOCK", 128))
PROMPTS = [
    "The capital of France is",
    "List three prime numbers greater than ten:",
    "Once upon a time",
]


def main() -> int:
    llm = LLM(
        model=MODEL,
        kv_cache_dtype=DTYPE,
        block_size=BLOCK,
        max_model_len=1024,
        max_num_seqs=8,
        enforce_eager=True,
        gpu_memory_utilization=float(os.environ.get("KIVI_E2E_MEM", 0.4)),
    )
    print(f"E2E engine up: dtype={DTYPE} block_size={BLOCK}", flush=True)

    out = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=24))
    for i, req in enumerate(out):
        text = req.outputs[0].text.replace("\n", "\\n")
        print(f"E2E[{i}] {PROMPTS[i]!r} -> {text!r}", flush=True)

    tokens = sum(len(r.outputs[0].token_ids) for r in out)
    print(f"E2E RESULT: dtype={DTYPE} requests={len(out)} tokens={tokens}", flush=True)
    return 0 if tokens > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
