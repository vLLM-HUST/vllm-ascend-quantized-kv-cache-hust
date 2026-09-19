#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Re-check the KIVI INT4 port against the archived ascend#116 patches.

The INT4 kernels and state machine were rebuilt from
``provenance/legacy-patches/ascend-pr-116``; this script re-runs that
comparison so the fidelity claim in ``PROVENANCE.md`` stays a measurement
instead of a memory.  Run it before a 910B2 verification session:

    python scripts/check_int4_patch_parity.py

Exit status is non-zero when a load-bearing invariant from the patches is no
longer present in the ported code.  Kernel line coverage is reported but not
gated: the patches contain multi-line statement fragments, plus 0007-era
per-request ``cu_seq_lens`` indexing that 0008/0009 replaced with the
``token_block_ids`` / ``token_block_offsets`` scheme the ported kernel does
use (that replacement is itself gated below).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = ROOT / "provenance" / "legacy-patches" / "ascend-pr-116"
KERNEL_FILES = (
    "src/vllm_ascend_quantized_kv_cache/ops/triton/kivi_pack.py",
    "src/vllm_ascend_quantized_kv_cache/ops/triton/kivi_gather_experimental.py",
    "src/vllm_ascend_quantized_kv_cache/ops/kivi_gather.py",
    "src/vllm_ascend_quantized_kv_cache/ops/kivi_layout.py",
)
PORTED_FILES = KERNEL_FILES + (
    "src/vllm_ascend_quantized_kv_cache/methods/kivi_int4/semantics.py",
    "src/vllm_ascend_quantized_kv_cache/methods/kivi_int4/geometry.py",
    "src/vllm_ascend_quantized_kv_cache/methods/kivi_int4/attention_backend.py",
)

#: invariants the pack/gather kernels and the residual state machine rely on
IN_VARIANTS: dict[str, str] = {
    "key slot alignment check": r"aligned token groups|block_offset % ",
    "group_size % 8 packing gate": r"group_size % 8",
    "int32 word with 8 int4 lanes": r"lane \* 4|4 \* lane|// 8",
    "value head-dim grouping": r"head_size % group_size|num_groups = ",
    "cumulative kv lengths": r"cumsum",
    "residual window divisibility": r"residual_length % ",
    "asymmetric min/max scale": r"clamp\(min=1e-6\)|maximum\(\(mx - mn\)",
    "half-up rounding in the kernel": r"floor\(",
    "int4 mask on unpack": r"& 0xF|&0xF",
    # 0008/0009 replaced the 0007 per-request cu_seq_lens loop with precomputed
    # per-token block ids/offsets; the ported kernel must use the new scheme
    "gather indexing scheme (0008/0009)": r"token_block_ids_ptr",
}


def _norm(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def _final_kernel_lines() -> dict[str, str]:
    """Replay patch add/remove events in order; the last event wins."""
    final: dict[str, str] = {}
    for patch in sorted(PATCH_DIR.glob("*.patch")):
        text = patch.read_text(errors="replace")
        for line in text.splitlines():
            if (line.startswith("+") or line.startswith("-")) and "tl." in line:
                body = _norm(line[1:])
                if line.startswith("+"):
                    final[body] = patch.name
                else:
                    final.pop(body, None)
    return final


def main() -> int:
    if not PATCH_DIR.is_dir():
        print(f"FAIL: patch archive not found at {PATCH_DIR}")
        return 1

    ported = {
        _norm(line)
        for path in KERNEL_FILES
        for line in (ROOT / path).read_text().splitlines()
    }
    final = _final_kernel_lines()
    missing = sorted((line, src) for line, src in final.items() if line not in ported)
    print(
        f"kernel lines: {len(final) - len(missing)}/{len(final)} final-state "
        f"triton lines match verbatim ({len(missing)} informational misses)"
    )
    for line, src in missing:
        print(f"  ~ [{src}] {line[:100]}")

    haystack = "\n".join((ROOT / path).read_text() for path in PORTED_FILES)
    failures = []
    for label, pattern in IN_VARIANTS.items():
        ok = re.search(pattern, haystack) is not None
        if not ok:
            failures.append(label)
        print(f"  [{'ok' if ok else 'MISSING'}] {label}")
    if failures:
        print(f"FAIL: {len(failures)} invariant(s) lost: {', '.join(failures)}")
        return 1
    print("PASS: INT4 port still carries every patch-level invariant")
    return 0


if __name__ == "__main__":
    sys.exit(main())
