# SPDX-License-Identifier: Apache-2.0
"""Measure which INT4 geometries the triton pack kernels really compile.

``geometry.validate_kivi_geometry`` accepts anything where the group size divides
head/block/residual sizes, and the CPU tests only use small legal shapes.  On
910B2 that turns out to be wider than what the pack kernels can execute: the
bishengir backend runs out of unified buffer and rejects the kernel, e.g. at
``head_size=128, block_size=256``::

    kivi_pack.py:135: error: ub overflow, requires 1655040 bits while
    1572864 bits available!

That is a fact the host has to know before it picks ``block_size``, and the
plugin should not pretend the geometry is fine.  This probe runs the real key
and value pack kernels over a grid of legal shapes and prints which ones the
compiler accepts, so the limit is measured rather than guessed.

    python scripts/npu_probe_kivi_geometry.py
"""

from __future__ import annotations

import re
import sys

sys.path.insert(0, "src")

import torch  # noqa: E402

from vllm_ascend_quantized_kv_cache.ops.triton.kivi_pack import (  # noqa: E402
    kivi_pack_key_cache,
    kivi_pack_value_cache,
)

DEV = "npu:0"
KVH = 2
# (head_size, group_size, block_size): every combination satisfies the plugin's
# own divisibility rules; only the buffer budget differs
SHAPES = [
    (32, 8, 16),
    (64, 32, 32),
    (64, 32, 64),
    (64, 32, 128),
    (64, 32, 256),
    (64, 32, 512),
    (128, 32, 512),
    (128, 64, 256),
    (128, 128, 128),
    (128, 128, 256),
    (256, 128, 128),
    (256, 128, 256),
    (64, 64, 256),
    (64, 64, 512),
    # an earlier candidate rule -- group_size * block_size <= 16384 -- was
    # falsified by (64, 64, 512) compiling fine, so this table is a measurement,
    # not a formula; the plugin must not promise more than it shows
    (256, 64, 256),
]


def run(shape: tuple[int, int, int]) -> tuple[str, str]:
    head, group, block = shape
    tokens = 2 * block  # two whole cache blocks, so slots stay inside one each
    torch.manual_seed(0)
    value = torch.randn(tokens, KVH, head, device=DEV, dtype=torch.float16)
    key = torch.randn_like(value)
    slots = torch.arange(tokens, dtype=torch.long, device=DEV)
    k_quant = torch.zeros(2, KVH, head, block // 8, dtype=torch.int32, device=DEV)
    k_scale = torch.zeros(2, KVH, head, block // group, dtype=torch.float32, device=DEV)
    k_mn = torch.zeros_like(k_scale)
    v_quant = torch.zeros(2, block, KVH, head // 8, dtype=torch.int32, device=DEV)
    v_scale = torch.zeros(2, block, KVH, head // group, dtype=torch.float32, device=DEV)
    v_mn = torch.zeros_like(v_scale)
    try:
        kivi_pack_key_cache(key, slots, k_quant, k_scale, k_mn, group)
        kivi_pack_value_cache(value, slots, v_quant, v_scale, v_mn, group)
        torch.npu.synchronize()
    except Exception as exc:  # triton/bishengir raise several unrelated types
        text = str(exc)
        wanted = re.search(r"requires (\d+) bits while (\d+) bits available", text)
        if wanted:
            need, have = int(wanted.group(1)), int(wanted.group(2))
            return (
                "FAIL",
                f"ub needs {need / 8 / 1024:.0f}KiB > {have / 8 / 1024:.0f}KiB",
            )
        first = next((line for line in text.splitlines() if line.strip()), "")
        return "FAIL", f"{type(exc).__name__}: {first[:70]}"
    bad = not bool(k_quant.abs().sum() and v_quant.abs().sum())
    return ("FAIL", "wrote nothing") if bad else ("ok", "")


def main() -> int:
    print(f"device {torch.npu.current_device()}", flush=True)
    print(f"{'head':>5} {'group':>6} {'block':>6} {'blk*head':>9}  result", flush=True)
    failures = 0
    for head, group, block in SHAPES:
        status, detail = run((head, group, block))
        failures += status == "FAIL"
        print(
            f"{head:>5} {group:>6} {block:>6} {block * head:>9}  {status:<4} {detail}",
            flush=True,
        )
    print(
        f"RESULT: {len(SHAPES) - failures}/{len(SHAPES)} shapes compile "
        "(informational: the shipped default must be among them)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
