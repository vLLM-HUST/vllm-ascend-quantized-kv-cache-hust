# SPDX-License-Identifier: Apache-2.0
"""Decisive probe for the KIVI key gather NaN: sentinel-prefilled output.

Calls the key dequant-gather kernel directly with the output prefilled to
777.0. Cells still 777 afterwards were never stored (store-mask bug); cells
with other wrong values were stored but computed wrong. Compiles the kernel
at tile sizes 16 and 32 to test whether the 0009 tile shrink is the trigger.
"""

import sys

sys.path.insert(0, "src")

import torch  # noqa: E402
import triton  # noqa: E402

from vllm_ascend_quantized_kv_cache.methods.base import MethodConfig  # noqa: E402
from vllm_ascend_quantized_kv_cache.methods.kivi_int4.semantics import (  # noqa: E402
    KiviInt4Semantics,
)
from vllm_ascend_quantized_kv_cache.ops.triton.kivi_gather_experimental import (  # noqa: E402
    _kivi_dequant_gather_key_cache_kernel,
)
from vllm_ascend_quantized_kv_cache.ops.triton.kivi_pack import (  # noqa: E402
    kivi_pack_key_cache,
)

DEV = "npu:0"
KVH, GROUP, BLOCK, NB = 2, 8, 16, 4


def run_case(head_size: int, block_dim: int) -> None:
    T = NB * BLOCK
    torch.manual_seed(0)
    sem = KiviInt4Semantics(
        MethodConfig(head_size=head_size, num_kv_heads=KVH, group_size=GROUP)
    )
    key = torch.randn(T, KVH, head_size)
    key_n = key.to(DEV).contiguous()
    slots = torch.arange(T, dtype=torch.long, device=DEV)

    k_quant = torch.zeros(NB, KVH, head_size, BLOCK // 8, dtype=torch.int32, device=DEV)
    k_scale = torch.zeros(
        NB, KVH, head_size, BLOCK // GROUP, dtype=torch.float32, device=DEV
    )
    k_mn = torch.zeros(
        NB, KVH, head_size, BLOCK // GROUP, dtype=torch.float32, device=DEV
    )
    kivi_pack_key_cache(key_n, slots, k_quant, k_scale, k_mn, GROUP)
    torch.npu.synchronize()

    # manual reference from the packed cache
    kq, ks, km = k_quant.cpu(), k_scale.cpu(), k_mn.cpu()
    q = sem.unpack_int4(kq).flatten(-2)
    scale = ks.repeat_interleave(GROUP, dim=-1)
    mn = km.repeat_interleave(GROUP, dim=-1)
    manual = (
        (q.to(scale.dtype) * scale + mn).permute(0, 3, 1, 2).reshape(T, KVH, head_size)
    )

    block_table = torch.arange(NB, dtype=torch.long, device=DEV).view(1, NB)
    cu = torch.tensor([0, T], dtype=torch.long, device=DEV)
    token_idx = torch.arange(T, dtype=torch.long, device=DEV)
    req_of_token = torch.searchsorted(cu[1:], token_idx, right=True)
    local_pos = token_idx - cu[req_of_token]
    logical = torch.div(local_pos, BLOCK, rounding_mode="floor")
    token_block_ids = block_table[req_of_token, logical].contiguous()
    token_block_offsets = (local_pos - logical * BLOCK).contiguous()
    out = torch.full((T, KVH, head_size), 777.0, device=DEV)

    grid = (
        triton.cdiv(T, block_dim),
        KVH,
        triton.cdiv(head_size, block_dim),
    )
    _kivi_dequant_gather_key_cache_kernel[grid](
        k_quant,
        k_scale,
        k_mn,
        token_block_ids,
        token_block_offsets,
        out,
        T,
        BLOCK,
        KVH,
        head_size,
        GROUP,
        block_dim,
    )
    torch.npu.synchronize()

    cpu_out = out.cpu()
    unstored = cpu_out == 777.0
    wrong = (~unstored) & ((cpu_out - manual).abs() > 1e-4)
    n_un = int(unstored.sum())
    n_wrong = int(wrong.sum())
    print(
        f"[head={head_size} tile={block_dim}] unstored={n_un} wrong-stored={n_wrong} "
        f"total={cpu_out.numel()}"
    )
    if n_un:
        pat = unstored.cpu().any(dim=1)  # [T, head]
        dims = pat.any(dim=0).nonzero().flatten().tolist()
        toks = pat.any(dim=1).nonzero().flatten().tolist()
        print(f"   unstored dims: {dims}")
        print(f"   unstored tokens (first 12): {toks[:12]}")
    if n_wrong:
        loc = (wrong & (cpu_out.abs() < 1e29)).nonzero()
        if loc.numel():
            t0, h0, d0 = loc[0].tolist()
            print(
                f"   sample wrong: out={cpu_out[t0, h0, d0]:.4f} manual={manual[t0, h0, d0]:.4f}"
            )
        bad_out = cpu_out[wrong]
        print(
            f"   wrong values finite: {bool(bad_out.isfinite().all())}, "
            f"min={bad_out.min().item():.3e}, max={bad_out.max().item():.3e}"
        )


for head_size in (32, 128):
    for block_dim in (16, 32):
        try:
            run_case(head_size, block_dim)
        except Exception:  # noqa: BLE001
            import traceback

            print(f"[head={head_size} tile={block_dim}] EXCEPTION:")
            traceback.print_exc()
