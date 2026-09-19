# SPDX-License-Identifier: Apache-2.0
"""Focused probe: where do the NaNs in the KIVI key gather output come from?"""

import sys

sys.path.insert(0, "src")

import torch  # noqa: E402

from vllm_ascend_quantized_kv_cache.methods.base import MethodConfig  # noqa: E402
from vllm_ascend_quantized_kv_cache.methods.kivi_int4.semantics import (  # noqa: E402
    KiviInt4Semantics,
)
from vllm_ascend_quantized_kv_cache.ops.kivi_gather import (  # noqa: E402
    kivi_dequant_gather_cache,
)
from vllm_ascend_quantized_kv_cache.ops.triton.kivi_pack import (  # noqa: E402
    kivi_pack_key_cache,
)

DEV = "npu:0"
HEAD, KVH, GROUP, BLOCK, NB = 32, 2, 8, 16, 4
T = NB * BLOCK

torch.manual_seed(0)
sem = KiviInt4Semantics(
    MethodConfig(head_size=HEAD, num_kv_heads=KVH, group_size=GROUP)
)

key = torch.randn(T, KVH, HEAD)
key_n = key.to(DEV).contiguous()
slots = torch.arange(T, dtype=torch.long, device=DEV)

k_quant = torch.zeros(NB, KVH, HEAD, BLOCK // 8, dtype=torch.int32, device=DEV)
k_scale = torch.zeros(NB, KVH, HEAD, BLOCK // GROUP, dtype=torch.float32, device=DEV)
k_mn = torch.zeros(NB, KVH, HEAD, BLOCK // GROUP, dtype=torch.float32, device=DEV)

kivi_pack_key_cache(key_n, slots, k_quant, k_scale, k_mn, GROUP)
torch.npu.synchronize()

kq, ks, km = k_quant.cpu(), k_scale.cpu(), k_mn.cpu()
print(
    "pack outputs finite:",
    "quant",
    bool(kq.isfinite().all()),
    "scale",
    bool(ks.isfinite().all()),
    "mn",
    bool(km.isfinite().all()),
)
print(
    "scale range:",
    ks.min().item(),
    ks.max().item(),
    " mn range:",
    km.min().item(),
    km.max().item(),
)

# expected scale/mn from the reference grouping, for group 0 of block 0
grp = key.view(NB, BLOCK // GROUP, GROUP, KVH, HEAD)
ref_scale = (grp.amax(2) - grp.amin(2)) / 15  # [NB, groups, KVH, HEAD]
ref_mn = grp.amin(2)
# pack cache layout: [NB, KVH, HEAD, groups]
print("scale layout check (block0, head0, dim0, all groups):")
print("  packed:", ks[0, 0, 0, :].tolist())
print("  ref:   ", ref_scale[0, :, 0, 0].tolist())
print("mn layout check:")
print("  packed:", km[0, 0, 0, :].tolist())
print("  ref:   ", ref_mn[0, :, 0, 0].tolist())

# manual dequant from the packed cache
q = sem.unpack_int4(kq).flatten(-2)  # [NB, KVH, HEAD, BLOCK]
scale_r = ks.repeat_interleave(GROUP, dim=-1)
mn_r = km.repeat_interleave(GROUP, dim=-1)
manual = (
    (q.to(scale_r.dtype) * scale_r + mn_r).permute(0, 3, 1, 2).reshape(T, KVH, HEAD)
)
ref_k = sem.fake_quant_key(key)
print("manual vs fake_quant: max|diff| =", (manual - ref_k).abs().max().item())

block_table = torch.arange(NB, dtype=torch.long, device=DEV).view(1, NB)
k_out, _ = kivi_dequant_gather_cache(
    k_quant,
    k_scale,
    k_mn,
    torch.zeros(NB, BLOCK, KVH, HEAD // 8, dtype=torch.int32, device=DEV),
    torch.zeros(NB, BLOCK, KVH, HEAD // GROUP, dtype=torch.float32, device=DEV),
    torch.zeros(NB, BLOCK, KVH, HEAD // GROUP, dtype=torch.float32, device=DEV),
    block_table,
    torch.tensor([T], dtype=torch.long, device=DEV),
    torch.float32,
    GROUP,
)
torch.npu.synchronize()
kout = k_out.cpu()
nan_mask = torch.isnan(kout)
n_nan = int(nan_mask.sum())
print(f"kernel output: {n_nan} NaNs of {kout.numel()} elements")
if n_nan:
    nan_tokens = nan_mask.any(dim=(1, 2)).nonzero().flatten().tolist()
    print("tokens containing NaN:", nan_tokens)
    t0 = nan_tokens[0]
    per_head = nan_mask[t0].any(dim=1)  # [KVH]
    per_dim = nan_mask[t0].any(dim=0)  # [HEAD]
    print(f"token {t0}: NaN per head {per_head.tolist()}, per dim {per_dim.tolist()}")
    finite = kout[~nan_mask]
    finite_manual = manual[~nan_mask]
    print(
        "max|kernel-manual| on finite cells:",
        (finite - finite_manual).abs().max().item() if finite.numel() else "n/a",
    )
