# SPDX-License-Identifier: Apache-2.0
"""NPU smoke test: KIVI triton-ascend kernels vs CPU reference math.

Runs on an Ascend 910B device:
  1. quantize-and-pack a full key/value window into the paged int4 cache,
  2. gather + dequantize it back,
  3. compare against two CPU references:
     a. KiviInt4Semantics.fake_quant_* (end-to-end quantization math),
     b. manual unpack of the packed cache via semantics.unpack_int4 +
        dequant_*_blocks (cache-layout math).

A mismatch against (a) but agreement with (b) means the pack kernel's
quantization semantics differ from the fake-quant reference; disagreement
with both points at the gather kernel.
"""

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
    kivi_pack_value_cache,
)

DEV = "npu:0"
HEAD, KVH, GROUP, BLOCK, NB = 32, 2, 8, 16, 4
T = NB * BLOCK

torch.manual_seed(0)
cfg = MethodConfig(
    head_size=HEAD,
    num_kv_heads=KVH,
    group_size=GROUP,
    residual_length=GROUP,
    block_size=BLOCK,
)
sem = KiviInt4Semantics(cfg)

key = torch.randn(T, KVH, HEAD)
value = torch.randn(T, KVH, HEAD)
key_n = key.to(DEV).contiguous()
value_n = value.to(DEV).contiguous()
slots = torch.arange(T, dtype=torch.long, device=DEV)

k_quant = torch.zeros(NB, KVH, HEAD, BLOCK // 8, dtype=torch.int32, device=DEV)
k_scale = torch.zeros(NB, KVH, HEAD, BLOCK // GROUP, dtype=torch.float32, device=DEV)
k_mn = torch.zeros(NB, KVH, HEAD, BLOCK // GROUP, dtype=torch.float32, device=DEV)
v_quant = torch.zeros(NB, BLOCK, KVH, HEAD // 8, dtype=torch.int32, device=DEV)
v_scale = torch.zeros(NB, BLOCK, KVH, HEAD // GROUP, dtype=torch.float32, device=DEV)
v_mn = torch.zeros(NB, BLOCK, KVH, HEAD // GROUP, dtype=torch.float32, device=DEV)

print("== pack key cache ==", flush=True)
kivi_pack_key_cache(key_n, slots, k_quant, k_scale, k_mn, GROUP)
torch.npu.synchronize()
assert int(k_quant.abs().sum()) > 0, "key quant cache untouched"

print("== pack value cache ==", flush=True)
kivi_pack_value_cache(value_n, slots, v_quant, v_scale, v_mn, GROUP)
torch.npu.synchronize()
assert int(v_quant.abs().sum()) > 0, "value quant cache untouched"

print("== dequant gather ==", flush=True)
block_table = torch.arange(NB, dtype=torch.long, device=DEV).view(1, NB)
k_out, v_out = kivi_dequant_gather_cache(
    k_quant,
    k_scale,
    k_mn,
    v_quant,
    v_scale,
    v_mn,
    block_table,
    torch.tensor([T], dtype=torch.long, device=DEV),
    torch.float32,
    GROUP,
)
torch.npu.synchronize()

failures = []

# ---- value path ----
ref_v = sem.fake_quant_value(value)
dv = (v_out.cpu() - ref_v).abs()
print(f"value max|diff|={dv.max():.6f}")
if dv.max() == 0:
    print("value path: EXACT match vs fake_quant reference")
else:
    failures.append("value")

# ---- key path: reference (a) fake_quant ----
ref_k = sem.fake_quant_key(key)
dk = (k_out.cpu() - ref_k).abs()
k_step = (
    key.view(-1, GROUP, KVH, HEAD).amax(1) - key.view(-1, GROUP, KVH, HEAD).amin(1)
) / 15
k_step_full = k_step.repeat_interleave(GROUP, dim=0)  # [T, KVH, HEAD]
print(
    f"key vs fake_quant: max|diff|={dk.max():.6f} (one step <= {k_step_full.max():.6f})"
)
bad = dk > k_step_full.unsqueeze(0)
if bool(bad.any()):
    bad_tokens = bad.any(dim=(1, 2)).nonzero().flatten().tolist()
    print(
        f"key: {int(bad.sum())} elements off; "
        f"offending tokens (first 20): {bad_tokens[:20]}"
    )
    t0, h0, d0 = (dk == dk.max()).nonzero()[0].tolist()
    print(
        f"  worst at token={t0} head={h0} dim={d0}: got {k_out.cpu()[t0, h0, d0]:.4f}, "
        f"ref {ref_k[t0, h0, d0]:.4f}, raw {key[t0, h0, d0]:.4f}"
    )

# ---- key path: reference (b) manual cache unpack ----
kq, ks, km = k_quant.cpu(), k_scale.cpu(), k_mn.cpu()
q = sem.unpack_int4(kq).flatten(-2)  # [NB, KVH, HEAD, BLOCK]
scale = ks.repeat_interleave(GROUP, dim=-1)  # [NB, KVH, HEAD, BLOCK]
mn = km.repeat_interleave(GROUP, dim=-1)
deq = q.to(scale.dtype) * scale + mn  # [NB, KVH, HEAD, BLOCK]
manual = deq.permute(0, 3, 1, 2).reshape(T, KVH, HEAD)
dm = (k_out.cpu() - manual).abs()
print(f"key vs manual-unpack: max|diff|={dm.max():.6f}")
if dm.max() <= 1e-4:
    print(
        "gather kernel MATCHES the packed cache; "
        "mismatch is pack-vs-fakequant semantics"
    )
else:
    print("gather kernel DISAGREES with the packed cache -> gather-side issue")
    failures.append("gather")
    bad2 = (dm > 1e-4).any(dim=(1, 2)).nonzero().flatten().tolist()
    print(f"  offending tokens (first 20): {bad2[:20]}")

if not failures and bool(bad.any()):
    print("diagnosis: pack kernel semantics differ from fake_quant reference")
    print("  k_scale sample [block0,h0,0,:]:", ks[0, 0, 0, :].tolist())
    print("  k_mn    sample [block0,h0,0,:]:", km[0, 0, 0, :].tolist())
    grp = key.view(-1, GROUP, KVH, HEAD)
    print(
        "  expected scale [tok0-7,h0,dim0]:",
        ((grp[:, :, 0, 0].amax(1) - grp[:, :, 0, 0].amin(1)) / 15)[:4].tolist(),
    )
    print("  expected mn     [tok0-7,h0,dim0]:", grp[:, :, 0, 0].amin(1)[:4].tolist())

verdict = "PASS" if not failures else f"FAIL: {failures}"
print("RESULT:", verdict)
sys.exit(0 if not failures else 1)
