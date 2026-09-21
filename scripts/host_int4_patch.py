# SPDX-License-Identifier: Apache-2.0
"""Apply (or revert) the kivi_int4 host edits listed in
``docs/int4-host-integration.md`` to two *copies* of the host repositories.

The plugin itself must not fake any of this at runtime; the host owns the
dtype literal, the storage dtype and the page size.  This script exists so the
same edits can be tried on a scratch copy of a host checkout, reviewed as a
diff, and handed to the host owners.  It never writes outside the two trees it
is pointed at, and every edit is an exact-anchor replacement that fails loudly
if the host has moved on.

    python scripts/host_int4_patch.py \
        --vllm /tmp/host4-int4/sandbox-vllm \
        --ascend /tmp/host4-int4/sandbox-ascend
    git -C /tmp/host4-int4/sandbox-vllm diff          # review
    python scripts/host_int4_patch.py ... --reverse   # undo

Verified against vllm-hust ``f18cf803c5`` and vllm-ascend-hust ``17ed0571d``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The plugin owns the byte layout inside each side's region, so the host only
# has to size it.  This formula must stay in sync with KiviByteCacheLayout in
# methods/kivi_int4/byte_cache.py (S = head_size/2 + 8*head_size/group_size).
HELPER = """
# --- KIVI INT4 (vllm-ascend-quantized-kv-cache plugin) ---------------------
# Bytes per token per kv head on ONE side (key or value): int4 payload plus
# one fp32 scale and one fp32 min per group, amortised over the group.
# Layout owner: vllm_ascend_quantized_kv_cache.methods.kivi_int4.byte_cache.
KIVI_INT4_GROUP_SIZE = int(__import__("os").environ.get("VLLM_KIVI_GROUP_SIZE", "128"))


def kivi_int4_bytes_per_token_head(head_size: int, group_size: int) -> int:
    return head_size // 2 + 8 * head_size // group_size

"""


NVFP4_LINE = (
    "    NVFP4_DS_MLA = 10  # opaque-bytes NVFP4 DS-MLA layouts (FlashMLA sparse)"
)
PLANE_LINE = (
    "        plane_size_bytes = spec.num_kv_heads * spec.head_size"
    " * get_dtype_size(spec.dtype)"
)
CLASS_DEF = (
    '@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")\n'
    "class AscendAttentionBackend(AttentionBackend):\n"
)


def edits(vllm: Path, ascend: Path) -> list[tuple[Path, str, str]]:
    cache = vllm / "vllm/config/cache.py"
    torch_utils = vllm / "vllm/utils/torch_utils.py"
    iface = vllm / "vllm/v1/kv_cache_interface.py"
    attn = ascend / "vllm_ascend/attention/attention_v1.py"
    runner = ascend / "vllm_ascend/worker/model_runner_v1.py"

    shape_call = (
        "                        kv_cache_shape = attn_backend.get_kv_cache_shape(\n"
        "                            num_blocks,\n"
        "                            current_kv_cache_spec.block_size,\n"
        "                            current_kv_cache_spec.num_kv_heads,\n"
        "                            current_kv_cache_spec.head_size,\n"
        "                        )\n"
    )
    shape_call_with_dtype = shape_call[:-2] + (
        "                            cache_dtype_str=self.vllm_config.cache_config."
        "cache_dtype,\n                        )\n"
    )
    hybrid_call = (
        "                        kv_cache_shape = attn_backend.get_kv_cache_shape(\n"
        "                            num_blocks * block_size_chunk,\n"
        "                            block_size,\n"
        "                            current_kv_cache_spec.num_kv_heads,\n"
        "                            current_kv_cache_spec.head_size,\n"
        "                        )\n"
    )
    hybrid_call_with_dtype = hybrid_call[:-2] + (
        "                            cache_dtype_str=self.vllm_config.cache_config."
        "cache_dtype,\n                        )\n"
    )

    return [
        # 1. CLI literal: without this the request dies while building
        #    CacheConfig, before any plugin code runs.
        (
            cache,
            '    "nvfp4",\n    "nvfp4_4over6",\n]',
            '    "nvfp4",\n    "nvfp4_4over6",\n    "kivi_int4",\n]',
        ),
        # 2. Storage dtype + "this is a quantized cache" for the string-based
        #    helpers in vllm.utils.torch_utils.  KIVI is deliberately NOT a
        #    per-token-head dtype: its scales are per group, plugin-side.
        (
            torch_utils,
            '    "int8": torch.int8,\n',
            '    "int8": torch.int8,\n    "kivi_int4": torch.uint8,\n',
        ),
        (
            torch_utils,
            '        or kv_cache_dtype.startswith("nvfp4")\n    )',
            '        or kv_cache_dtype.startswith("nvfp4")\n'
            '        or kv_cache_dtype == "kivi_int4"\n    )',
        ),
        # 3. Quant mode, so get_kv_quant_mode()/is_quantized_kv_cache() in
        #    vllm/v1/kv_cache_interface.py agree with the above.
        (
            iface,
            NVFP4_LINE,
            NVFP4_LINE + "\n    KIVI_INT4 = 11  # KIVI group int4 + fp32 scale/min",
        ),
        (
            iface,
            '    if kv_cache_dtype == "int4_per_token_head":',
            '    if kv_cache_dtype == "kivi_int4":\n'
            "        return KVQuantMode.KIVI_INT4\n"
            '    if kv_cache_dtype == "int4_per_token_head":',
        ),
        # 4. Byte-region shape and page size on the Ascend backend, plus the
        #    cache_dtype_str the generic reshape path never passed.  The helper
        #    goes above the class: a dedented statement inside a class body
        #    silently ends it, which lost every later method (measured).
        (
            attn,
            CLASS_DEF,
            HELPER + "\n\n" + CLASS_DEF,
        ),
        (
            attn,
            "    ) -> tuple[int, ...]:\n"
            "        return (2, num_blocks, block_size, num_kv_heads, head_size)\n",
            "    ) -> tuple[int, ...]:\n"
            '        if cache_dtype_str == "kivi_int4":\n'
            "            per_head = kivi_int4_bytes_per_token_head(\n"
            "                head_size, KIVI_INT4_GROUP_SIZE\n"
            "            )\n"
            "            return (2, num_blocks, block_size, num_kv_heads, per_head)\n"
            "        return (2, num_blocks, block_size, num_kv_heads, head_size)\n",
        ),
        (
            attn,
            "from vllm.v1.kv_cache_interface import "
            "AttentionSpec, CrossAttentionSpec\n",
            "from vllm.v1.kv_cache_interface import (\n"
            "    AttentionSpec,\n"
            "    CrossAttentionSpec,\n"
            "    KVQuantMode,\n"
            ")\n",
        ),
        (
            attn,
            PLANE_LINE,
            "        if spec.kv_quant_mode == KVQuantMode.KIVI_INT4:\n"
            "            per_head = kivi_int4_bytes_per_token_head(\n"
            "                spec.head_size, KIVI_INT4_GROUP_SIZE\n"
            "            )\n"
            "            return replace(\n"
            "                spec,\n"
            "                num_head_slots=2,\n"
            "                state_content_bytes=spec.num_kv_heads * per_head,\n"
            "            )\n" + PLANE_LINE,
        ),
        (runner, shape_call, shape_call_with_dtype),
        (runner, hybrid_call, hybrid_call_with_dtype),
    ]


def apply(path: Path, old: str, new: str, reverse: bool) -> None:
    text = path.read_text()
    src, dst = (new, old) if reverse else (old, new)
    hits = text.count(src)
    if hits != 1:
        raise SystemExit(
            f"{path}: anchor matched {hits} times, expected 1 -- the host has "
            f"moved on, re-measure before patching:\n{src[:160]}"
        )
    path.write_text(text.replace(src, dst))
    verb = "reverted" if reverse else "patched"
    print(f"  {verb} {path.name}: {src.strip()[:60]!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", required=True, type=Path, help="copy of vllm-hust")
    ap.add_argument(
        "--ascend", required=True, type=Path, help="copy of vllm-ascend-hust"
    )
    ap.add_argument("--reverse", action="store_true")
    args = ap.parse_args()

    for root in (args.vllm, args.ascend):
        if "/root/vllm/" in str(root.resolve()) or root.resolve() in (
            Path("/root/vllm/vllm-hust"),
            Path("/root/vllm/vllm-ascend-hust"),
        ):
            raise SystemExit(f"refusing to patch a live host checkout: {root}")

    pairs = edits(args.vllm, args.ascend)
    for path, _old, _new in pairs:
        if not path.is_file():
            raise SystemExit(f"{path} is not a file -- wrong checkout?")
    for path, old, new in reversed(pairs) if args.reverse else pairs:
        apply(path, old, new, args.reverse)
    print("RESULT: reverted" if args.reverse else "RESULT: host edits applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
