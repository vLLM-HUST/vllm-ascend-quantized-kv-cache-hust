# SPDX-License-Identifier: Apache-2.0
"""Real-host check: does ``--kv-cache-dtype`` actually select the plugin impl?

Every other dispatch test runs against a stubbed host, which proves the shape of
the table but not that it composes with the live ``AscendAttentionBackend``.
Run this inside the host venv (no NPU needed) from a clean checkout of this
repo::

    python scripts/probe_host_dispatch.py

Only the vllm-config context and the context-parallel flags are stubbed; both
read process-global state that a bare script has not set up. Everything else --
the host backend, its impl, and the plugin's composed classes -- is real.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

# vllm-ascend-hust 17ed0571d cannot import attention_v1 before vllm_ascend.ops
# (partially initialized vllm_ascend.device.device_op).
import vllm_ascend.ops  # noqa: E402,F401  isort:skip
import vllm_ascend.attention.utils as attention_utils  # noqa: E402  isort:skip
from vllm.config import (  # noqa: E402
    CacheConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm_ascend.attention.attention_v1 import (  # noqa: E402
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
)

from vllm_ascend_quantized_kv_cache.adapters.vllm_ascend_hust.backend import (  # noqa: E402
    install_kv_impl_dispatch,
)

# cache dtype literal -> (impl class name, mixin it must carry)
EXPECTED: dict[str, tuple[str, str | None]] = {
    "auto": (AscendAttentionBackendImpl.__name__, None),
    "int8": ("AscendInt8KvAttentionImpl", "AscendInt8AttentionBackendMixin"),
    "kivi_int4": (
        "AscendKiviInt4KvAttentionImpl",
        "AscendKiviInt4AttentionBackendMixin",
    ),
}
# other quantized literals the plugin must delegate rather than claim
DELEGATED = ("fp8", "fp8_e4m3", "float16")


def select(cache_dtype: str, *, context_parallel: bool = False) -> type:
    """Ask the live host factory which impl class a dtype literal resolves to.

    The context-parallel flags are *not* stubbed here: this host revision has no
    ``enable_cp``, so the guard has to answer from the real ``enable_dcp`` /
    ``enable_pcp`` pair. ``enable_dcp`` is lru_cached, hence the cache_clear.
    """
    config = VllmConfig(cache_config=CacheConfig(), parallel_config=ParallelConfig())
    # The host's cache_dtype is a validated Literal that does not carry the
    # plugin's literals yet, so the field is set after construction. See
    # docs/int4-host-integration.md for that host-side edit.
    object.__setattr__(config.cache_config, "cache_dtype", cache_dtype)
    if context_parallel:
        object.__setattr__(config.parallel_config, "decode_context_parallel_size", 2)
    attention_utils.enable_dcp.cache_clear()
    with set_current_vllm_config(config):
        return AscendAttentionBackend.get_impl_cls()


def main() -> int:
    failures: list[str] = []
    helpers = [
        n
        for n in ("enable_cp", "enable_dcp", "enable_pcp")
        if hasattr(attention_utils, n)
    ]
    print(f"host CP helpers: {', '.join(helpers) or 'none'}")
    if "enable_cp" in helpers:
        print("note: this host still has enable_cp, so the split path is not exercised")
    # the host's own factory needs a vllm config context (enable_dcp reads it)
    print(f"host before: {select('auto').__name__}")

    backend = install_kv_impl_dispatch()
    if backend is not AscendAttentionBackend:
        failures.append(f"install() returned {backend}, not the host backend")

    for cache_dtype, (expected, mixin) in EXPECTED.items():
        impl = select(cache_dtype)
        ok = impl.__name__ == expected
        print(f"{cache_dtype:<10} -> {impl.__name__:<30} {'ok' if ok else '?'}")
        if not ok:
            failures.append(f"{cache_dtype} selected {impl.__name__}, want {expected}")
            continue
        if mixin is None:
            continue
        if mixin not in [b.__name__ for b in impl.__mro__]:
            failures.append(f"{cache_dtype} impl lost its {mixin}")
        bound = AscendAttentionBackendImpl.__name__
        if bound not in [b.__name__ for b in impl.__mro__[1:4]]:
            failures.append(f"{cache_dtype} impl is not bound to the live host impl")

    for cache_dtype in DELEGATED:
        impl = select(cache_dtype)
        print(f"{cache_dtype:<10} -> {impl.__name__} (delegated)")
        if impl is not AscendAttentionBackendImpl:
            failures.append(f"{cache_dtype} should stay on the host impl")

    try:
        select("kivi_int4", context_parallel=True)
    except NotImplementedError as exc:
        print(f"CP guard: {exc}")
    else:
        failures.append("context parallel INT4 selection was not rejected")

    # repeated plugin loading (as a second entry-point activation would do)
    install_kv_impl_dispatch()
    if select("kivi_int4").__name__ != EXPECTED["kivi_int4"][0]:
        failures.append("repeated install changed the INT4 selection")

    print("RESULT:", "PASS" if not failures else f"FAIL {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
