# SPDX-License-Identifier: Apache-2.0
"""Installed-artifact check: does the wheel register itself in a real vLLM run?

Everything else in ``scripts/`` runs from a source checkout with
``PYTHONPATH=src``. That proves the code works but not the delivery contract:
vLLM discovers this plugin through the ``vllm.general_plugins`` entry point of
the *installed distribution*, and the dispatcher has to be installed over the
host backend from that path, in a process that never saw the repository.

Install the built wheel somewhere that is not the source tree, then run the
script with that directory ahead of ``sys.path`` (the script refuses any import
that resolves inside a repository ``src``)::

    python -m build --wheel
    python -m pip install --target /tmp/kivi-probe --no-deps dist/*.whl
    PYTHONPATH=/tmp/kivi-probe python scripts/probe_installed_plugin.py

Needs the host venv (vllm-hust + vllm-ascend-hust installed); no NPU required.
"""

from __future__ import annotations

import importlib
import importlib.metadata as im
import os
import sys

ENTRY_POINT = "vllm-ascend-quantized-kv-cache"
PACKAGE = "vllm_ascend_quantized_kv_cache"
EXPECTED = {
    "auto": "AscendAttentionBackendImpl",
    "int8": "AscendInt8KvAttentionImpl",
    "kivi_int4": "AscendKiviInt4KvAttentionImpl",
}


def main() -> int:
    failures: list[str] = []

    # 1) the installed distribution advertises the entry point vLLM looks for
    entry_points = [
        ep
        for ep in im.entry_points(group="vllm.general_plugins")
        if ep.name == ENTRY_POINT
    ]
    if not entry_points:
        print("no entry point registered; install the wheel first")
        return 1
    entry_point = entry_points[0]
    print(f"entry point: {entry_point.value}  (dist {ENTRY_POINT})")

    # 2) vLLM's own loader must bring the dispatcher up, unprompted
    from vllm.plugins import load_general_plugins

    load_general_plugins()

    module = importlib.import_module(PACKAGE)
    loaded_from = os.path.realpath(str(getattr(module, "__file__", "")))
    print(f"package loaded from: {loaded_from}")
    if os.sep + "src" + os.sep in loaded_from:
        failures.append(f"{PACKAGE} came from a source tree, not the wheel")

    import vllm_ascend.ops  # noqa: F401  # host import-order requirement
    from vllm.config import (
        CacheConfig,
        ParallelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionBackend,
        AscendAttentionBackendImpl,
    )

    marker = "_quantized_kv_plugin_dispatch_installed"
    if not getattr(AscendAttentionBackend, marker, False):
        failures.append("the host backend has no quantized-KV dispatcher installed")

    # 3) selection now happens through the host factory the plugin patched
    def select(cache_dtype: str) -> type:
        config = VllmConfig(
            cache_config=CacheConfig(), parallel_config=ParallelConfig()
        )
        # the host's cache_dtype Literal does not carry the plugin's literals
        # yet (docs/int4-host-integration.md), so bypass that one validation
        object.__setattr__(config.cache_config, "cache_dtype", cache_dtype)
        with set_current_vllm_config(config):
            return AscendAttentionBackend.get_impl_cls()

    for cache_dtype, expected in EXPECTED.items():
        impl = select(cache_dtype)
        ok = impl.__name__ == expected
        print(f"{cache_dtype:<10} -> {impl.__name__:<30} {'ok' if ok else '?'}")
        if not ok:
            failures.append(f"{cache_dtype} selected {impl.__name__}")
            continue
        if cache_dtype == "auto":
            continue
        module_of_impl = importlib.import_module(impl.__module__)
        origin = os.path.realpath(str(getattr(module_of_impl, "__file__", "")))
        if not origin.startswith(loaded_from.rsplit(os.sep, 1)[0]):
            failures.append(f"{cache_dtype} impl came from {origin}")
        if not issubclass(impl, AscendAttentionBackendImpl):
            failures.append(f"{cache_dtype} impl is not bound to the host impl")

    print("RESULT:", "PASS" if not failures else f"FAIL {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
