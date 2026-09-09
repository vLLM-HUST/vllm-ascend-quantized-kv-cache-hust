# SPDX-License-Identifier: Apache-2.0
"""vllm-hust host adapter.

Attach mechanism (no host changes required): register a fully qualified
backend class path under ``AttentionBackendEnum.CUSTOM`` via the host
attention registry, negotiate an existing ``CacheDType`` literal for the
solution, and run the engine with ``--attention-backend CUSTOM``. Device
execution always targets Ascend NPU kernels and fails closed elsewhere.

Note: the backend class itself is *not* re-exported here on purpose — it
is built lazily through ``backend.__getattr__`` when a vllm process
resolves the registered class path, and importing it eagerly would pull
in vllm.
"""

from .register import (
    BACKEND_CLASS_PATH,
    DTYPE_LITERAL_MAP,
    VllmHustAdapter,
    map_cache_dtype,
)

__all__ = [
    "BACKEND_CLASS_PATH",
    "DTYPE_LITERAL_MAP",
    "VllmHustAdapter",
    "map_cache_dtype",
]
