# SPDX-License-Identifier: Apache-2.0
"""横切层：宿主名单与探测（hosts）、设备能力探测（runtime）、统一激活管线（activation）。

跨层共享、自身无领域语义的部分收在这里；量化方法的模型与实现见
``methods/``。

调用方契约：**任何进程都可导入本层**（零重依赖——torch / vllm /
triton 都不会被拉起）；宿主探测与激活守卫（``detect_host`` /
``require_host_stack``）是"vllm-hust 与 vllm-ascend-hust 进程各自能
调什么"的权威出处。
"""

from .activation import activate
from .hosts import (
    ALL_HOSTS,
    VLLM_ASCEND_HUST,
    VLLM_HUST,
    detect_host,
    require_host_stack,
)
from .runtime import npu_available, require_npu

__all__ = [
    "ALL_HOSTS",
    "VLLM_ASCEND_HUST",
    "VLLM_HUST",
    "activate",
    "detect_host",
    "npu_available",
    "require_host_stack",
    "require_npu",
]
