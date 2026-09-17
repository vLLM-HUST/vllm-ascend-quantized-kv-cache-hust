# SPDX-License-Identifier: Apache-2.0
"""vllm-hust 宿主适配器。

挂载机制（零宿主改动）：经宿主 attention 注册表，把全限定 backend
类路径注册到 ``AttentionBackendEnum.CUSTOM``，为方法协商一个既有的
CacheDType 字面量，引擎以 ``--attention-backend CUSTOM`` 启动。设备
执行始终走 Ascend NPU 内核，非 NPU 环境 fail-closed。

注意：backend 类刻意不在这里再导出——它经 ``backend.__getattr__``
在 vllm 进程解析注册类路径时才惰性构建；提前 import 会把 vllm 拽进来。
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
