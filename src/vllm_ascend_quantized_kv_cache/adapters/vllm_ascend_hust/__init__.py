# SPDX-License-Identifier: Apache-2.0
"""vllm-ascend-hust 宿主适配器。

挂载机制（零宿主改动）：

1. 方法的 scheme 类以带命名空间的 quant_type 键注册进宿主的
   @register_scheme 注册表；
2. attention 层经 checkpoint 的 fa_quant_type 键（ModelSlim 路径）
   命中该 scheme，或由 bootstrap 在每个进程里注册；
3. 有状态方法由 scheme 的 create_weights 把 layer impl 换成 mixin
   支撑的 AscendAttentionBackendImpl 子类（在树 C8 先例）。
"""

from .attention import apply_impl_surgery, build_impl_cls, supported_impl_methods
from .register import AscendHustAdapter
from .scheme import build_scheme_cls, packed_semantics_for

__all__ = [
    "AscendHustAdapter",
    "apply_impl_surgery",
    "build_impl_cls",
    "build_scheme_cls",
    "packed_semantics_for",
    "supported_impl_methods",
]
