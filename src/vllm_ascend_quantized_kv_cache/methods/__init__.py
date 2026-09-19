# SPDX-License-Identifier: Apache-2.0
"""量化方法层：方法模型（base / registry）+ 全部内置方法实现。

import 本包即把每个方法的"元数据"注册进注册表；此刻不会导入
torch / vllm / 任何设备模块（各方法的重模块全部惰性加载）。

调用方契约：**任何进程都可导入本层**（方法发现/契约解析完全宿主无
关）。语义对象（semantics.py，需 torch）CPU 可测；设备执行路径
（attention_backend.py，需 torch_npu）只应在 Ascend NPU
上被宿主 impl 触发。宿主插拔不在这里——走 adapters/ 或
``kv_methods.activate``。

新增一个方法：新建 methods/<名字>/（或单文件），构造 MethodSpec 并
register_method —— 不需要改动任何现有方法。
"""

from . import int8_dynamic, kivi_int4  # noqa: F401  (注册副作用)

__all__ = ["int8_dynamic", "kivi_int4"]
