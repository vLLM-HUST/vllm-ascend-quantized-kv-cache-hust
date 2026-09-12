# SPDX-License-Identifier: Apache-2.0
"""设备执行层（ops）：torch 张量算子与 triton-ascend 内核。

调用方契约：**本层只被 methods 的 attention mixin / 语义路径与本库的
NPU 诊断脚本（scripts/npu_*.py）调用**；宿主进程与方法发现路径不应
直接 import 本层。torch / triton / torch_npu 在各子模块内部惰性加载，
缺设备栈时 fail-closed 并给出精确报错（经 core.runtime 的守卫）。

- ``int8_ops`` / ``kivi_gather`` / ``kivi_layout``：纯 torch（CPU 可测）。
- ``ops/triton``：triton-ascend 内核，仅 Ascend NPU 在线路径。
"""

__all__: list[str] = []
