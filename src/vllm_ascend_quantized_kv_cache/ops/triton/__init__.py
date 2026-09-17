# SPDX-License-Identifier: Apache-2.0
"""triton-ascend 内核（仅 Ascend NPU）。

- ``kivi_pack``：KIVI 打包内核，**在线路径**（910B2 逐位验证）。
- ``kivi_gather_experimental``：融合 dequant-gather 内核，triton-ascend
  3.5 上误编译，**保留不路由**（实际走 ``ops.kivi_gather`` 的纯 torch
  实现）；triton-ascend 修复后用 ``scripts/npu_probe_kivi_dim.py`` 重验。

调用方契约：仅 KIVI 方法的 mixin 路径与本库 NPU 诊断脚本可导入。
"""

__all__: list[str] = []
