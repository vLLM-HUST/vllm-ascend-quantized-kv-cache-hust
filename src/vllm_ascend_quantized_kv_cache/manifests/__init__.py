# SPDX-License-Identifier: Apache-2.0
"""静态扩展 manifest 包。

``vllm_hust.extension_bundles`` 仅用于 Extension Manager 的无导入发现；
运行时注册仍由 ``vllm.general_plugins`` 调用 bootstrap 完成。
"""
