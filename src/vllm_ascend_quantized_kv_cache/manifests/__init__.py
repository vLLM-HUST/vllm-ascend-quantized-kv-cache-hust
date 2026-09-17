# SPDX-License-Identifier: Apache-2.0
"""Extension Manager 的 manifest 包。

pyproject 里的 ``vllm_hust.extension_bundles`` entry point 指向本模块。
管理器从包数据里读静态的 ``vllm-hust-extension-v0.2.json`` 描述符；
发现阶段绝不导入实现模块，所以本包刻意保持为空。
"""
