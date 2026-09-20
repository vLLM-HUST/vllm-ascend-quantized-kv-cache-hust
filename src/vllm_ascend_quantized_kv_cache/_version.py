# SPDX-License-Identifier: Apache-2.0
"""发行版本的唯一来源（single source of truth）。

Bundle 指南要求静态 manifest 里的 ``bundle_version`` 与
发行版本完全一致。版本号只在这里写一份，其他地方（pyproject 动态读取、
manifest、测试断言）都引用它。
"""

__version__ = "0.2.0rc1"
