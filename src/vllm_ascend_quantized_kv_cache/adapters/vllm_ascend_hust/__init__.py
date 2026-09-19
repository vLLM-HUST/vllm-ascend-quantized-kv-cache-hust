# SPDX-License-Identifier: Apache-2.0
"""vllm-ascend-hust 量化 KV attention backend 适配器。"""

from .register import BACKEND_CLASS_PATH_BY_DTYPE, AscendHustAdapter

__all__ = ["BACKEND_CLASS_PATH_BY_DTYPE", "AscendHustAdapter"]
