# SPDX-License-Identifier: Apache-2.0
"""vllm-ascend-hust INT8 attention backend 适配器。"""

from .register import BACKEND_CLASS_PATH, AscendHustAdapter

__all__ = ["BACKEND_CLASS_PATH", "AscendHustAdapter"]
