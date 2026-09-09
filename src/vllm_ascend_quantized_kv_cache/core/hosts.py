# SPDX-License-Identifier: Apache-2.0
"""Host names recognised by the adapter layer."""

VLLM_HUST = "vllm_hust"
VLLM_ASCEND_HUST = "vllm_ascend_hust"

ALL_HOSTS = (VLLM_HUST, VLLM_ASCEND_HUST)

__all__ = ["VLLM_HUST", "VLLM_ASCEND_HUST", "ALL_HOSTS"]
