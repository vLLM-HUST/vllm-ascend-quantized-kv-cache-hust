# SPDX-License-Identifier: Apache-2.0
"""Built-in quantized KV-cache solutions.

Importing this package registers every solution's *metadata* into the
registry; no torch / vllm / device module is imported at this point.
"""

from . import int8_dynamic, kivi_int4, packed  # noqa: F401  (registration)

__all__ = ["int8_dynamic", "kivi_int4", "packed"]
