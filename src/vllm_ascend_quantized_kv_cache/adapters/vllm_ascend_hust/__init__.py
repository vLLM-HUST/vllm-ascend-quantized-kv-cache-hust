# SPDX-License-Identifier: Apache-2.0
"""vllm-ascend-hust host adapter.

Attach mechanism (no host changes required):

1. the solution's scheme class is registered into the host's
   ``@register_scheme`` registry under a namespaced quant_type key;
2. attention layers pick it up through the checkpoint ``fa_quant_type``
   key (ModelSlim path), or the bootstrap registers it in every process;
3. for the stateful solutions the scheme's ``create_weights`` swaps the
   layer impl to our mixin-backed ``AscendAttentionBackendImpl`` subclass
   (the in-tree C8 precedent).
"""

from .attention import apply_impl_surgery, build_impl_cls, supported_impl_solutions
from .register import AscendHustAdapter
from .scheme import build_scheme_cls, packed_scheme_for

__all__ = [
    "AscendHustAdapter",
    "apply_impl_surgery",
    "build_impl_cls",
    "build_scheme_cls",
    "packed_scheme_for",
    "supported_impl_solutions",
]
