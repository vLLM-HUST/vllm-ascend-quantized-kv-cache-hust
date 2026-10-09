# SPDX-License-Identifier: Apache-2.0
"""Numeric checks shared by the KIVI INT4 device probes.

The probes compare what the triton pack kernel stored against what
``KiviInt4Semantics`` says it should have stored.  A plain magnitude tolerance
cannot express that comparison honestly:

* too tight and a rounding **tie** fails -- a value whose normalised form is an
  exact .5 leaves the kernel's fp32 arithmetic a hair below it, so the kernel
  and ``quantize_group`` land on opposite levels (measured on 910B2);
* too loose and a token that was never quantised at all passes, because a
  full-precision value is *closer* to the exact value than any level is.

So the check is done per level: an element may differ from the reference only by
sitting on the adjacent level, and only when the two candidate levels are
equidistant from the exact value -- which is what a tie looks like.
"""

from __future__ import annotations

import torch

#: fp16 rounding slack for values that must still be full precision
TOLERANCE = 1e-2
BITS = 4


def group_stats(sem, group_size: int, x: torch.Tensor, *, along_tokens: bool):
    """Per-element ``(min, step)`` of the reference grid.

    Keys group along the token dim, values along the head dim; mirroring that
    here is what makes a level comparison possible instead of a fuzzy magnitude
    one.  Padding follows ``fake_quant_key`` / ``fake_quant_value`` (repeat the
    last row/column) so a partial group still lines up.
    """
    work = x.float()
    if along_tokens:
        pad = (group_size - work.shape[0] % group_size) % group_size
        if pad:
            work = torch.cat([work, work[-1:].expand(pad, *work.shape[1:])], 0)
        grouped = work.view(-1, group_size, *x.shape[1:])
        mn = grouped.amin(dim=1, keepdim=True)
        mx = grouped.amax(dim=1, keepdim=True)
        scale = sem.group_scale(mn, mx, BITS)
        flat = tuple(x.shape[1:])
        return (
            mn.expand_as(grouped).reshape((-1, *flat))[: x.shape[0]],
            scale.expand_as(grouped).reshape((-1, *flat))[: x.shape[0]],
        )
    pad = (group_size - work.shape[-1] % group_size) % group_size
    if pad:
        work = torch.cat([work, work[..., -1:].expand(*work.shape[:-1], pad)], -1)
    grouped = work.view(*work.shape[:-1], -1, group_size)
    mn = grouped.amin(dim=-1, keepdim=True)
    mx = grouped.amax(dim=-1, keepdim=True)
    scale = sem.group_scale(mn, mx, BITS)
    shape = tuple(x.shape)
    return (
        mn.expand_as(grouped).reshape(shape),
        scale.expand_as(grouped).reshape(shape),
    )


def prefix_check(
    sem,
    group_size: int,
    got: torch.Tensor,
    expected: torch.Tensor,
    exact: torch.Tensor,
    *,
    along_tokens: bool,
) -> tuple[int, int]:
    """Violations and tie flips in a quantised region.

    ``exact`` is the full-precision input, ``expected`` the reference
    quantisation of it, ``got`` what the cache reads back.
    """
    got = got.float()
    expected = expected.float()
    exact = exact.float()
    _, scale = group_stats(sem, group_size, exact, along_tokens=along_tokens)
    scale = scale.float()
    diff = (got - expected).abs()
    d_ref = (exact - expected).abs()
    d_got = (exact - got).abs()
    tie = (diff <= 1.05 * scale) & ((d_got - d_ref).abs() <= 0.05 * scale)
    allowed = (diff <= TOLERANCE) | tie
    return int((~allowed).sum()), int((tie & (diff > TOLERANCE)).sum())


def score(
    sem,
    group_size: int,
    got: torch.Tensor,
    expected: torch.Tensor,
    exact: torch.Tensor,
    split: int,
    *,
    along_tokens: bool,
):
    """Worst deviation, violation count and tie flips across both regions.

    ``split`` is the number of leading elements that must be quantised; the
    rest has to still be full precision.
    """
    worst = 0.0
    broken = 0
    ties = 0
    if split:
        diff = (got[:split] - expected[:split]).abs().amax().item()
        worst = max(worst, diff)
        broken_here, ties_here = prefix_check(
            sem,
            group_size,
            got[:split],
            expected[:split],
            exact[:split],
            along_tokens=along_tokens,
        )
        broken += broken_here
        ties += ties_here
    if split < got.shape[0]:
        diff = (got[split:] - expected[split:]).abs().amax().item()
        worst = max(worst, diff)
        if diff > TOLERANCE:
            broken += 1
    return worst, broken, ties
