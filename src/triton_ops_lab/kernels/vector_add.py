"""Elementwise grid, following the Triton programming model (see docs/sources.md)."""

import torch
import triton
import triton.language as tl

from .capture import capture


@triton.jit
def _add(X, Y, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0).to(tl.float32)
    y = tl.load(Y + i, i < N, other=0).to(tl.float32)
    tl.store(Out + i, x + y, i < N)


def vector_add(x, y):
    out = torch.empty_like(x)
    compiled = _add[(triton.cdiv(x.numel(), 256),)](x, y, out, x.numel(), BLOCK=256)
    capture(compiled, "vector-add")
    return out
