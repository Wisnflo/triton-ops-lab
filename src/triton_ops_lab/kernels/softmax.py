import torch
import triton
import triton.language as tl

from .capture import capture


@triton.jit
def _softmax(X, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, other=-float("inf")).to(tl.float32)
    p = tl.exp(x - tl.max(x, axis=0))
    y = p / tl.sum(p, axis=0)
    tl.store(Out + row * N + col, y, col < N)


def softmax(x):
    block = triton.next_power_of_2(x.shape[1])
    if block > 16384:
        raise ValueError("Draft row softmax supports at most 16384 columns")
    out = torch.empty_like(x)
    compiled = _softmax[(x.shape[0],)](
        x, out, x.shape[1], block, num_warps=4 if block <= 2048 else 8
    )
    capture(compiled, "softmax")
    return out
