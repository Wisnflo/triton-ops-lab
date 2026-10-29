"""Online softmax attention; algorithm attribution in docs/project.md."""

import math

import torch
import triton
import triton.language as tl

from .capture import capture


@triton.jit
def _attention(
    Q,
    K,
    V,
    Out,
    N: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    block_m, bh = tl.program_id(0), tl.program_id(1)
    rows = block_m * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    ds = tl.arange(0, D)
    offset = bh * N * D
    q = tl.load(Q + offset + rows[:, None] * D + ds[None, :], rows[:, None] < N, other=0)
    m = tl.full((BM,), -float("inf"), tl.float32)
    denom = tl.full((BM,), 0, tl.float32)
    acc = tl.full((BM, D), 0, tl.float32)
    stop = N
    if CAUSAL:
        stop = tl.minimum(N, (block_m + 1) * BM)
    for j in range(tl.cdiv(stop, BN)):
        keys = j * BN + cols
        k = tl.load(K + offset + keys[None, :] * D + ds[:, None], keys[None, :] < N, other=0)
        v = tl.load(V + offset + keys[:, None] * D + ds[None, :], keys[:, None] < N, other=0)
        scores = tl.dot(q, k) * SCALE
        valid = keys[None, :] < N
        if CAUSAL:
            valid = valid & (keys[None, :] <= rows[:, None])
        scores = tl.where(valid, scores, -float("inf"))
        new_m = tl.maximum(m, tl.max(scores, axis=1))
        p = tl.exp(scores - new_m[:, None])
        alpha = tl.exp(m - new_m)
        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
        denom = denom * alpha + tl.sum(p, axis=1)
        m = new_m
    result = acc / denom[:, None]
    tl.store(Out + offset + rows[:, None] * D + ds[None, :], result, rows[:, None] < N)


def attention(q, k, v, causal=False, config=None):
    cfg = config or {"bm": 32, "bn": 32, "warps": 4, "stages": 2}
    b, h, n, d = q.shape
    out = torch.empty_like(q)
    compiled = _attention[(triton.cdiv(n, cfg["bm"]), b * h)](
        q,
        k,
        v,
        out,
        n,
        d,
        1 / math.sqrt(d),
        causal,
        cfg["bm"],
        cfg["bn"],
        num_warps=cfg["warps"],
        num_stages=cfg["stages"],
    )
    capture(compiled, "attention")
    return out
