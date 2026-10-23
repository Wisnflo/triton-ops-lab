"""Implicit GEMM: virtual patch matrix never materialized in global memory."""

import torch
import triton
import triton.language as tl

from .capture import capture


@triton.jit
def _conv(
    X,
    W,
    Out,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    WI: tl.constexpr,
    OC: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pos = tl.program_id(0) * BM + tl.arange(0, BM)
    oc = tl.program_id(1) * BN + tl.arange(0, BN)
    rr = tl.arange(0, BK)
    batch = pos // (OH * OW)
    oy, ox = (pos % (OH * OW)) // OW, pos % OW
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(C * KH * KW, BK)):
        r = block * BK + rr
        channel = r // (KH * KW)
        # KH=KW=1 specializes into a strided GEMM with no patch extraction.
        iy = oy[:, None] * STRIDE - PAD + (r % (KH * KW) // KW)[None, :]
        ix = ox[:, None] * STRIDE - PAD + (r % KW)[None, :]
        address = ((batch[:, None] * C + channel[None, :]) * H + iy) * WI + ix
        valid = (pos[:, None] < B * OH * OW) & (r[None, :] < C * KH * KW)
        valid = valid & (iy >= 0) & (iy < H) & (ix >= 0) & (ix < WI)
        x = tl.load(X + address, valid, other=0)
        w = tl.load(
            W + oc[None, :] * (C * KH * KW) + r[:, None],
            (oc[None, :] < OC) & (r[:, None] < C * KH * KW),
            other=0,
        )
        acc += tl.dot(x, w)
    out_addr = ((batch[:, None] * OC + oc[None, :]) * OH + oy[:, None]) * OW + ox[:, None]
    tl.store(Out + out_addr, acc, (pos[:, None] < B * OH * OW) & (oc[None, :] < OC))


def conv2d(x, weight, stride=1, padding=0, config=None):
    cfg = config or {"bm": 32, "bn": 32, "bk": 32, "warps": 4, "stages": 2}
    b, c, h, w = x.shape
    oc, _, kh, kw = weight.shape
    oh, ow = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    out = torch.empty((b, oc, oh, ow), device=x.device, dtype=x.dtype)
    compiled = _conv[(triton.cdiv(b * oh * ow, cfg["bm"]), triton.cdiv(oc, cfg["bn"]))](
        x,
        weight,
        out,
        b,
        c,
        h,
        w,
        oc,
        kh,
        kw,
        oh,
        ow,
        stride,
        padding,
        cfg["bm"],
        cfg["bn"],
        cfg["bk"],
        num_warps=cfg["warps"],
        num_stages=cfg["stages"],
    )
    capture(compiled, "conv2d")
    return out
