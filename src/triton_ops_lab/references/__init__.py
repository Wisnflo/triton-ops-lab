"""Readable forward algorithms; float64 inputs remain float64 for oracle checks."""

import math

import torch
import torch.nn.functional as F


def accumulator(x):
    return x if x.dtype == torch.float64 else x.float()


def vector_add(x, y):
    return (accumulator(x) + accumulator(y)).to(x.dtype)


def softmax(x):
    z = accumulator(x)
    z = torch.exp(z - z.amax(dim=-1, keepdim=True))
    return (z / z.sum(dim=-1, keepdim=True)).to(x.dtype)


def matmul(a, b, bias=None, activation=None, block=32):
    aa, bb = accumulator(a), accumulator(b)
    m, k = a.shape
    n = b.shape[1]
    out = torch.zeros((m, n), device=a.device, dtype=aa.dtype)
    for i in range(0, m, block):
        for j in range(0, n, block):
            for p in range(0, k, block):
                out[i : i + block, j : j + block] += (
                    aa[i : i + block, p : p + block] @ bb[p : p + block, j : j + block]
                )
    if bias is not None:
        out += accumulator(bias)
    if activation == "gelu":
        out = F.gelu(out, approximate="tanh")
    return out.to(a.dtype)


def attention_explicit(q, k, v, causal=False):
    qq, kk, vv = map(accumulator, (q, k, v))
    scores = (qq @ kk.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if causal:
        n = q.shape[-2]
        mask = torch.ones((n, n), device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, -torch.inf)
    return (torch.softmax(scores, dim=-1) @ vv).to(q.dtype)


def attention(q, k, v, causal=False, block=32):
    """Online softmax: carry max, denominator and *unnormalized* numerator."""
    qq, kk, vv = map(accumulator, (q, k, v))
    n, d = q.shape[-2:]
    out = torch.empty_like(qq)
    for i in range(0, n, block):
        qi = qq[..., i : i + block, :]
        shape = qi.shape[:-1] + (1,)
        m = torch.full(shape, -torch.inf, device=q.device, dtype=qq.dtype)
        denom = torch.zeros(shape, device=q.device, dtype=qq.dtype)
        numerator = torch.zeros_like(qi)
        # Causal future tiles are skipped so no tile has entirely masked valid rows.
        stop = min(n, i + block) if causal else n
        for j in range(0, stop, block):
            scores = qi @ kk[..., j : j + block, :].transpose(-1, -2) / math.sqrt(d)
            if causal:
                rows = torch.arange(i, min(i + block, n), device=q.device)[:, None]
                cols = torch.arange(j, min(j + block, n), device=q.device)[None, :]
                scores = scores.masked_fill(cols > rows, -torch.inf)
            new_m = torch.maximum(m, scores.amax(-1, keepdim=True))
            correction = torch.exp(m - new_m)
            p = torch.exp(scores - new_m)
            numerator = numerator * correction + p @ vv[..., j : j + block, :]
            denom = denom * correction + p.sum(-1, keepdim=True)
            m = new_m
        out[..., i : i + block, :] = numerator / denom
    return out.to(q.dtype)


def conv2d(x, weight, stride=1, padding=0):
    """Direct cross-correlation, without im2col; small CPU reference only."""
    xx, ww = accumulator(x), accumulator(weight)
    b, _, h, w = x.shape
    oc, _, kh, kw = weight.shape
    oh, ow = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    padded = F.pad(xx, (padding, padding, padding, padding))
    out = torch.empty((b, oc, oh, ow), dtype=xx.dtype, device=x.device)
    for i in range(oh):
        for j in range(ow):
            patch = padded[:, :, i * stride : i * stride + kh, j * stride : j * stride + kw]
            out[:, :, i, j] = torch.einsum("bchw,ochw->bo", patch, ww)
    return out.to(x.dtype)
