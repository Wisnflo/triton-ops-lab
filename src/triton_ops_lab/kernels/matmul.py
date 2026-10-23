"""Tiled GEMM, grouped scheduling and FP32 epilogue. GPU validation pending."""

import torch
import triton
import triton.language as tl

from .capture import capture
from ..contracts import BoundTensorGuard


@triton.jit
def _gemm(
    A,
    B,
    Bias,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    BM: tl.constexpr,
    BN_TILE: tl.constexpr,
    BK_TILE: tl.constexpr,
    GROUP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GELU: tl.constexpr,
):
    pid = tl.program_id(0)
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN_TILE)
    group_id = pid // (GROUP * nn)
    first_m = group_id * GROUP
    size_m = tl.minimum(nm - first_m, GROUP)
    pm = first_m + (pid % (GROUP * nn)) % size_m
    pn = (pid % (GROUP * nn)) // size_m
    mi = pm * BM + tl.arange(0, BM)
    ni = pn * BN_TILE + tl.arange(0, BN_TILE)
    ki = tl.arange(0, BK_TILE)
    acc = tl.full((BM, BN_TILE), 0, tl.float32)
    for start in range(tl.cdiv(K, BK_TILE)):
        kk = start * BK_TILE + ki
        a = tl.load(
            A + mi[:, None] * AM + kk[None, :] * AK, (mi[:, None] < M) & (kk[None, :] < K), other=0
        )
        b = tl.load(
            B + kk[:, None] * BK + ni[None, :] * BN, (kk[:, None] < K) & (ni[None, :] < N), other=0
        )
        acc += tl.dot(a, b)
    if HAS_BIAS:
        acc += tl.load(Bias + ni, ni < N, other=0)[None, :].to(tl.float32)
    if GELU:
        z = 0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)
        # tanh(z) = 2 sigmoid(2z)-1, same tanh GELU definition as the reference.
        acc = 0.5 * acc * (1 + (2 / (1 + tl.exp(-2 * z)) - 1))
    tl.store(Out + mi[:, None] * N + ni[None, :], acc, (mi[:, None] < M) & (ni[None, :] < N))


@triton.jit
def _epilogue(
    X,
    Bias,
    Out,
    SIZE: tl.constexpr,
    N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GELU: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    z = tl.load(X + i, i < SIZE, other=0).to(tl.float32)
    if HAS_BIAS:
        z += tl.load(Bias + i % N, i < SIZE, other=0).to(tl.float32)
    if GELU:
        t = 0.7978845608028654 * (z + 0.044715 * z * z * z)
        z = 0.5 * z * (1 + (2 / (1 + tl.exp(-2 * t)) - 1))
    tl.store(Out + i, z, i < SIZE)


def matmul(a, b, bias=None, activation=None, variant="grouped", config=None):
    cfg = config or {"bm": 32, "bn": 64, "bk": 32, "warps": 4, "stages": 2}
    m, k = a.shape
    n = b.shape[1]
    fused = variant == "fused"
    epilogue = bias is not None or activation is not None
    # Unfused variant stores FP32 intermediate, preserving identical epilogue precision.
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    intermediate = (
        torch.empty((m, n), device=a.device, dtype=torch.float32) if epilogue and not fused else out
    )
    compiled = _gemm[(triton.cdiv(m, cfg["bm"]) * triton.cdiv(n, cfg["bn"]),)](
        a,
        b,
        bias if bias is not None else a,
        intermediate,
        m,
        n,
        k,
        *a.stride(),
        *b.stride(),
        cfg["bm"],
        cfg["bn"],
        cfg["bk"],
        1 if variant == "basic" else cfg.get("group", 8),
        fused and bias is not None,
        fused and activation == "gelu",
        num_warps=cfg["warps"],
        num_stages=cfg["stages"],
    )
    capture(compiled, "gemm-" + variant)
    if epilogue and not fused:
        compiled_epilogue = _epilogue[(triton.cdiv(m * n, 256),)](
            intermediate,
            bias if bias is not None else a,
            out,
            m * n,
            n,
            bias is not None,
            activation == "gelu",
            256,
        )
        capture(compiled_epilogue, "gemm-epilogue")
    return out


def prepare_matmul(a, b, bias=None, activation=None, config=None):
    """Called after ops.prepare_matmul validates the complete public contract."""
    cfg = dict(config or {"bm": 32, "bn": 64, "bk": 32, "warps": 4, "stages": 2})
    m, k = a.shape
    n = b.shape[1]
    guard = BoundTensorGuard((a, b) if bias is None else (a, b, bias))
    launch = _gemm[(triton.cdiv(m, cfg["bm"]) * triton.cdiv(n, cfg["bn"]),)]
    constants = (
        m,
        n,
        k,
        *a.stride(),
        *b.stride(),
        cfg["bm"],
        cfg["bn"],
        cfg["bk"],
        cfg.get("group", 8),
        bias is not None,
        activation == "gelu",
    )
    bias_pointer = bias if bias is not None else a

    def invoke():
        guard.check()
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
        compiled = launch(
            a, b, bias_pointer, out, *constants, num_warps=cfg["warps"], num_stages=cfg["stages"]
        )
        capture(compiled, "gemm-prepared")
        return out

    return invoke
