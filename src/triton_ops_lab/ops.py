"""Explicit forward-only dispatch. Importing this module never imports Triton."""

import importlib
import platform

import torch
import torch.nn.functional as F

from . import references
from .compat import mm_fp32


class UnsupportedBackend(RuntimeError):
    pass


def require_cuda():
    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise UnsupportedBackend(
            "Triton experiments require a Linux NVIDIA CUDA host; no CPU fallback"
        )
    if torch.cuda.get_device_capability()[0] < 8:
        raise UnsupportedBackend("GPU compute capability must be >= 8.0")
    try:
        importlib.import_module("triton")
    except ImportError as exc:
        raise UnsupportedBackend("Compatible Triton installation is missing") from exc


def _validate(tensors, backend):
    if backend not in {"torch", "torch_mm_compat", "reference", "triton"}:
        raise ValueError(f"Unknown backend: {backend}")
    if not tensors or any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("Inputs must be torch tensors")
    first = tensors[0]
    for t in tensors:
        if t.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
            raise ValueError("Only floating point FP16/BF16/FP32/FP64 inputs supported")
        if t.dtype != first.dtype or t.device != first.device:
            raise ValueError("All inputs must have the same dtype and device")
        if not t.numel() or any(s <= 0 for s in t.stride()):
            raise ValueError("Empty tensors and non-positive strides are unsupported")
        if t.requires_grad:
            raise ValueError("Forward inference only; detach inputs before calling")
    if backend == "triton":
        require_cuda()
        if first.device.type != "cuda" or first.dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("Triton backend requires CUDA FP16/BF16 tensors")


def _kernel(name):
    return getattr(importlib.import_module(f".kernels.{name}", __package__), name)


def _validate_config(config, op, backend):
    if config is None:
        return
    if backend != "triton" or not isinstance(config, dict):
        raise ValueError("Explicit tile config is only supported by the Triton backend")
    keys = {"bm", "bn", "warps", "stages"}
    if op != "attention":
        keys.add("bk")
    allowed = keys | ({"group"} if op == "matmul" else set())
    if not keys <= set(config) <= allowed or any(type(v) is not int for v in config.values()):
        raise ValueError("Tile config requires exact integer bm/bn/[bk]/warps/stages fields")
    if any(config[k] not in {16, 32, 64, 128} for k in ("bm", "bn")):
        raise ValueError("Tile dimensions must be 16/32/64/128")
    if (
        config.get("bk", 32) not in {32, 64}
        or config["warps"] not in {4, 8}
        or not 1 <= config["stages"] <= 4
    ):
        raise ValueError("Unsupported reduction tile, warp count or pipeline stage count")
    if config.get("group", 8) not in {1, 4, 8, 16}:
        raise ValueError("Unsupported GEMM group size")


@torch.no_grad()
def vector_add(x, y, *, backend="torch"):
    _validate((x, y), backend)
    if x.ndim != 1 or x.shape != y.shape or not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("vector_add requires equal contiguous vectors")
    if backend == "triton":
        return _kernel("vector_add")(x, y)
    return references.vector_add(x, y) if backend == "reference" else x + y


@torch.no_grad()
def softmax(x, *, backend="torch"):
    _validate((x,), backend)
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("softmax requires a contiguous 2D tensor")
    if backend == "triton":
        return _kernel("softmax")(x)
    return references.softmax(x) if backend == "reference" else torch.softmax(x, dim=-1)


def _validate_matmul(a, b, bias, activation, backend, variant, config):
    _validate_config(config, "matmul", backend)
    _validate((a, b) if bias is None else (a, b, bias), backend)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("matmul requires [M,K] and [K,N]")
    if any(not (t.is_contiguous() or t.T.is_contiguous()) for t in (a, b)):
        raise ValueError("matmul accepts contiguous or transposed contiguous matrices")
    if bias is not None and (bias.shape != (b.shape[1],) or not bias.is_contiguous()):
        raise ValueError("bias must be a contiguous [N] vector")
    if activation not in {None, "gelu"} or variant not in {"basic", "grouped", "fused"}:
        raise ValueError("Unsupported activation or GEMM variant")


@torch.no_grad()
def matmul(a, b, bias=None, *, activation=None, backend="torch", variant="grouped", config=None):
    _validate_matmul(a, b, bias, activation, backend, variant, config)
    if backend == "triton":
        return _kernel("matmul")(a, b, bias, activation, variant, config)
    if backend == "reference":
        return references.matmul(a, b, bias, activation)
    # Match fused epilogue semantics: FP32 accumulation and epilogue before final cast.
    dtype = torch.float64 if a.dtype == torch.float64 else torch.float32
    # CUDA low-precision inputs retain Tensor Core GEMM with FP32 epilogue output.
    if bias is None and activation is None:
        return a @ b
    if a.is_cuda and a.dtype in {torch.float16, torch.bfloat16}:
        out = (
            mm_fp32(a, b)
            if backend == "torch_mm_compat"
            else torch.mm(a, b, out_dtype=torch.float32)
        )
    else:
        out = a.to(dtype) @ b.to(dtype)
    if bias is not None:
        out = out + bias.to(dtype)
    if activation == "gelu":
        out = F.gelu(out, approximate="tanh")
    return out.to(a.dtype)


def prepare_matmul(a, b, bias=None, *, activation=None, config=None):
    """Bind CUDA inputs and fused launch parameters, retaining per-call metadata guards.

    Contents may change in place; storage/shape/strides/dtype/device/grad state may
    not. Every invocation allocates and returns a fresh output. Rebind explicitly
    for new tensors or metadata. Preparation cost is separate from steady calls.
    """
    _validate_matmul(a, b, bias, activation, "triton", "fused", config)
    module = importlib.import_module(".kernels.matmul", __package__)
    return module.prepare_matmul(a, b, bias, activation, config)


@torch.no_grad()
def attention(q, k, v, *, causal=False, backend="torch", config=None):
    _validate_config(config, "attention", backend)
    _validate((q, k, v), backend)
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[-1] not in {32, 64, 128}:
        raise ValueError("attention requires equal [B,H,N,D] with D=32/64/128")
    if any(not t.is_contiguous() for t in (q, k, v)) or not isinstance(causal, bool):
        raise ValueError("attention requires contiguous tensors and a boolean causal flag")
    if backend == "triton":
        return _kernel("attention")(q, k, v, causal, config)
    if backend == "reference":
        return references.attention(q, k, v, causal)
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)


@torch.no_grad()
def conv2d(x, weight, *, stride=1, padding=0, backend="torch", config=None):
    _validate_config(config, "conv2d", backend)
    _validate((x, weight), backend)
    if x.ndim != 4 or weight.ndim != 4 or x.shape[1] != weight.shape[1]:
        raise ValueError("conv2d requires NCHW inputs and OIHW weights with matching channels")
    if (
        weight.shape[2:] not in {(1, 1), (3, 3)}
        or type(stride) is not int
        or type(padding) is not int
        or stride not in {1, 2}
        or padding not in {0, 1}
    ):
        raise ValueError("Only 1x1/3x3, stride=1/2 and padding=0/1 are supported")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("conv2d requires contiguous NCHW/OIHW")
    if any(s + 2 * padding < k for s, k in zip(x.shape[2:], weight.shape[2:], strict=True)):
        raise ValueError("Convolution output dimensions must be positive")
    if backend == "triton":
        return _kernel("conv2d")(x, weight, stride, padding, config)
    if backend == "reference":
        return references.conv2d(x, weight, stride, padding)
    return F.conv2d(x, weight, stride=stride, padding=padding)
