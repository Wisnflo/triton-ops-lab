"""Explicit torch 2.8 FP32-output GEMM bridge for a named compile baseline.

The CUDA implementation is the original torch.mm(out_dtype=float32). The
registered fake implementation supplies metadata missing in torch 2.8's
aten.mm.dtype path. Inductor sees an opaque external GEMM and can compile its
surrounding epilogue. This does not claim to fix or lower aten.mm.dtype itself.
"""

import torch


def _validate_mm(a, b):
    if (
        not a.is_cuda
        or b.device != a.device
        or a.dtype != b.dtype
        or a.dtype not in {torch.float16, torch.bfloat16}
    ):
        raise ValueError("mm_fp32 requires CUDA FP16/BF16 inputs of matching dtype/device")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("mm_fp32 requires [M,K] and [K,N]")
    if a.requires_grad or b.requires_grad:
        raise ValueError("mm_fp32 is forward inference only")


@torch.library.custom_op("triton_ops_lab::mm_fp32", mutates_args=(), device_types="cuda")
def mm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    _validate_mm(a, b)
    return torch.mm(a, b, out_dtype=torch.float32)


@mm_fp32.register_fake
def _mm_fp32_fake(a, b):
    _validate_mm(a, b)
    return a.new_empty((a.shape[0], b.shape[1]), dtype=torch.float32)
