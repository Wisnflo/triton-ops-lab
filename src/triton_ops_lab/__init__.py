"""Forward-only research operators. GPU kernels are drafts awaiting CUDA validation."""

from .ops import attention, conv2d, matmul, prepare_matmul, softmax, vector_add

__all__ = ["attention", "conv2d", "matmul", "prepare_matmul", "softmax", "vector_add"]
