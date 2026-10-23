import torch

from .benchmarks.cases import make_callable, make_inputs, oracle
from .ops import require_cuda
from .validation import compare


SMOKE_CASES = [
    {"id": "add-tail", "op": "vector_add", "shape": [259]},
    {"id": "softmax-tail", "op": "softmax", "shape": [3, 37]},
    {"id": "gemm-tail", "op": "matmul", "shape": [17, 23, 19], "transpose": True},
    {
        "id": "gemm-epilogue",
        "op": "matmul",
        "shape": [17, 23, 19],
        "bias": True,
        "activation": "gelu",
    },
    {"id": "attention", "op": "attention", "shape": [1, 2, 35, 32], "causal": False},
    {"id": "attention-causal", "op": "attention", "shape": [1, 2, 35, 64], "causal": True},
    {"id": "conv1", "op": "conv2d", "shape": [1, 3, 7, 9, 5, 1], "stride": 2, "padding": 1},
    {"id": "conv3", "op": "conv2d", "shape": [1, 3, 7, 9, 5, 3], "stride": 2, "padding": 1},
    {"id": "transformer", "op": "transformer", "shape": [1, 7, 64]},
    {"id": "cnn", "op": "cnn", "shape": [1, 3, 7, 9]},
]


def check(device="cpu"):
    if device not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or cuda")
    if device == "cuda":
        require_cuda()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    rows = []
    dtypes = [torch.float32] if device == "cpu" else [torch.float16, torch.bfloat16]
    for dtype in dtypes:
        for original in SMOKE_CASES:
            case = {**original, "dtype": str(dtype).removeprefix("torch.")}
            args, model = make_inputs(case, device, dtype)
            expected = oracle(case, args, model)
            variants = ["reference"] if device == "cpu" else ["triton_fixed"]
            if device == "cuda" and case["op"] == "matmul":
                variants = ["triton_basic", "triton_grouped", "triton_fused"]
            for backend in variants:
                actual = make_callable(case, args, model, backend)()
                rows.append(
                    {
                        "case_id": case["id"],
                        "dtype": case["dtype"],
                        "backend": backend,
                        **compare(actual, expected, case["op"], case["dtype"]),
                    }
                )
    return rows
