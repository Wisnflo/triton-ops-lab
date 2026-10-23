"""Never emulate these tests on CPU. Run on a real Linux CUDA GPU."""

import pytest
import torch

from triton_ops_lab.benchmarks.cases import make_callable, make_inputs, oracle
from triton_ops_lab.benchmarks.selection import candidates
from triton_ops_lab.checks import SMOKE_CASES
from triton_ops_lab.validation import compare
from triton_ops_lab.benchmarks.runner import measure


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", SMOKE_CASES, ids=lambda c: c["id"])
def test_gpu_operators(case, dtype):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    case = {**case, "dtype": str(dtype).removeprefix("torch.")}
    args, model = make_inputs(case, "cuda", dtype)
    expected = oracle(case, args, model)
    backends = (
        ["triton_basic", "triton_grouped", "triton_fused"]
        if case["op"] == "matmul"
        else ["triton_fixed"]
    )
    for backend in backends:
        for cfg in candidates(case["op"], case["shape"], "search"):
            actual = make_callable(case, args, model, backend, cfg)()
            torch.cuda.synchronize()
            result = compare(actual, expected, case["op"], case["dtype"])
            assert result["passed"], (backend, cfg, result)


@pytest.mark.gpu
@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_attention_tiles(d, causal):
    case = {
        "id": "attention-tiles",
        "op": "attention",
        "shape": [1, 2, 129, d],
        "dtype": "float16",
        "causal": causal,
    }
    args, model = make_inputs(case, "cuda")
    expected = oracle(case, args, model)
    for cfg in candidates("attention", case["shape"], "search"):
        assert compare(
            make_callable(case, args, model, "triton_fixed", cfg)(), expected, "attention"
        )["passed"]


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "kernel,stride,padding", [(k, s, p) for k in (1, 3) for s in (1, 2) for p in (0, 1)]
)
def test_conv_boundaries(dtype, kernel, stride, padding):
    case = {
        "id": "conv-boundary",
        "op": "conv2d",
        "shape": [2, 3, 7, 9, 5, kernel],
        "dtype": str(dtype).removeprefix("torch."),
        "stride": stride,
        "padding": padding,
    }
    args, model = make_inputs(case, "cuda", dtype)
    expected = oracle(case, args, model)
    actual = make_callable(case, args, model, "triton_fixed")()
    assert compare(actual, expected, "conv2d")["passed"]


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("index", [0, 3, 5])
def test_graph_replay_correctness(dtype, index):
    case = {**SMOKE_CASES[index], "dtype": dtype}
    args, model = make_inputs(case, "cuda")
    expected = oracle(case, args, model)
    for backend in ("torch", "triton_fixed"):
        fn = make_callable(case, args, model, backend)
        result = measure(
            fn, dict(mode="cuda_graph", warmup=3, samples=3, repeats=3), expected, case
        )
        assert result["median_ms"] > 0 and len(result["wall_samples_ms"]) == 3


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("transpose", [False, True])
def test_mm_bridge_dispatch_contract(dtype, transpose):
    from triton_ops_lab.compat import mm_fp32

    a = torch.randn((49, 65) if transpose else (65, 49), device="cuda", dtype=dtype)
    b = torch.randn((97, 49) if transpose else (49, 97), device="cuda", dtype=dtype)
    if transpose:
        a, b = a.T, b.T
    expected = torch.mm(a, b, out_dtype=torch.float32)
    torch.testing.assert_close(mm_fp32(a, b), expected, rtol=0, atol=0)
    result = torch.library.opcheck(mm_fp32, (a, b))
    assert all(status == "SUCCESS" for status in result.values()), result


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "case",
    [
        {"op": "matmul", "shape": [65, 97, 49], "bias": True, "activation": "gelu"},
        {"op": "matmul", "shape": [33, 17, 65], "bias": True, "transpose": True},
        {"op": "matmul", "shape": [17, 65, 33], "activation": "gelu"},
        {"op": "transformer", "shape": [1, 7, 64]},
    ],
)
def test_compile_mm_compat_correctness_and_graph(dtype, case):
    case = {**case, "dtype": dtype}
    args, model = make_inputs(case, "cuda")
    expected = oracle(case, args, model)
    fn = make_callable(case, args, model, "compile_mm_compat")
    assert compare(fn(), expected, case["op"], dtype)["passed"]
    fn.compile_audit.require_single_graph()
    audit = fn.compile_audit.snapshot()
    assert audit["mm_policy"] == "explicit_external_fp32_mm"
    assert audit["opaque_external_ops"], audit
    measure(fn, dict(mode="cuda_graph", warmup=3, samples=3, repeats=3), expected, case)
    fn.compile_audit.require_single_graph()


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "case",
    [
        {"op": "matmul", "shape": [65, 97, 49], "bias": True, "activation": "gelu"},
        {"op": "matmul", "shape": [128, 128, 128], "bias": True},
        {"op": "matmul", "shape": [17, 23, 19], "transpose": True},
        {"op": "matmul", "shape": [257, 193, 129], "activation": "gelu"},
    ],
)
def test_prepared_matmul(dtype, case):
    case = {**case, "dtype": dtype}
    args, model = make_inputs(case, "cuda")
    original = args[0].clone()
    for cfg in candidates("matmul", case["shape"], "search", 6):
        args[0].copy_(original)
        fn = make_callable(case, args, model, "triton_prepared", cfg)
        expected = oracle(case, args, model)
        first, second = fn(), fn()
        assert first.data_ptr() != second.data_ptr()
        assert compare(first, expected, "matmul", dtype)["passed"]
        assert compare(second, expected, "matmul", dtype)["passed"]
        args[0].mul_(0.5)
        assert compare(fn(), oracle(case, args, model), "matmul", dtype)["passed"]
        measure(
            fn,
            dict(mode="cuda_graph", warmup=3, samples=3, repeats=3),
            oracle(case, args, model),
            case,
        )
    args[0].transpose_(0, 1)
    with pytest.raises(RuntimeError, match="metadata/storage"):
        fn()
