"""Dynamo contract tests on CPU; these do not validate GPU Inductor kernels."""

import pytest
import torch

from triton_ops_lab.benchmarks.cases import make_callable, make_inputs
from triton_ops_lab.benchmarks.compilation import CompileAudit, compile_strict


@pytest.fixture
def fx_backend(monkeypatch):
    import torch._inductor

    monkeypatch.setattr(torch._inductor, "compile", lambda gm, inputs, **kwargs: gm.forward)
    yield
    torch.compiler.reset()


def test_case_cache_does_not_exhaust_shared_code_limit(fx_backend):
    # More cases than PyTorch 2.8's default shared-frame recompile limit (8).
    for n in range(17, 29):
        case = {"op": "vector_add", "shape": [n], "dtype": "float16"}
        args, model = make_inputs(case)
        fn = make_callable(case, args, model, "compile")
        torch.testing.assert_close(fn(), args[0] + args[1])
        for _ in range(3):
            fn()
        fn.compile_audit.require_single_graph()
        assert "add" in str(fn.compile_audit.snapshot()["graphs"])


def test_graph_break_cannot_be_reported_as_compiled(fx_backend):
    def breaks(x):
        return x + x.sum().item()

    fn = compile_strict(breaks)
    with pytest.raises(torch._dynamo.exc.Unsupported):
        fn(torch.ones(2))
    assert fn.compile_audit.snapshot()["graph_count"] == 0


def test_audit_rejects_missing_or_multiple_graphs():
    audit = CompileAudit()
    with pytest.raises(RuntimeError, match="exactly one"):
        audit.require_single_graph()
    audit.graphs = [{}, {}]
    with pytest.raises(RuntimeError, match="got 2"):
        audit.require_single_graph()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("transpose", [False, True])
def test_mm_bridge_metadata_and_fx_graph(dtype, transpose):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    from triton_ops_lab.compat import mm_fp32

    # CUDA FakeTensors require no driver or GPU allocation. This checks metadata
    # and dispatch registration only; GPU opcheck and numerical tests are separate.
    with FakeTensorMode():
        a = torch.empty((11, 7) if transpose else (7, 11), device="cuda", dtype=dtype)
        b = torch.empty((13, 11) if transpose else (11, 13), device="cuda", dtype=dtype)
        if transpose:
            a, b = a.T, b.T
        out = mm_fp32(a, b)
        assert out.shape == (7, 13) and out.dtype == torch.float32
        assert out.device == a.device and out.stride() == (13, 1)
        gm = make_fx(lambda x, y: mm_fp32(x, y) + 1)(a, b)
        assert "triton_ops_lab.mm_fp32" in gm.code


def test_mm_bridge_metadata_rejects_wrong_reduction():
    from torch._subclasses.fake_tensor import FakeTensorMode

    from triton_ops_lab.compat import mm_fp32

    with FakeTensorMode():
        a = torch.empty((7, 11), device="cuda", dtype=torch.float16)
        b = torch.empty((12, 13), device="cuda", dtype=torch.float16)
        with pytest.raises(ValueError, match="M,K"):
            mm_fp32(a, b)


def test_mm_compat_audit_exposes_external_node(fx_backend):
    from torch._subclasses.fake_tensor import FakeTensorMode

    case = dict(op="matmul", shape=[7, 13, 11], dtype="float16", bias=True, activation="gelu")
    with FakeTensorMode():
        args = tuple(
            torch.empty(shape, device="cuda", dtype=torch.float16)
            for shape in ((7, 11), (11, 13), (13,))
        )
        fn = make_callable(case, args, None, "compile_mm_compat")
        out = fn()
        assert out.shape == (7, 13) and out.dtype == torch.float16
        fn.compile_audit.require_single_graph()
        assert fn.compile_audit.snapshot()["opaque_external_ops"] == [
            "triton_ops_lab.mm_fp32.default"
        ]
