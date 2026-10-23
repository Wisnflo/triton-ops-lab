from contextlib import nullcontext

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .. import ops, references
from ..models import TinyCNN, TinyTransformer
from ..policy import FrozenPolicy
from ..artifacts import project_root
from .compilation import compile_strict
from .selection import GEMM_ABLATIONS, RULE_BACKENDS, PREPARED_RULE_BACKENDS


def make_inputs(case, device="cpu", dtype=None, seed=0):
    dtype = dtype or getattr(torch, case["dtype"])
    gen = torch.Generator(device=device).manual_seed(seed)

    def rand(shape):
        return torch.randn(shape, device=device, dtype=dtype, generator=gen)

    op, s = case["op"], case["shape"]
    model = None
    if op == "vector_add":
        args = (rand(s), rand(s))
    elif op == "softmax":
        args = (rand(s),)
    elif op == "matmul":
        m, n, k = s
        a, b = (
            (rand((k, m)).T, rand((n, k)).T)
            if case.get("transpose")
            else (rand((m, k)), rand((k, n)))
        )
        args = (a, b, rand((n,)) if case.get("bias") else None)
    elif op == "attention":
        args = tuple(rand(s) for _ in range(3))
    elif op == "conv2d":
        b, c, h, w, oc, kernel = s
        args = (rand((b, c, h, w)), rand((oc, c, kernel, kernel)))
    else:
        # fork_rng keeps weight construction reproducible without changing caller RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = TinyTransformer(width=s[-1]) if op == "transformer" else TinyCNN(channels=s[1])
        model = model.to(device=device, dtype=dtype).eval()
        args = (rand(s),)
    return args, model


def applicable(case, backend):
    op = case["op"]
    if backend in GEMM_ABLATIONS:
        return op == "matmul"
    if backend in PREPARED_RULE_BACKENDS:
        return op == "matmul"
    if backend in RULE_BACKENDS:
        return op in (
            {"matmul", "attention", "conv2d", "transformer", "cnn"}
            if backend == "triton_rule"
            else {"matmul", "attention", "conv2d"}
        )
    if backend.startswith("sdpa_"):
        return op == "attention"
    if backend in {"triton_basic", "triton_grouped", "triton_fused"}:
        return op == "matmul"
    if backend in {"triton_prepared", "triton_prepared_search"} | PREPARED_RULE_BACKENDS:
        return op == "matmul"
    if backend in {"triton_search", "triton_pruned"}:
        return op in {"matmul", "attention", "conv2d"}
    return True


def make_callable(case, args, model, backend, config=None, policy=None):
    op = case["op"]
    if not applicable(case, backend):
        raise ValueError("Backend does not apply to this operator")
    if backend in GEMM_ABLATIONS:
        config = dict(GEMM_ABLATIONS[backend]) if config is None else config
        backend = "triton_fused"
    plan = None
    if backend in RULE_BACKENDS:
        policy = policy or FrozenPolicy.load(project_root() / "configs/policy-v1.json")
        if model is not None:
            plan = model.make_plan(case["shape"], case["dtype"], policy)
        elif config is None:
            config = policy.choose(case)
    if backend in {"triton_prepared", "triton_prepared_search"} | PREPARED_RULE_BACKENDS:
        return ops.prepare_matmul(*args, activation=case.get("activation"), config=config)
    dispatch = (
        "triton"
        if backend.startswith("triton_")
        else "reference"
        if backend == "reference"
        else "torch_mm_compat"
        if backend == "compile_mm_compat"
        else "torch"
    )
    variant = (
        backend.removeprefix("triton_")
        if backend in {"triton_basic", "triton_grouped", "triton_fused"}
        else "fused"
    )
    forced = {
        "sdpa_math": SDPBackend.MATH,
        "sdpa_flash": SDPBackend.FLASH_ATTENTION,
        "sdpa_efficient": SDPBackend.EFFICIENT_ATTENTION,
    }.get(backend)

    def invoke():
        context = sdpa_kernel(forced) if forced is not None else nullcontext()
        with context:
            if op == "matmul":
                return ops.matmul(
                    *args,
                    activation=case.get("activation"),
                    backend=dispatch,
                    variant=variant,
                    config=config,
                )
            if op == "attention":
                return ops.attention(
                    *args, causal=case.get("causal", False), backend=dispatch, config=config
                )
            if op == "conv2d":
                return ops.conv2d(
                    *args,
                    stride=case.get("stride", 1),
                    padding=case.get("padding", 0),
                    backend=dispatch,
                    config=config,
                )
            if op in {"cnn", "transformer"}:
                return model(*args, backend=dispatch, plan=plan)
            return getattr(ops, op)(*args, backend=dispatch)

    if backend in {"compile", "compile_mm_compat"}:
        return compile_strict(invoke, mm_compat=backend == "compile_mm_compat")
    if plan is not None:
        invoke.kernel_plan = plan
    return invoke


def oracle(case, args, model):
    """Untimed FP32 oracle from the *quantized input values*, never fresh inputs."""
    converted = tuple(t.float() if t is not None else None for t in args)
    if case["op"] == "attention":
        return references.attention_explicit(*converted, causal=case.get("causal", False))
    if model is not None:
        # Layernorm/residual boundaries round in the real module: match those boundaries.
        return make_callable(case, args, model, "torch")()
    return make_callable(case, converted, None, "torch")()
