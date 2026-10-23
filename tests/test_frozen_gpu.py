"""Run only after the development-trained policy has been frozen and committed.

These tests validate numerical outputs and Graph replay; they never rank or time
evaluation candidates for training. Benchmark experiments remain separate.
"""

import pytest
import torch

from triton_ops_lab.artifacts import project_root
from triton_ops_lab.benchmarks.cases import make_callable, make_inputs, oracle
from triton_ops_lab.benchmarks.config import load_config
from triton_ops_lab.benchmarks.runner import measure
from triton_ops_lab.policy import FrozenPolicy
from triton_ops_lab.validation import compare


@pytest.mark.gpu
@pytest.mark.parametrize(
    "case",
    load_config(project_root() / "configs/eval.json")["cases"],
    ids=lambda c: c["op"] + "-" + c["id"],
)
def test_frozen_policy_outputs_and_replay(case):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    policy = FrozenPolicy.load(project_root() / "configs/policy-v1.json")
    args, model = make_inputs(case, "cuda", seed=2026)
    expected = oracle(case, args, model)
    if model is None:
        for cfg in policy.shortlist(case):
            for backend in (
                ["triton_rule", "triton_prepared_rule"]
                if case["op"] == "matmul"
                else ["triton_rule"]
            ):
                fn = make_callable(case, args, model, backend, cfg, policy=policy)
                accuracy = compare(fn(), expected, case["op"], case["dtype"])
                assert accuracy["passed"], (case, cfg, accuracy)
    fn = make_callable(case, args, model, "triton_rule", policy=policy)
    assert compare(fn(), expected, case["op"], case["dtype"])["passed"]
    measure(fn, dict(mode="cuda_graph", warmup=3, samples=3, repeats=3), expected, case)
