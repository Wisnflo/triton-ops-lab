import importlib.util

import pytest
import torch

from triton_ops_lab.artifacts import project_root
from triton_ops_lab.benchmarks.config import assert_disjoint, identity, load_config
from triton_ops_lab.benchmarks.selection import GEMM_ABLATIONS, candidates
from triton_ops_lab.ops import _validate_config


def test_ablation_changes_one_factor_and_uses_valid_tiles():
    base = GEMM_ABLATIONS["triton_ablation_base"]
    assert len(GEMM_ABLATIONS) == 13
    for label, cfg in GEMM_ABLATIONS.items():
        _validate_config(cfg, "matmul", "triton")
        assert sum(cfg[k] != base[k] for k in base) == (0 if label.endswith("_base") else 1)
    with pytest.raises(ValueError):
        _validate_config({**base, "group": 3}, "matmul", "triton")
    with pytest.raises(ValueError):
        _validate_config(base, "conv2d", "triton")


def test_default_spelling_does_not_hide_overlap():
    case = dict(id="a", op="matmul", shape=[31, 45, 67], dtype="float16")
    explicit = dict(case, id="b", dtype="bfloat16", bias=False, activation=None, transpose=False)
    assert identity(case) == identity(explicit)
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint({"cases": [case]}, {"cases": [explicit]})


def test_development_suite_excludes_evaluation_and_all_jobs_parse():
    root = project_root()
    spec = importlib.util.spec_from_file_location("run_study", root / "scripts/run_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = module.load_suite(root / "configs/study/development-suite.json")
    assert len(suite["jobs"]) == 20
    for job in suite["jobs"]:
        cfg = load_config(root / job["config"])
        assert_disjoint(cfg, load_config(root / "configs/eval.json"))
    assert candidates("conv2d", [1, 8, 16, 16, 16, 1])[0]["bn"] == 32


def test_formal_suite_cannot_drop_cases_or_fake_warm_caches(tmp_path):
    import json

    root = project_root()
    spec = importlib.util.spec_from_file_location("formal_study", root / "scripts/run_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for purpose, count in [("evaluation", 30), ("replication", 30), ("cost", 84)]:
        suite = module.load_suite(root / f"configs/study/{purpose}-suite.json")
        assert len(suite["jobs"]) == count
    suite = module.load_suite(root / "configs/study/evaluation-suite.json")
    suite["jobs"] = suite["jobs"][1:]
    bad = tmp_path / "missing.json"
    bad.write_text(json.dumps(suite))
    with pytest.raises(ValueError, match="all heldout"):
        module.load_suite(bad)
    suite = module.load_suite(root / "configs/study/cost-suite.json")
    suite["jobs"] = suite["jobs"][1:]
    bad.write_text(json.dumps(suite))
    with pytest.raises(ValueError, match="must follow"):
        module.load_suite(bad)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("transpose", [False, True])
def test_group_and_one_factor_kernels(dtype, transpose):
    from triton_ops_lab.benchmarks.cases import make_callable, make_inputs, oracle
    from triton_ops_lab.validation import compare

    case = dict(
        id="ablation-correctness",
        op="matmul",
        shape=[73, 37, 41],
        dtype=dtype,
        transpose=transpose,
        bias=True,
        activation="gelu",
    )
    args, model = make_inputs(case, "cuda")
    expected = oracle(case, args, model)
    for backend, cfg in GEMM_ABLATIONS.items():
        public = make_callable(case, args, model, backend, cfg)
        prepared = make_callable(case, args, model, "triton_prepared", cfg)
        for fn in (public, prepared):
            assert compare(fn(), expected, "matmul", dtype)["passed"], (backend, dtype, transpose)
    torch.cuda.synchronize()
