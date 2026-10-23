from copy import deepcopy
import importlib.util
import json

import pytest
import torch

from triton_ops_lab.policy import FrozenPolicy, features
from triton_ops_lab.artifacts import project_root
from triton_ops_lab.benchmarks.config import identity
from triton_ops_lab.benchmarks.selection import candidates


def fixture():
    case = dict(op="matmul", shape=[64, 128, 32], dtype="float16")
    pool = [dict(bm=x, bn=32, bk=32, warps=4, stages=2) for x in (32, 64, 128)]
    data = dict(
        schema_version=1,
        purpose="development_only",
        neighbors=3,
        shortlist=2,
        groups={
            "matmul:float16": dict(
                pool=pool,
                samples=[
                    dict(identity="donor", features=features(case), relative_costs=[2.0, 1.0, 1.5])
                ],
            )
        },
    )
    return case, data


def test_policy_uses_shape_and_semantics_not_case_names():
    case, data = fixture()
    policy = FrozenPolicy(data)
    assert policy.choose({**case, "id": "arbitrary-evaluation-label"})["bm"] == 64
    assert policy.choose({**case, "id": "training-label"}) == policy.choose(case)
    assert [x["bm"] for x in policy.shortlist(case)] == [64, 128, 32]
    before = deepcopy(data)
    policy.rank({**case, "shape": [65, 129, 33]})
    assert before == data  # Inference never updates from evaluation queries.


def test_policy_rejects_uncovered_inputs_and_wrong_provenance():
    case, data = fixture()
    with pytest.raises(ValueError, match="coverage"):
        FrozenPolicy(data).choose({**case, "dtype": "bfloat16"})
    with pytest.raises(ValueError, match="development"):
        FrozenPolicy({**data, "purpose": "evaluation"})
    with pytest.raises(ValueError, match="positive"):
        features({**case, "shape": [0, 128, 32]})


def test_freeze_refuses_evaluation_measurements_and_modified_features(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "freeze_policy", project_root() / "scripts/freeze_policy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "suite-status.json").write_text(
        json.dumps(dict(complete=True, suite=dict(purpose="evaluation")))
    )
    with pytest.raises(ValueError, match="eval/replication"):
        module.fit(tmp_path, tmp_path / "policy.json")
    case, data = fixture()
    (tmp_path / "policy.json").write_text(
        json.dumps({**data, "feature_implementation_sha256": "invalid"})
    )
    with pytest.raises(ValueError, match="implementation"):
        FrozenPolicy.load(tmp_path / "policy.json")


def test_prepared_policy_label_uses_the_prepared_contract(monkeypatch):
    # CPU routing test only; real GPU correctness remains a separate gate.
    from triton_ops_lab.benchmarks.cases import make_callable
    from triton_ops_lab import ops

    case, data = fixture()
    calls = []

    def prepared(*args, **kwargs):
        calls.append(kwargs)
        return lambda: "prepared-output"

    monkeypatch.setattr(ops, "prepare_matmul", prepared)
    fn = make_callable(case, (), None, "triton_prepared_rule", policy=FrozenPolicy(data))
    assert fn() == "prepared-output" and calls[0]["config"]["bm"] == 64


def test_committed_policy_and_sealed_decisions_match():
    from triton_ops_lab.benchmarks.config import load_config

    root = project_root()
    policy = FrozenPolicy.load(root / "configs/policy-v1.json")
    sealed = json.loads((root / "configs/study/policy-decisions-v1.json").read_text())
    assert sealed["policy_sha256"] == policy.sha256 and len(policy.data["groups"]) == 6
    cases = load_config(root / "configs/eval.json")["cases"]
    assert [x["case"] for x in sealed["decisions"]] == cases
    for item in sealed["decisions"]:
        if "selected" in item["decision"]:
            assert policy.choose(item["case"]) == item["decision"]["selected"]
            assert policy.shortlist(item["case"]) == item["decision"]["shortlist"]
    assert all(len(v["samples"]) == 6 for v in policy.data["groups"].values())


@pytest.mark.parametrize("kind", ["transformer", "cnn"])
def test_module_plan_matches_actual_forward_operator_shapes(kind, monkeypatch):
    import triton_ops_lab.models as models

    seen = []

    class RecordingPolicy:
        def choose(self, case):
            seen.append(case)
            return candidates(case["op"], case["shape"])[0]

    model = models.TinyTransformer() if kind == "transformer" else models.TinyCNN()
    x = torch.randn([1, 7, 64] if kind == "transformer" else [1, 3, 9, 11])
    plan = model.make_plan(list(x.shape), "float32", RecordingPolicy())
    observed = []
    for op in ("matmul", "attention", "conv2d"):
        original = getattr(models, op)

        def wrap(*args, _op=op, _original=original, **kwargs):
            if _op == "matmul":
                c = dict(
                    op=_op,
                    shape=[args[0].shape[0], args[1].shape[1], args[0].shape[1]],
                    bias=len(args) > 2 and args[2] is not None,
                    activation=kwargs.get("activation"),
                )
            elif _op == "attention":
                c = dict(op=_op, shape=list(args[0].shape), causal=kwargs.get("causal", False))
            else:
                a, w = args
                c = dict(
                    op=_op,
                    shape=[*a.shape, w.shape[0], w.shape[-1]],
                    stride=kwargs.get("stride", 1),
                    padding=kwargs.get("padding", 0),
                )
            observed.append(c)
            return _original(*args, **kwargs)

        monkeypatch.setattr(models, op, wrap)
    assert torch.isfinite(model(x)).all()
    assert [identity(c) for c in seen] == [identity(c) for c in observed]
    assert len(plan["layers"]) == len(seen)
    with pytest.raises(ValueError, match="original input shape"):
        model(x, backend="torch", plan=plan)
    with pytest.raises(ValueError, match="original input shape"):
        model(x.half(), backend="triton", plan=plan)
