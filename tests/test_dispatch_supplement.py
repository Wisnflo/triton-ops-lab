import hashlib
import json

import pytest

from triton_ops_lab.analysis.dispatch import (
    load_dispatch_supplement,
    verify_dispatch_observation,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def evidence(tmp_path):
    study = tmp_path / "study"
    file = study / "job/raw/run/metrics.csv"
    file.parent.mkdir(parents=True)
    file.write_text("original measurement,unchanged\n")
    case = dict(id="long-bf16", op="attention", shape=[1, 2, 2048, 128], dtype="bfloat16")
    row = dict(backend="triton_rule", selected_config='{"bm": 32}', kernel_names="[]")
    metadata = dict(config=dict(seed=2026), environment=dict(git_commit="frozen"))
    (file.parent / "metadata.json").write_text(json.dumps(metadata))
    state = dict(ended=1, suite=dict(purpose="evaluation"), frozen_policy_sha256="policy")
    (study / "suite-status.json").write_text(json.dumps(state))
    root = tmp_path / "dispatch-supplement"
    root.mkdir()
    (root / "capture.py").write_text("# untimed observation\n")
    trace = root / "trace.json"
    trace.write_text(
        json.dumps(
            dict(
                traceEvents=[
                    dict(name="_attention", cat="kernel", ph="X", dur=10, args=dict(device=0))
                ]
            )
        )
    )
    plan = dict(
        job="job",
        case=case,
        backend=row["backend"],
        seed=2026,
        selected_config=dict(bm=32),
        original_metrics=str(file.relative_to(study)),
        original_metrics_sha256=sha(file),
        original_metadata_sha256=sha(file.parent / "metadata.json"),
    )
    attempt = dict(
        trace="trace.json",
        trace_sha256=sha(trace),
        kernel_names=["_attention"],
        accuracy=dict(passed=True),
    )
    data = dict(
        schema="missing-dispatch-observation-v1",
        complete=True,
        purpose="evaluation",
        suite_status_sha256=sha(study / "suite-status.json"),
        policy_sha256="policy",
        script="capture.py",
        script_sha256=sha(root / "capture.py"),
        planned_at=2,
        plan=[plan],
        entries=[
            dict(
                **plan,
                environment=metadata["environment"],
                accuracy=dict(passed=True),
                attempts=[attempt],
            )
        ],
    )
    manifest = root / "kernel-observation-supplement.json"
    manifest.write_text(json.dumps(data))
    return study, state, file, metadata, case, row, data, manifest


def validate(evidence):
    study, state, file, metadata, case, row, _, _ = evidence
    supplement = load_dispatch_supplement(study, state)
    result = verify_dispatch_observation(supplement, file, metadata, dict(label="job"), case, row)
    assert supplement["used"] == {("job", case["id"], row["backend"])}
    return result


def test_separate_trace_preserves_original_row(evidence):
    original = dict(evidence[5])
    assert validate(evidence) == ["_attention"]
    assert evidence[5] == original and original["kernel_names"] == "[]"


@pytest.mark.parametrize("change", ["measurement", "config", "environment", "trace", "later"])
def test_supplement_cannot_replace_identity_or_choose_a_later_success(evidence, change):
    _, _, file, _, _, row, data, manifest = evidence
    if change == "measurement":
        file.write_text("changed timing\n")
    elif change == "config":
        row["selected_config"] = '{"bm": 64}'
    elif change == "environment":
        data["entries"][0]["environment"] = dict(git_commit="changed")
    elif change == "trace":
        (manifest.parent / "trace.json").write_text("{}")
    else:
        data["entries"][0]["attempts"] *= 2
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        validate(evidence)


def test_empty_observation_without_supplement_is_rejected(evidence):
    _, _, file, metadata, case, row, _, manifest = evidence
    manifest.unlink()
    assert load_dispatch_supplement(evidence[0], evidence[1]) is None
    with pytest.raises(ValueError, match="not recorded"):
        verify_dispatch_observation(None, file, metadata, dict(label="job"), case, row)


def test_supplement_cannot_reclassify_an_existing_observation(evidence):
    evidence[5]["kernel_names"] = '["profiling unavailable: failure"]'
    with pytest.raises(ValueError):
        validate(evidence)
