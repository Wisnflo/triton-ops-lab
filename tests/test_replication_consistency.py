import pytest

from triton_ops_lab.analysis.study import replication_consistency


def measured(purpose, times, mode="cuda_graph"):
    rows = []
    for rep, time in enumerate(times):
        for backend, value in (("torch", 1), ("triton_rule", time)):
            rows.append(
                dict(
                    job=dict(mode=mode, repetition=rep),
                    case=dict(op="matmul", shape=[2, 2, 2], dtype="float16"),
                    row=dict(backend=backend, median_ms=str(value)),
                )
            )
    return dict(
        state=dict(suite=dict(purpose=purpose), environment=dict(session=purpose)), rows=rows
    )


def test_direction_change_retains_both_sessions():
    data = replication_consistency(
        measured("evaluation", [0.5, 0.5, 0.5]), measured("replication", [2, 2, 2])
    )
    row = data["comparisons"][0]
    assert row["primary"]["run_speedups"] == [2, 2, 2]
    assert row["replication"]["run_speedups"] == [0.5, 0.5, 0.5]
    assert not row["median_direction_agrees"]
    assert not row["both_session_medians_faster"]


def test_median_gain_is_distinct_from_all_repetitions_gain():
    row = replication_consistency(
        measured("evaluation", [0.5, 0.5, 0.5]), measured("replication", [0.5, 0.5, 2])
    )["comparisons"][0]
    assert row["both_session_medians_faster"]
    assert not row["every_repetition_faster_in_both_sessions"]


def test_protocols_cannot_be_matched_across_sessions():
    with pytest.raises(ValueError, match="same comparisons"):
        replication_consistency(
            measured("evaluation", [0.5]), measured("replication", [0.5], "cuda_event")
        )
