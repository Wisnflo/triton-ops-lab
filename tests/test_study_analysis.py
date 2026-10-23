from types import SimpleNamespace

from triton_ops_lab.analysis.study import comparisons, cost_amortization, summarize_study


def study():
    rows = []
    for state, scale in [("cold", 1), ("warm", 2)]:
        for backend, time in [("triton_prepared", 10), ("triton_prepared_rule_search", 5)]:
            rows.append(
                dict(
                    job=dict(mode="cuda_graph", cache_state=state, repetition=0),
                    case=dict(op="matmul", shape=[64, 64, 64], dtype="float16"),
                    row=dict(
                        backend=backend,
                        median_ms=str(time * scale),
                        tuning_ms="100",
                        selection_ms="1",
                        construction_ms="2",
                        first_call_ms="3",
                    ),
                    metadata=dict(frozen_policy=dict(load_ms=4)),
                )
            )
    return dict(
        state=dict(suite=dict(purpose="cost"), environment=dict(git_commit="test")),
        jobs=[],
        rows=rows,
    )


def test_cache_states_are_never_pooled_and_cost_is_not_double_counted():
    data = study()
    summary = summarize_study(data)["summary"]
    assert len(summary) == 4 and all(x["repetitions"] == 1 for x in summary)
    pairs = comparisons(data)
    assert len(pairs) == 2 and all(x["median_speedup"] == 2 for x in pairs)
    costs = cost_amortization(
        data,
        SimpleNamespace(
            data=dict(
                offline_development_search_ms=900,
                offline_fitting_ms=100,
                development_suite_elapsed_seconds=2,
            )
        ),
    )
    cold = next(x for x in costs if x["cache_state"] == "cold")
    assert cold["startup_ms"] == 104  # Search already includes setup, first call and selection.
    assert cold["baseline_startup_ms"] == 10
    assert cold["break_even_invocations"] == 19
    assert cold["break_even_with_donor_search_cost"] == 219
    assert cold["break_even_with_full_development_suite"] == 439


def test_regression_never_claims_break_even():
    data = study()
    for item in data["rows"]:
        if "rule" in item["row"]["backend"]:
            item["row"]["median_ms"] = "100"
    costs = cost_amortization(
        data,
        SimpleNamespace(
            data=dict(
                offline_development_search_ms=900,
                offline_fitting_ms=100,
                development_suite_elapsed_seconds=2,
            )
        ),
    )
    assert all(x["break_even_invocations"] is None for x in costs)
    assert all(x["latency_reduction_percent"] < 0 for x in comparisons(data))
