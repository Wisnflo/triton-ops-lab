"""Verify raw study evidence before using summaries or declaring completion."""

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from ..benchmarks.config import identity
from ..benchmarks.selection import GEMM_ABLATIONS, RULE_BACKENDS
from .dispatch import load_dispatch_supplement, verify_dispatch_observation


def read_json(path):
    return json.loads(Path(path).read_text())


def audit_study(path, purpose):
    path = Path(path)
    state = read_json(path / "suite-status.json")
    if (
        not state["complete"]
        or state["suite"]["purpose"] != purpose
        or len(state["jobs"]) != len(state["suite"]["jobs"])
    ):
        raise ValueError("Study is incomplete or has the wrong purpose")
    rows = []
    jobs = []
    env = state["environment"]
    dispatch = load_dispatch_supplement(path, state)
    for planned, finished in zip(state["suite"]["jobs"], state["jobs"], strict=True):
        if (
            planned != finished["job"]
            or finished["status"] != "finished"
            or finished["exit_code"] != 0
        ):
            raise ValueError("A planned job did not finish cleanly")
        group = path / planned["label"] / "raw"
        files = list(group.glob("*/metrics.csv"))
        if len(files) != 1:
            raise ValueError("Missing or duplicate raw measurements")
        file = files[0]
        meta = read_json(file.parent / "metadata.json")
        cfg = meta["config"]
        if (
            not meta["complete"]
            or meta["kind"] != "gpu_measured"
            or not meta["environment"]["gpu"]
            or meta["environment"]["git_dirty"]
        ):
            raise ValueError("Unverified measurement provenance")
        if any(
            meta["environment"][k] != env[k]
            for k in ("git_commit", "source_sha256", "lock_sha256", "gpu", "driver")
        ):
            raise ValueError("Source or hardware changed within one study")
        expected_mode = (
            "cuda-graph-replay-v2"
            if planned["mode"] == "cuda_graph"
            else "cuda-events-with-wall-v2"
        )
        if (
            meta["protocol"] != expected_mode
            or cfg["timing"].get("mode", "cuda_event") != planned["mode"]
        ):
            raise ValueError("Timing protocols differ")
        preflight = read_json(file.parent / "preflight.json")
        if not preflight or not all(x["passed"] for x in preflight):
            raise ValueError("Preflight did not pass")
        values = list(csv.DictReader(file.open()))
        expected = {(c["id"], b) for c in cfg["cases"] for b in cfg["backends"]}
        if (
            len(values) != len(expected)
            or {(v["case_id"], v["backend"]) for v in values} != expected
        ):
            raise ValueError("Measured rows do not cover the configured cases/backends")
        by_id = {c["id"]: c for c in cfg["cases"]}
        for v in values:
            if v["status"] != "passed":
                raise ValueError(
                    f"Nonpassed row: {planned['label']} {v['case_id']} {v['backend']} {v['reason']}"
                )
            if (
                json.loads(v["shape"]) != by_id[v["case_id"]]["shape"]
                or v["dtype"] != by_id[v["case_id"]]["dtype"]
            ):
                raise ValueError("Row shape/dtype mismatch")
            for field in ("samples_ms", "wall_samples_ms"):
                samples = json.loads(v[field])
                if len(samples) != cfg["timing"]["samples"] or any(
                    not math.isfinite(x) or x <= 0 for x in samples
                ):
                    raise ValueError("Invalid timing samples")
                column = "median_ms" if field == "samples_ms" else "wall_median_ms"
                if not math.isclose(
                    statistics.median(samples), float(v[column]), rel_tol=1e-10, abs_tol=1e-12
                ):
                    raise ValueError("Timing median does not match raw samples")
            attempts = json.loads(v["candidates"])
            if not attempts or any(
                a["status"] != "passed" or not a["accuracy"]["passed"] for a in attempts
            ):
                raise ValueError("An intended candidate did not pass correctness")
            chosen = json.loads(v["selected_config"] or "null")
            if chosen not in [a["config"] for a in attempts]:
                raise ValueError("Selected configuration was not checked")
            if v["backend"] in GEMM_ABLATIONS and chosen != GEMM_ABLATIONS[v["backend"]]:
                raise ValueError("Ablation row changed the wrong factor")
            if v["backend"].startswith("compile"):
                audit = json.loads(v["compile_audit"])
                if audit["graph_count"] != 1:
                    raise ValueError("Compiled baseline is not a single graph")
            kernels = json.loads(v["kernel_names"])
            if not kernels:
                kernels = verify_dispatch_observation(
                    dispatch, file, meta, planned, by_id[v["case_id"]], v
                )
            if any("profiling unavailable" in k for k in kernels):
                raise ValueError("Actual dispatched kernels were not recorded")
            if v["backend"].startswith("sdpa_"):
                kind = {
                    "sdpa_math": "attention_math",
                    "sdpa_flash": "flash_attention",
                    "sdpa_efficient": "efficient_attention",
                }[v["backend"]]
                if not any(kind in k for k in kernels):
                    raise ValueError("Forced SDPA backend was not observed")
            if v["backend"] in RULE_BACKENDS:
                if meta.get("frozen_policy", {}).get("sha256") != state.get("frozen_policy_sha256"):
                    raise ValueError("Policy identity mismatch")
                decisions = json.loads(v["policy_decisions"])
                if decisions["policy_sha256"] != state["frozen_policy_sha256"]:
                    raise ValueError("Policy decision identity mismatch")
            rows.append(
                dict(job=planned, case=by_id[v["case_id"]], row=v, metadata=meta, raw_csv=str(file))
            )
        if "cache_group" in planned:
            if planned["cache_state"] == "cold" and finished["cache_files_before"] != 0:
                raise ValueError("Cold cache was populated")
            if planned["cache_state"] == "warm" and finished["cache_files_before"] <= 0:
                raise ValueError("Warm cache was empty")
            if finished["cache_files_after"] <= 0:
                raise ValueError("No compiler cache artifacts were generated")
        jobs.append(
            dict(
                label=planned["label"],
                rows=len(values),
                csv_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            )
        )
    if dispatch is not None and dispatch["used"] != set(dispatch["entries"]):
        raise ValueError("Dispatch supplement contains observations for nonmissing rows")
    return dict(
        path=str(path),
        state=state,
        rows=rows,
        jobs=jobs,
        dispatch_supplement=None
        if dispatch is None
        else {k: dispatch[k] for k in ("manifest", "manifest_sha256", "note")}
        | {"row_count": len(dispatch["used"])},
    )


def summarize_study(audited):
    # Keep sessions/protocols/dtypes separate; retain every case and regression.
    grouped = {}
    for item in audited["rows"]:
        key = (
            item["job"]["mode"],
            item["job"].get("cache_state"),
            item["case"]["dtype"],
            identity(item["case"]),
            item["row"]["backend"],
        )
        grouped.setdefault(key, []).append(item["row"])
    result = []
    for (mode, cache_state, dtype, case, backend), measured in sorted(grouped.items()):
        samples = [float(x["median_ms"]) for x in measured]
        details = {}
        for field in (
            "wall_median_ms",
            "peak_allocated_bytes",
            "extra_peak_bytes",
            "first_call_ms",
            "construction_ms",
            "selection_ms",
            "tuning_ms",
            "p20_ms",
            "p80_ms",
            "relative_l2",
        ):
            if all(x.get(field) not in (None, "") for x in measured):
                details[field] = [float(x[field]) for x in measured]
        variability = []
        for row in measured:
            if row.get("samples_ms"):
                raw = json.loads(row["samples_ms"])
                variability.append(
                    dict(
                        samples=len(raw),
                        mean_ms=statistics.mean(raw),
                        population_stdev_ms=statistics.pstdev(raw),
                        coefficient_of_variation=statistics.pstdev(raw) / statistics.mean(raw),
                    )
                )
        result.append(
            dict(
                protocol=mode,
                cache_state=cache_state,
                dtype=dtype,
                case=json.loads(case),
                backend=backend,
                repetitions=len(samples),
                run_medians_ms=samples,
                median_ms=statistics.median(samples),
                min_ms=min(samples),
                max_ms=max(samples),
                run_details=details,
                within_run_variability=variability,
            )
        )
    return dict(
        environment=audited["state"]["environment"],
        purpose=audited["state"]["suite"]["purpose"],
        source_commit=audited["state"]["environment"]["git_commit"],
        job_count=len(audited["jobs"]),
        row_count=len(audited["rows"]),
        jobs=audited["jobs"],
        dispatch_supplement=audited.get("dispatch_supplement"),
        summary=result,
    )


def comparisons(audited):
    """Paired run ratios, without pooling cases, dtypes, protocols or sessions."""
    groups = {}
    for item in audited["rows"]:
        key = (
            item["job"]["mode"],
            item["job"].get("cache_state"),
            item["case"]["dtype"],
            identity(item["case"]),
        )
        groups.setdefault(key, {}).setdefault(item["row"]["backend"], []).append(item)
    result = []
    for (mode, cache_state, dtype, case), backends in sorted(groups.items()):
        for baseline in (
            "torch",
            "compile",
            "compile_mm_compat",
            "sdpa_flash",
            "sdpa_efficient",
            "triton_fixed",
            "triton_prepared",
        ):
            if baseline not in backends:
                continue
            for backend, items in sorted(backends.items()):
                if backend == baseline:
                    continue
                refs = {x["job"]["repetition"]: x for x in backends[baseline]}
                pairs = [(refs[x["job"]["repetition"]], x) for x in items]
                ratios = [
                    float(a["row"]["median_ms"]) / float(b["row"]["median_ms"]) for a, b in pairs
                ]
                result.append(
                    dict(
                        protocol=mode,
                        cache_state=cache_state,
                        dtype=dtype,
                        case=json.loads(case),
                        baseline=baseline,
                        backend=backend,
                        run_speedups=ratios,
                        median_speedup=statistics.median(ratios),
                        latency_reduction_percent=100 * (1 - 1 / statistics.median(ratios)),
                        throughput_increase_percent=100 * (statistics.median(ratios) - 1),
                        run_wall_speedups=[
                            float(a["row"]["wall_median_ms"]) / float(b["row"]["wall_median_ms"])
                            for a, b in pairs
                        ]
                        if all(
                            a["row"].get("wall_median_ms") and b["row"].get("wall_median_ms")
                            for a, b in pairs
                        )
                        else None,
                        paired_extra_peak_bytes=[
                            dict(
                                baseline=int(a["row"]["extra_peak_bytes"]),
                                backend=int(b["row"]["extra_peak_bytes"]),
                            )
                            for a, b in pairs
                        ]
                        if all(
                            a["row"].get("extra_peak_bytes") and b["row"].get("extra_peak_bytes")
                            for a, b in pairs
                        )
                        else None,
                        note="Positive latency reduction means faster; negative values are regressions.",
                    )
                )
    return result


def replication_consistency(primary, replica):
    """Match paired ratios while retaining each independent session and repetition."""
    if (
        primary["state"]["suite"]["purpose"] != "evaluation"
        or replica["state"]["suite"]["purpose"] != "replication"
    ):
        raise ValueError("Evaluation and independent replication evidence are required")

    def indexed(audit):
        result = {}
        for item in comparisons(audit):
            key = (
                item["protocol"],
                item["cache_state"],
                item["dtype"],
                json.dumps(item["case"], sort_keys=True),
                item["baseline"],
                item["backend"],
            )
            if key in result:
                raise ValueError("Duplicate cross-session comparison identity")
            result[key] = item
        return result

    first, second = indexed(primary), indexed(replica)
    if not first or set(first) != set(second):
        raise ValueError("Independent sessions do not cover the same comparisons")
    rows = []
    for key in sorted(first):
        a, b = first[key], second[key]
        sa, sb = a["median_speedup"], b["median_speedup"]
        rows.append(
            dict(
                **{
                    k: a[k]
                    for k in ("protocol", "cache_state", "dtype", "case", "baseline", "backend")
                },
                primary=dict(
                    median_speedup=sa,
                    run_speedups=a["run_speedups"],
                    latency_reduction_percent=a["latency_reduction_percent"],
                ),
                replication=dict(
                    median_speedup=sb,
                    run_speedups=b["run_speedups"],
                    latency_reduction_percent=b["latency_reduction_percent"],
                ),
                both_session_medians_faster=sa > 1 and sb > 1,
                both_session_medians_slower=sa < 1 and sb < 1,
                median_direction_agrees=(sa > 1) == (sb > 1) and (sa < 1) == (sb < 1),
                every_repetition_faster_in_both_sessions=all(
                    x > 1 for x in a["run_speedups"] + b["run_speedups"]
                ),
                every_repetition_slower_in_both_sessions=all(
                    x < 1 for x in a["run_speedups"] + b["run_speedups"]
                ),
            )
        )
    return dict(
        primary_environment=primary["state"]["environment"],
        replication_environment=replica["state"]["environment"],
        comparisons=rows,
        note="Per-case sign agreement of observed ratios, not a statistical significance test. Sessions, protocols and repetitions are not pooled; regressions and direction changes remain visible.",
    )


def cost_amortization(audited, policy):
    if audited["state"]["suite"]["purpose"] != "cost":
        raise ValueError("Controlled cost suite required")
    groups = {}
    for item in audited["rows"]:
        key = (item["case"]["op"], item["job"]["repetition"], item["job"]["cache_state"])
        groups.setdefault(key, {})[item["row"]["backend"]] = item
    result = []
    for (op, rep, state), backends in sorted(groups.items()):
        base = "triton_prepared" if op == "matmul" else "triton_fixed"
        baseline = backends[base]

        def startup(item):
            row = item["row"]
            # Search time already includes construction, first call, correctness and candidate timing.
            load = item["metadata"].get("frozen_policy", {}).get("load_ms", 0)
            if "search" in row["backend"] or row["backend"] == "triton_pruned":
                return float(row["tuning_ms"]) + load
            return (
                sum(float(row[k]) for k in ("selection_ms", "construction_ms", "first_call_ms"))
                + load
            )

        for backend, item in sorted(backends.items()):
            if backend == base:
                continue
            saved = float(baseline["row"]["median_ms"]) - float(item["row"]["median_ms"])
            overhead = max(0, startup(item) - startup(baseline))
            offline = (
                policy.data["offline_development_search_ms"] + policy.data["offline_fitting_ms"]
            )
            full_suite = (
                policy.data["development_suite_elapsed_seconds"] * 1000
                + policy.data["offline_fitting_ms"]
            )
            rule = backend in RULE_BACKENDS
            result.append(
                dict(
                    op=op,
                    repetition=rep,
                    cache_state=state,
                    backend=backend,
                    baseline=base,
                    baseline_startup_ms=startup(baseline),
                    startup_ms=startup(item),
                    saved_per_invocation_ms=saved,
                    incremental_startup_ms=overhead,
                    break_even_invocations=math.ceil(overhead / saved) if saved > 0 else None,
                    offline_acquisition_and_fit_ms=offline if rule else 0,
                    break_even_with_donor_search_cost=math.ceil((overhead + offline) / saved)
                    if saved > 0 and rule
                    else None,
                    full_development_suite_and_fit_ms=full_suite if rule else 0,
                    break_even_with_full_development_suite=math.ceil(
                        (overhead + full_suite) / saved
                    )
                    if saved > 0 and rule
                    else None,
                    note="Graph steady-state break-even only. Donor-search sum excludes acquisition preflight/compilation in earlier jobs; full-suite bound includes controls and ablation and is conservatively charged to one workload. None means no measured steady-state gain. Process launch/preflight/driver caches excluded from per-backend startup.",
                )
            )
    return result


def ablation_comparisons(audited):
    groups = {}
    for item in audited["rows"]:
        if item["row"]["backend"] not in GEMM_ABLATIONS:
            continue
        key = (
            item["job"]["mode"],
            item["case"]["dtype"],
            identity(item["case"]),
            item["job"]["repetition"],
        )
        groups.setdefault(key, {})[item["row"]["backend"]] = item
    result = []
    for (mode, dtype, case, rep), rows in sorted(groups.items()):
        if set(rows) != set(GEMM_ABLATIONS):
            raise ValueError("Incomplete orthogonal ablation")
        base = rows["triton_ablation_base"]["row"]
        for name, item in sorted(rows.items()):
            if name == "triton_ablation_base":
                continue
            factor = [
                k
                for k in GEMM_ABLATIONS[name]
                if GEMM_ABLATIONS[name][k] != GEMM_ABLATIONS["triton_ablation_base"][k]
            ]
            if len(factor) != 1:
                raise ValueError("Ablation changes multiple factors")
            speedup = float(base["median_ms"]) / float(item["row"]["median_ms"])
            result.append(
                dict(
                    protocol=mode,
                    dtype=dtype,
                    case=json.loads(case),
                    repetition=rep,
                    backend=name,
                    factor=factor[0],
                    baseline_config=GEMM_ABLATIONS["triton_ablation_base"],
                    config=GEMM_ABLATIONS[name],
                    speedup=speedup,
                    latency_reduction_percent=100 * (1 - 1 / speedup),
                )
            )
    return result
