"""Fit a committed policy from a complete development study, never from eval.

All donor measurements and compilation identities are retained in the external
study archive. The small output stores sufficient statistics plus their hashes.
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from triton_ops_lab.artifacts import project_root, write_json
from triton_ops_lab.analysis.study import audit_study
from triton_ops_lab.benchmarks.config import assert_disjoint, identity, load_config
from triton_ops_lab.benchmarks.selection import candidates
from triton_ops_lab.policy import NEIGHBORS, SHORTLIST, POLICY_SCHEMA, config_key, features


def fit(study_path, output):
    started = time.perf_counter()
    path = Path(study_path)
    state = json.loads((path / "suite-status.json").read_text())
    if not state["complete"] or state["suite"]["purpose"] != "development":
        raise ValueError("A complete development suite is required; eval/replication is forbidden")
    audited = audit_study(path, "development")
    search_acquisition_ms = sum(
        float(item["row"]["tuning_ms"])
        for item in audited["rows"]
        if item["job"]["mode"] == "cuda_graph" and item["row"]["backend"] == "triton_search"
    )
    holdout = load_config(project_root() / "configs/eval.json")
    grouped, pools, sources = defaultdict(list), {}, []
    measured_environment = None
    for job in state["jobs"]:
        if job["job"]["mode"] != "cuda_graph":
            continue
        run = path / job["job"]["label"] / "raw"
        metrics = list(run.glob("*/metrics.csv"))
        if len(metrics) != 1:
            raise ValueError("Missing or duplicate development run")
        f = metrics[0]
        meta = json.loads((f.parent / "metadata.json").read_text())
        if (
            not meta["complete"]
            or meta["environment"]["git_dirty"]
            or meta["environment"]["git_commit"] != state["environment"]["git_commit"]
        ):
            raise ValueError("Unverified development provenance")
        assert_disjoint(meta["config"], holdout)
        by_id = {c["id"]: c for c in meta["config"]["cases"]}
        for row in csv.DictReader(f.open()):
            if row["backend"] != "triton_search":
                continue
            case = by_id[row["case_id"]]
            if measured_environment is None:
                measured_environment = meta["environment"]
            elif measured_environment != meta["environment"]:
                raise ValueError("Development measurement environments differ")
            if row["status"] != "passed":
                raise ValueError("Correctness/compilation failure must be resolved before freezing")
            attempts = json.loads(row["candidates"])
            if any(x["status"] != "passed" for x in attempts):
                raise ValueError("Every intended development candidate must pass before freeze")
            key = (case["op"], case["dtype"], identity(case))
            pool = candidates(case["op"], case["shape"], "search", 6)
            if [x["config"] for x in attempts] != pool:
                raise ValueError("Candidate-pool version mismatch")
            pools[case["op"] + ":" + case["dtype"]] = pool
            grouped[key].append(
                (case, {config_key(x["config"]): float(x["median_ms"]) for x in attempts})
            )
        sources.append(
            dict(
                run=f.parent.name,
                metrics_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                source_commit=meta["environment"]["git_commit"],
                source_sha256=meta["environment"]["source_sha256"],
            )
        )
    data = dict(
        schema_version=POLICY_SCHEMA,
        purpose="development_only",
        policy_id="nearest-shape-v1",
        frozen_at=datetime.now(timezone.utc).isoformat(),
        neighbors=NEIGHBORS,
        shortlist=SHORTLIST,
        training_environment=measured_environment,
        offline_development_search_ms=search_acquisition_ms,
        development_suite_elapsed_seconds=sum(j["seconds"] for j in state["jobs"]),
        acquisition_cost_note="Search acquisition is the sum of all three Graph donor-search runs; suite time also includes controls and ablation. CPU fitting alone excludes data acquisition.",
        selection_metric="cuda-graph-replay-v2 candidate medians normalized within each donor",
        source_measurements=sources,
        feature_implementation_sha256=hashlib.sha256(
            (project_root() / "src/triton_ops_lab/policy.py").read_bytes()
        ).hexdigest(),
        groups={},
    )
    for (op, dtype, sample_identity), values in sorted(grouped.items()):
        if len(values) != 3:
            raise ValueError("Require three Graph development runs per donor")
        pool_key = op + ":" + dtype
        pool = pools[pool_key]
        medians = [statistics.median(v[config_key(cfg)] for _, v in values) for cfg in pool]
        if any(not math.isfinite(x) or x <= 0 for x in medians):
            raise ValueError("Invalid development measurements")
        best = min(medians)
        group = data["groups"].setdefault(pool_key, dict(pool=pool, samples=[]))
        group["samples"].append(
            dict(
                identity=sample_identity,
                features=features(values[0][0]),
                candidate_medians_ms=medians,
                relative_costs=[x / best for x in medians],
            )
        )
    if set(data["groups"]) != {
        op + ":" + dtype
        for op in ("matmul", "attention", "conv2d")
        for dtype in ("float16", "bfloat16")
    }:
        raise ValueError("Missing an intended operator/dtype training group")
    if Path(output).exists():
        raise ValueError("Refusing to replace an already frozen policy")
    data["offline_fitting_ms"] = (time.perf_counter() - started) * 1000
    write_json(output, data)
    print(output, hashlib.sha256(Path(output).read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study")
    parser.add_argument("--output", default="configs/policy-v1.json")
    args = parser.parse_args()
    fit(args.study, args.output)
