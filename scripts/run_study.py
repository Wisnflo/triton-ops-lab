"""Execute committed experiment jobs with live progress and raw evidence.

This orchestrates existing correctness-gated benchmarks. It never chooses a rule
from evaluation data or buys/manages cloud resources. Run artifacts stay outside
the repository. A started job is never restarted just because observation stops.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from triton_ops_lab.artifacts import environment, new_run, output_root, project_root, write_json
from triton_ops_lab.benchmarks.config import assert_disjoint, identity, load_config
from triton_ops_lab.ops import require_cuda
from triton_ops_lab.policy import FrozenPolicy


def load_suite(filename):
    suite = json.loads(Path(filename).read_text())
    if set(suite) != {"name", "purpose", "jobs"} or not suite["jobs"]:
        raise ValueError("Suite needs name, purpose and jobs")
    if suite["purpose"] not in {"development", "evaluation", "replication", "cost"}:
        raise ValueError("Invalid suite purpose")
    labels = set()
    cache_jobs = {}
    holdout = load_config(project_root() / "configs/eval.json")
    expected = {(identity(c), c["dtype"]) for c in holdout["cases"]}
    coverage = {}
    for job in suite["jobs"]:
        required = {"label", "config", "mode", "repetition"}
        if not required <= set(job) <= required | {"only_backend", "cache_group", "cache_state"}:
            raise ValueError("Invalid job fields")
        if job["label"] in labels or not all(c.isalnum() or c in "-_" for c in job["label"]):
            raise ValueError("Unique safe job labels required")
        if (
            job["mode"] not in {"cuda_event", "cuda_graph"}
            or type(job["repetition"]) is not int
            or not 0 <= job["repetition"] < 6
        ):
            raise ValueError("Invalid protocol/repetition")
        cfg = load_config(project_root() / job["config"])
        if suite["purpose"] in {"evaluation", "replication", "cost"}:
            keys = {(identity(c), c["dtype"]) for c in cfg["cases"]}
            if not keys <= expected:
                raise ValueError("Formal/cost jobs must use the predeclared holdout cases")
            if suite["purpose"] in {"evaluation", "replication"}:
                index = (job["mode"], job["repetition"])
                seen = coverage.setdefault(index, set())
                if seen & keys:
                    raise ValueError(
                        "A heldout case is repeated within a formal protocol/repetition"
                    )
                seen.update(keys)
        if "only_backend" in job and job["only_backend"] not in cfg["backends"]:
            raise ValueError("Backend filter is not part of the predeclared configuration")
        if "cache_group" in job or "cache_state" in job:
            group = job.get("cache_group", "")
            if (
                not group
                or not all(c.isalnum() or c in "-_" for c in group)
                or job.get("cache_state") not in {"cold", "warm"}
                or "only_backend" not in job
            ):
                raise ValueError(
                    "Controlled caches require safe group, backend and cold/warm state"
                )
            previous = cache_jobs.get(group)
            if previous is None:
                if job["cache_state"] != "cold":
                    raise ValueError("A warm-cache job must follow its cold-cache job")
                cache_jobs[group] = job
            elif (
                job["cache_state"] != "warm"
                or previous["cache_state"] != "cold"
                or any(
                    job[k] != previous[k] for k in ("config", "mode", "only_backend", "repetition")
                )
            ):
                raise ValueError("Cache pairs must match shape, protocol, repetition and backend")
            else:
                cache_jobs[group] = job
        if suite["purpose"] == "development":
            assert_disjoint(cfg, holdout)
        labels.add(job["label"])
    if any(j["cache_state"] != "warm" for j in cache_jobs.values()):
        raise ValueError("Each controlled cold-cache job needs one matching warm-cache job")
    if suite["purpose"] in {"evaluation", "replication"}:
        wanted = {(mode, rep) for mode in ("cuda_event", "cuda_graph") for rep in range(3)}
        if set(coverage) != wanted or any(v != expected for v in coverage.values()):
            raise ValueError(
                "Formal suites require all heldout cases in both protocols and three repetitions"
            )
    return suite


def results_for(root):
    metrics = list(root.glob("*/metrics.csv"))
    if len(metrics) != 1:
        return dict(verified=False, reason="Expected one benchmark result directory")
    rows = list(csv.DictReader(metrics[0].open()))
    meta = json.loads((metrics[0].parent / "metadata.json").read_text())
    counts = {}
    candidates = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        for attempt in json.loads(row.get("candidates") or "[]"):
            status = attempt["status"]
            candidates[status] = candidates.get(status, 0) + 1
    return dict(
        verified=meta["complete"],
        counts=counts,
        candidate_counts=candidates,
        run=str(metrics[0].parent),
        environment=meta["environment"],
        expected_rows=len(meta["config"]["cases"]) * len(meta["config"]["backends"]),
        rows=len(rows),
    )


def run(suite_filename, root):
    require_cuda()
    project = project_root()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=project).strip():
        raise RuntimeError("A clean committed checkout is required")
    suite = load_suite(suite_filename)
    frozen = (
        None
        if suite["purpose"] == "development"
        else FrozenPolicy.load(project / "configs/policy-v1.json")
    )
    path = new_run(root, "study-" + suite["name"])
    env = dict(
        os.environ,
        TORCHINDUCTOR_COMPILE_THREADS="1",
        TRITON_CACHE_DIR=str(path / "triton-cache"),
        TORCHINDUCTOR_CACHE_DIR=str(path / "inductor-cache"),
    )
    env.pop("TRITON_LAB_CAPTURE_DIR", None)
    state = dict(
        suite=suite,
        complete=False,
        environment=environment(),
        suite_sha256=hashlib.sha256(Path(suite_filename).read_bytes()).hexdigest(),
        process_id=os.getpid(),
        cache_policy="controlled-cold-warm-pairs"
        if suite["purpose"] == "cost"
        else "shared-within-suite-starts-empty",
        compilation_environment={
            k: env[k]
            for k in (
                "TORCHINDUCTOR_COMPILE_THREADS",
                "TRITON_CACHE_DIR",
                "TORCHINDUCTOR_CACHE_DIR",
            )
        },
        jobs=[],
    )
    if frozen is not None:
        state["frozen_policy_sha256"] = frozen.sha256
    write_json(path / "suite-status.json", state)
    print("STUDY_START", path, flush=True)
    for index, job in enumerate(suite["jobs"]):
        cfg = load_config(project / job["config"])
        cfg["name"] = job["label"]
        if "only_backend" in job:
            cfg["backends"] = [job["only_backend"]]
        cfg["timing"]["mode"] = job["mode"]
        cfg["timing"]["repeats"] = 64 if job["mode"] == "cuda_graph" else 20
        shift = job["repetition"] * len(cfg["backends"]) // 3
        cfg["backends"] = cfg["backends"][shift:] + cfg["backends"][:shift]
        job_path = path / job["label"]
        job_path.mkdir()
        config_file = job_path / "config.json"
        write_json(config_file, cfg)
        raw_root = job_path / "raw"
        job_env = dict(env)
        cache_before = None
        if "cache_group" in job:
            cache_path = path / "controlled-cache" / job["cache_group"]
            if job["cache_state"] == "cold":
                if cache_path.exists():
                    raise RuntimeError("Cold compilation-cache directory already exists")
                cache_path.mkdir(parents=True)
            elif not cache_path.is_dir():
                raise RuntimeError("Warm compilation-cache directory is missing")
            cache_before = sum(x.is_file() for x in cache_path.rglob("*"))
            if job["cache_state"] == "warm" and cache_before == 0:
                raise RuntimeError("Warm cache has no artifacts from its cold run")
            job_env.update(
                TRITON_CACHE_DIR=str(cache_path / "triton"),
                TORCHINDUCTOR_CACHE_DIR=str(cache_path / "inductor"),
            )
        command = [
            sys.executable,
            "-m",
            "triton_ops_lab.cli",
            "bench",
            "--config",
            str(config_file),
            "--output-root",
            str(raw_root),
        ]
        entry = dict(
            job=job,
            command=command,
            started=time.time(),
            status="running",
            cache_files_before=cache_before,
            compilation_environment={
                k: job_env[k]
                for k in (
                    "TORCHINDUCTOR_COMPILE_THREADS",
                    "TRITON_CACHE_DIR",
                    "TORCHINDUCTOR_CACHE_DIR",
                )
            },
            compilation_cache_note="Cache state at process start; CUDA preflight may warm other specializations, process and driver state remain distinct",
        )
        state["jobs"].append(entry)
        write_json(path / "suite-status.json", state)
        print("JOB_START", index + 1, "/", len(suite["jobs"]), job["label"], flush=True)
        with (job_path / "execution.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=project,
                env=job_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            entry["process_id"] = process.pid
            write_json(path / "suite-status.json", state)
            try:
                code = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code = 124
        entry.update(
            exit_code=code,
            seconds=time.time() - entry["started"],
            status="finished",
            results=results_for(raw_root),
        )
        if "cache_group" in job:
            entry["cache_files_after"] = sum(x.is_file() for x in cache_path.rglob("*"))
        if entry["results"]["verified"]:
            child_env = entry["results"]["environment"]
            if child_env["git_dirty"] or any(
                child_env[k] != state["environment"][k]
                for k in ("git_commit", "source_sha256", "lock_sha256")
            ):
                entry["results"].update(
                    verified=False, reason="Child source/environment identity changed"
                )
        write_json(path / "suite-status.json", state)
        print(
            "JOB_END",
            job["label"],
            "exit=",
            code,
            json.dumps(entry["results"].get("counts", {})),
            flush=True,
        )
        if (
            code not in {0, 1}
            or not entry["results"]["verified"]
            or entry["results"]["rows"] != entry["results"]["expected_rows"]
        ):
            print("STUDY_STOP incomplete job; retain evidence", path, flush=True)
            return 2
    state["complete"] = True
    state["ended"] = time.time()
    write_json(path / "suite-status.json", state)
    print("STUDY_COMPLETE", path, flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.dry_run:
            print(json.dumps(load_suite(args.suite), indent=2))
            sys.exit(0)
        sys.exit(run(args.suite, output_root(args.output_root)))
    except (RuntimeError, ValueError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
