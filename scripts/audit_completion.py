"""Audit technical completion from raw evidence; does not create an academic report."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import ast
import re
import subprocess

from run_study import load_suite
from triton_ops_lab.analysis.study import (
    audit_study,
    comparisons,
    cost_amortization,
    read_json,
    summarize_study,
    ablation_comparisons,
    replication_consistency,
)
from triton_ops_lab.analysis.hardware import audit_hardware
from triton_ops_lab.analysis.initialization import audit_attributed_safety
from triton_ops_lab.analysis.safety import (
    vendor_race_attribution,
    VENDOR_WINOGRAD,
)
from triton_ops_lab.artifacts import project_root, write_json
from triton_ops_lab.benchmarks.config import load_config, identity
from triton_ops_lab.benchmarks.selection import candidates, RULE_SEARCH_BACKENDS
from triton_ops_lab.policy import FrozenPolicy
from triton_ops_lab.resource_audit import safety_result, profiler_result


def source_archive(path, commit):
    for parent in (Path(path), *Path(path).parents):
        bundle = parent / "source.bundle"
        manifest = parent / "manifest.json"
        if not bundle.is_file() or not manifest.is_file():
            continue
        data = read_json(manifest)
        if (
            data["commit"] != commit
            or hashlib.sha256(bundle.read_bytes()).hexdigest() != data["bundle_sha256"]
        ):
            raise ValueError("Source delivery archive differs from recorded identity")
        notebooks = list(parent.glob("*.ipynb"))
        if (
            len(notebooks) != 1
            or hashlib.sha256(notebooks[0].read_bytes()).hexdigest() != data["notebook_sha256"]
        ):
            raise ValueError("Notebook delivery identity mismatch")
        refs = subprocess.check_output(
            ["git", "bundle", "list-heads", str(bundle.resolve())], text=True
        )
        if commit + " HEAD" not in refs:
            raise ValueError("Source bundle does not identify the measured commit")
        return dict(
            bundle=str(bundle),
            sha256=data["bundle_sha256"],
            commit=commit,
            notebook_sha256=data["notebook_sha256"],
        )
    raise ValueError("Colab source delivery archive not found beside evidence")


def verify_plan(audit):
    purpose = audit["state"]["suite"]["purpose"]
    expected = load_suite(project_root() / f"configs/study/{purpose}-suite.json")
    if expected != audit["state"]["suite"]:
        raise ValueError("Executed suite differs from the committed experiment plan")
    for item in audit["rows"]:
        job = item["job"]
        cfg = load_config(project_root() / job["config"])
        cfg["name"] = job["label"]
        if "only_backend" in job:
            cfg["backends"] = [job["only_backend"]]
        cfg["timing"]["mode"] = job["mode"]
        cfg["timing"]["repeats"] = 64 if job["mode"] == "cuda_graph" else 20
        shift = job["repetition"] * len(cfg["backends"]) // 3
        cfg["backends"] = cfg["backends"][shift:] + cfg["backends"][:shift]
        if cfg != item["metadata"]["config"]:
            raise ValueError("Raw measurement configuration differs from the committed job")


def clean_safety(path, tool, expected, recorded):
    command = read_json(path / (tool + ".command.json"))
    log = (path / (tool + ".log")).read_text()
    actual = safety_result(tool, command["exit_code"], log, expected)
    if (
        actual["status"] != "passed"
        or actual != recorded
        or re.search(r"=+\s*(?:Warning|WARNING)[: ]", log)
    ):
        raise ValueError("A safety log does not prove a clean full test run: " + tool)
    return command["command"]


def qualification(path, policy):
    path = Path(path)
    scope = None
    if path.is_file():
        combined = read_json(path)
        root = path.parent
        if combined.get("schema") == "project-safety-v2":
            scope = audit_attributed_safety(root, combined, policy, project_root())
            hardware_path = root / combined["hardware"]
            hardware_meta = read_json(hardware_path / "metadata.json")
            if (
                hardware_meta["environment"]
                != scope["qualification_environments"]["original_initcheck"]
            ):
                raise ValueError(
                    "Reused hardware evidence differs from its original qualification session"
                )
            return hardware_evidence(hardware_path, policy, scope)
        original = root / combined["primary"]
        supp = root / combined["supplement"]
        primary = read_json(original / "metadata.json")
        followup = read_json(supp / "metadata.json")
        remaining = root / combined["remaining"] if "remaining" in combined else supp
        remaining_meta = read_json(remaining / "metadata.json") if remaining != supp else followup
        if (
            not primary["complete"]
            or not followup["complete"]
            or not remaining_meta["complete"]
            or not (
                remaining_meta.get("all_remaining_passed")
                if remaining != supp
                else followup["all_project_checks_passed"]
            )
        ):
            raise ValueError("Supplemental safety evidence is incomplete")
        if primary["expected_gpu_tests"] != 96 or any(
            x["policy_sha256"] != policy.sha256 for x in (primary, followup, remaining_meta)
        ):
            raise ValueError("Supplemental policy/test scope differs")
        if any(x["environment"] != primary["environment"] for x in (followup, remaining_meta)):
            raise ValueError("Source or environment changed during warning attribution")
        clean_safety(original, "memcheck", 96, primary["results"]["memcheck"])
        race = read_json(original / "racecheck.command.json")
        race_log = (original / "racecheck.log").read_text()
        parsed = safety_result("racecheck", race["exit_code"], race_log, 96)
        if (
            race["exit_code"] != 1
            or parsed != primary["results"]["racecheck"]
            or parsed["pytest_passed"] != 96
        ):
            raise ValueError("Original unfiltered race evidence differs")
        if any(x.startswith("--kernel-name") for x in race["command"]):
            raise ValueError("Original warning attribution must use an unfiltered run")
        repro = (supp / "vendor-repro.log").read_text()
        repro_command = read_json(supp / "vendor-repro.command.json")
        if repro_command["exit_code"] != 1:
            raise ValueError("Isolated vendor reproduction did not reproduce the warning")
        source = (supp / "vendor-repro.py").read_text()
        imports = [
            n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.Import, ast.ImportFrom))
        ]
        if not imports or any(
            not isinstance(n, ast.Import)
            or any(a.name not in {"torch", "torch.nn.functional"} for a in n.names)
            for n in imports
        ):
            raise ValueError("Vendor-only reproduction imports non-framework code")
        scope = vendor_race_attribution(race_log, repro)
        if "initcheck_global" in combined:
            raise ValueError("Split filtered initialization evidence is not accepted")
        for tool in ("racecheck", "initcheck", "synccheck"):
            directory, record = (
                (supp, followup) if tool == "racecheck" else (remaining, remaining_meta)
            )
            command = clean_safety(directory, tool, 96, record["results"][tool])
            filters = [
                (x, command[i + 1])
                for i, x in enumerate(command[:-1])
                if x.startswith("--kernel-name")
            ]
            expected_filters = (
                [("--kernel-name-exclude", "kne=" + VENDOR_WINOGRAD)] if tool == "racecheck" else []
            )
            if filters != expected_filters:
                raise ValueError("Safety filters omit an unintended kernel")
            if (
                tool == "initcheck"
                and command[command.index("--initcheck-address-space") + 1] != "all"
            ):
                raise ValueError("Incomplete initcheck address-space coverage")
        script = root / "qualification-supplement.py"
        if hashlib.sha256(script.read_bytes()).hexdigest() != followup["source_script_sha256"]:
            raise ValueError("Supplemental execution script identity differs")
        if remaining != supp:
            source = root / (combined["remaining"] + ".py")
            if (
                hashlib.sha256(source.read_bytes()).hexdigest()
                != remaining_meta["source_script_sha256"]
            ):
                raise ValueError("Remaining safety execution script identity differs")
            package = remaining_meta["tool_package"]
            if package[
                "sha256"
            ] != "b9637777a31cd0f0ffe1e965523e8b0d382019b8496855fd688c886988583cf4" or not package[
                "url"
            ].startswith("https://developer.download.nvidia.com/compute/cuda/redist/"):
                raise ValueError("Standalone sanitizer tool identity differs")
            if remaining_meta["results"]["initcheck_smoke"] != {
                "exit_code": 0,
                "complete": True,
            }:
                raise ValueError("Standalone sanitizer did not pass compatibility smoke")
            scope["standalone_tool_package"] = package
            scope["remaining_sanitizer_version"] = (remaining / "sanitizer-version.log").read_text()
        path = root / combined["hardware"]
    meta = read_json(path / "metadata.json")
    if (
        not meta["complete"]
        or not meta["all_requested_passed"]
        or meta["policy_sha256"] != policy.sha256
    ):
        raise ValueError("Qualification is incomplete or used a different policy")
    if scope is None:
        expected = meta["expected_gpu_tests"]
        if expected != 96:
            raise ValueError("Qualification did not cover the complete GPU test suite")
        for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
            clean_safety(path, tool, expected, meta["results"][tool])
    elif any(
        meta["environment"][k] != primary["environment"][k]
        for k in ("git_commit", "source_sha256", "lock_sha256", "gpu", "driver", "runtime_identity")
    ):
        raise ValueError("Hardware inspection differs from supplemental safety source/environment")
    return hardware_evidence(path, policy, scope or dict(global_racecheck_status="passed"))


def hardware_evidence(path, policy, scope):
    path = Path(path)
    meta = read_json(path / "metadata.json")
    if (
        not meta["complete"]
        or not meta["all_requested_passed"]
        or meta["policy_sha256"] != policy.sha256
    ):
        raise ValueError("Representative hardware evidence is incomplete")
    capture = read_json(path / "capture/worker-results.json")
    if (
        not capture["complete"]
        or capture["policy_sha256"] != policy.sha256
        or len(capture["results"]) != 16
    ):
        raise ValueError("Incomplete representative compiler capture")
    resources = list((path / "capture").glob("*/*/resources.json"))
    if len(resources) != 16:
        raise ValueError("Expected 16 complete compiler captures")
    for resource in resources:
        data = read_json(resource)
        if not {"ttir", "ttgir", "llir", "ptx", "cubin"} <= set(data["asm_files"]):
            raise ValueError("Missing compiler IR")
        for info in data["asm_files"].values():
            file = resource.parent / info["file"]
            if (
                file.stat().st_size != info["bytes"]
                or hashlib.sha256(file.read_bytes()).hexdigest() != info["sha256"]
            ):
                raise ValueError("Compiler artifact hash mismatch")
    sass = read_json(path / "sass-manifest.json")
    if len(sass) != 16:
        raise ValueError("Expected 16 SASS disassemblies")
    for row in sass:
        log = (path / row["disassembly_log"]).read_bytes()
        if row["exit_code"] or not log or hashlib.sha256(log).hexdigest() != row["sha256"]:
            raise ValueError("SASS disassembly is incomplete")
    cmd = read_json(path / "nsight.command.json")
    result = profiler_result(
        cmd["exit_code"], (path / "nsight.log").read_text(), (path / "hardware.ncu-rep").is_file()
    )
    profile = read_json(path / "profile-worker/worker-results.json")
    if result["status"] != "collected" or not profile["complete"] or len(profile["results"]) != 6:
        raise ValueError("Three-operator hardware profiling is incomplete")
    if (
        not (path / "nsight-metrics.log").stat().st_size
        or meta["results"]["nsight"].get("csv_export_exit_code") != 0
    ):
        raise ValueError("Missing raw hardware counters")
    return {
        **meta,
        "hardware_root": str(path),
        "safety_scope": scope or dict(global_racecheck_status="passed"),
    }


def complete(development, evaluation, replication, cost, hardware, output):
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=project_root()).strip():
        raise ValueError("Commit the verifier before recording a completion audit")
    policy = FrozenPolicy.load(project_root() / "configs/policy-v1.json")
    sealed = read_json(project_root() / "configs/study/policy-decisions-v1.json")
    if sealed["policy_sha256"] != policy.sha256:
        raise ValueError("Pre-evaluation policy seal differs")
    if (
        sealed["eval_config_sha256"]
        != hashlib.sha256((project_root() / "configs/eval.json").read_bytes()).hexdigest()
    ):
        raise ValueError("Heldout configuration differs from its pre-evaluation seal")
    sealed_cases = {
        (identity(x["case"]), x["case"]["dtype"]): x["decision"] for x in sealed["decisions"]
    }
    audits = {
        key: audit_study(path, key)
        for key, path in [
            ("development", development),
            ("evaluation", evaluation),
            ("replication", replication),
            ("cost", cost),
        ]
    }
    for audit in audits.values():
        verify_plan(audit)
    archives = {
        key: source_archive(value["path"], value["state"]["environment"]["git_commit"])
        for key, value in audits.items()
    }
    if len(audits["development"]["rows"]) != 1488:
        raise ValueError("Development coverage changed")
    primary = audits["evaluation"]["state"]["environment"]
    replica = audits["replication"]["state"]["environment"]
    changed = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            primary["git_commit"],
            "--",
            "src",
            "configs",
            "scripts",
            "uv.lock",
            "pyproject.toml",
        ],
        cwd=project_root(),
        text=True,
    ).splitlines()
    if any(
        not x.startswith("src/triton_ops_lab/analysis/") and x != "scripts/audit_completion.py"
        for x in changed
    ):
        raise ValueError(
            "GPU workload/configuration changed since the measured source; analysis-only changes are allowed"
        )
    for label in ("evaluation", "replication", "cost"):
        state = audits[label]["state"]
        if state["frozen_policy_sha256"] != policy.sha256:
            raise ValueError("Policy changed between phases")
        if any(
            j["started"] <= datetime.fromisoformat(policy.data["frozen_at"]).timestamp()
            for j in state["jobs"]
        ):
            raise ValueError("Heldout measurements were started before policy freeze")
        if any(
            state["environment"][k] != primary[k]
            for k in ("git_commit", "source_sha256", "lock_sha256", "gpu", "driver", "packages")
        ):
            raise ValueError("Formal source, environment or hardware differs")
        for item in audits[label]["rows"]:
            row = item["row"]
            case = item["case"]
            seal = sealed_cases[(identity(case), case["dtype"])]
            attempts = json.loads(row["candidates"])
            if row["backend"] in RULE_SEARCH_BACKENDS:
                if [a["config"] for a in attempts] != seal["shortlist"]:
                    raise ValueError("Search used a different frozen shortlist")
            elif row["backend"] in {"triton_search", "triton_prepared_search"}:
                if [a["config"] for a in attempts] != candidates(
                    case["op"], case["shape"], "search", 6
                ):
                    raise ValueError("Full-search control changed its candidate pool")
            if row["backend"] in {"triton_rule", "triton_prepared_rule"}:
                decisions = json.loads(row["policy_decisions"])
                if case["op"] in {"transformer", "cnn"}:
                    plan = decisions["plan"]
                    if plan != seal["plan"]:
                        raise ValueError("Module plan differs from the pre-evaluation seal")
                    if plan["input_shape"] != case["shape"] or plan["dtype"] != case["dtype"]:
                        raise ValueError("Module used a mismatched layer plan")
                    for layer in plan["layers"].values():
                        if layer["config"] != policy.choose(layer["case"]):
                            raise ValueError("Module configuration was changed")
                elif json.loads(row["selected_config"]) != policy.choose(case):
                    raise ValueError("Static policy configuration was changed")
                elif json.loads(row["selected_config"]) != seal["selected"]:
                    raise ValueError("Static decision differs from its pre-evaluation seal")
    if primary["gpu"]["name"] != "NVIDIA L4":
        raise ValueError("Primary replication requires L4")
    for key in ("hostname_sha256", "boot_id_sha256"):
        if (
            not primary["runtime_identity"][key]
            or primary["runtime_identity"][key] == replica["runtime_identity"][key]
        ):
            raise ValueError("Independent L4 runtime identity is unproven")
    if (
        any(len(audits[k]["rows"]) != 1284 for k in ("evaluation", "replication"))
        or len(audits["cost"]["jobs"]) != 84
    ):
        raise ValueError("Formal/cost coverage changed")
    pairs = {}
    for entry in audits["cost"]["state"]["jobs"]:
        key = entry["job"]["cache_group"]
        if entry["job"]["cache_state"] == "cold":
            pairs[key] = entry["cache_files_after"]
        elif entry["cache_files_before"] != pairs[key]:
            raise ValueError("Warm cache file state does not match its paired cold run")
    trained = {x["metrics_sha256"] for x in policy.data["source_measurements"]}
    raw = {x["csv_sha256"] for x in audits["development"]["jobs"]}
    if not trained <= raw:
        raise ValueError("Frozen training source is not the audited development evidence")
    hardware_meta = qualification(hardware, policy)
    archives["qualification"] = source_archive(hardware, hardware_meta["environment"]["git_commit"])
    if any(
        hardware_meta["environment"][k] != primary[k]
        for k in ("git_commit", "source_sha256", "lock_sha256", "gpu", "driver")
    ):
        raise ValueError("Hardware qualification differs from formal source/environment")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, audit in audits.items():
        write_json(output / (name + "-raw-summary.json"), summarize_study(audit))
        write_json(output / (name + "-comparisons.json"), comparisons(audit))
    write_json(output / "cost-amortization.json", cost_amortization(audits["cost"], policy))
    write_json(
        output / "replication-consistency.json",
        replication_consistency(audits["evaluation"], audits["replication"]),
    )
    write_json(output / "orthogonal-ablation.json", ablation_comparisons(audits["development"]))
    write_json(output / "hardware-crosscheck.json", audit_hardware(hardware_meta["hardware_root"]))
    result = dict(
        complete=True,
        policy_sha256=policy.sha256,
        environment=primary,
        verifier_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root(), text=True
        ).strip(),
        analysis_only_changes_since_measurement=changed,
        source_archives=archives,
        evidence={k: v["path"] for k, v in audits.items()},
        dispatch_supplements={k: v.get("dispatch_supplement") for k, v in audits.items()},
        qualification=str(hardware),
        safety_scope=hardware_meta["safety_scope"],
        scope="Technical evidence only; no paper/report. No general speedup claim beyond measured cases and protocols.",
    )
    write_json(output / "completion-audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("development", "evaluation", "replication", "cost", "hardware", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    complete(**vars(args))
