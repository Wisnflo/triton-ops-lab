"""Frozen-policy safety and representative three-operator hardware inspection.

Instrumentation stays outside benchmark timing. No clock/security changes,
installation, online learning or evaluation-based policy changes occur.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

import torch

from triton_ops_lab.artifacts import environment, new_run, output_root, write_json
from triton_ops_lab.benchmarks.cases import make_callable, make_inputs, oracle
from triton_ops_lab.benchmarks.config import load_config
from triton_ops_lab.benchmarks.selection import candidates
from triton_ops_lab.ops import require_cuda
from triton_ops_lab.policy import FrozenPolicy
from triton_ops_lab.resource_audit import profiler_result, safety_result
from triton_ops_lab.validation import compare

spec = importlib.util.spec_from_file_location(
    "gpu_resource_tools", Path(__file__).with_name("gpu_resource_audit.py")
)
gpu_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_tools)


def worker(config_path, root, profile=False):
    require_cuda()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False
    policy = FrozenPolicy.load("configs/policy-v1.json")
    cfg = load_config(config_path)
    rows = []
    for case in cfg["cases"]:
        if profile and case["dtype"] != "float16":
            continue
        args, model = make_inputs(case, "cuda", seed=cfg["seed"])
        expected = oracle(case, args, model)
        for label, tile in [
            ("fixed", candidates(case["op"], case["shape"])[0]),
            ("rule", policy.choose(case)),
        ]:
            backends = (
                ["triton_fixed", "triton_prepared"]
                if case["op"] == "matmul" and not profile
                else ["triton_fixed"]
            )
            for backend in backends:
                folder = root / (case["id"] + "-" + label + "-" + backend)
                folder.mkdir(parents=True, exist_ok=False)
                if not profile:
                    os.environ["TRITON_LAB_CAPTURE_DIR"] = str(folder)
                fn = make_callable(case, args, model, backend, tile)
                try:
                    accuracy = compare(fn(), expected, case["op"], case["dtype"])
                finally:
                    os.environ.pop("TRITON_LAB_CAPTURE_DIR", None)
                if not accuracy["passed"]:
                    raise RuntimeError(f"Qualification accuracy failed: {case} {label} {accuracy}")
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push("triton_lab")
                actual = fn()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                second = compare(actual, expected, case["op"], case["dtype"])
                if not second["passed"]:
                    raise RuntimeError("Warm invocation accuracy failed")
                resources = list(folder.glob("*/resources.json"))
                if not profile:
                    if len(resources) != 1:
                        raise RuntimeError("Expected one compiled device kernel")
                    resource = json.loads(resources[0].read_text())
                    if not {"ttir", "ttgir", "llir", "ptx", "cubin"} <= set(resource["asm_files"]):
                        raise RuntimeError("Compiled IR chain is incomplete")
                    if (
                        resource["registers"] is None
                        or resource["spills"] is None
                        or resource["compiler_metadata"].get("shared") is None
                    ):
                        raise RuntimeError("Compiler resources missing")
                rows.append(
                    dict(
                        case=case,
                        configuration=label,
                        tile=tile,
                        backend=backend,
                        accuracy=accuracy,
                        resources=[str(f) for f in resources],
                    )
                )
                write_json(root / "worker-results.json", dict(complete=False, results=rows))
    write_json(
        root / "worker-results.json",
        dict(complete=True, environment=environment(), policy_sha256=policy.sha256, results=rows),
    )


def run(action, config_path, root):
    require_cuda()
    provenance = environment()
    if provenance["git_dirty"]:
        raise RuntimeError("Qualification requires a clean committed checkout")
    policy = FrozenPolicy.load("configs/policy-v1.json")
    path = new_run(root, "qualification-" + action)
    env = dict(
        os.environ,
        TORCHINDUCTOR_COMPILE_THREADS="1",
        TRITON_CACHE_DIR=str(path / "triton-cache"),
        TORCHINDUCTOR_CACHE_DIR=str(path / "inductor-cache"),
    )
    env.pop("TRITON_LAB_CAPTURE_DIR", None)
    meta = dict(
        complete=False,
        environment=provenance,
        policy_sha256=policy.sha256,
        instrumentation_environment={
            k: env[k]
            for k in (
                "TORCHINDUCTOR_COMPILE_THREADS",
                "TRITON_CACHE_DIR",
                "TORCHINDUCTOR_CACHE_DIR",
            )
        },
        results={},
    )
    write_json(path / "metadata.json", meta)
    execute = gpu_tools.execute
    print("QUALIFICATION_START", path, flush=True)
    if action in {"all", "safety"}:
        sanitizer = gpu_tools.find_tool("compute-sanitizer")
        if not sanitizer:
            meta["results"]["safety"] = dict(
                status="unavailable", reason="compute-sanitizer is not installed"
            )
        else:
            execute([sanitizer, "--version"], path, "sanitizer-version", env, 30)
            code, log = execute(
                [sys.executable, "-m", "pytest", "-m", "gpu", "--collect-only", "-q"],
                path,
                "collect-gpu",
                env,
                60,
            )
            found = re.findall(r"(\d+)(?:/\d+)? tests? collected", log)
            if code or not found or int(found[-1]) <= 0:
                raise RuntimeError("Cannot establish intended GPU test count")
            expected = int(found[-1])
            meta["expected_gpu_tests"] = expected
            for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
                command = [
                    sanitizer,
                    "--tool",
                    tool,
                    "--error-exitcode",
                    "1",
                    "--target-processes",
                    "all",
                    "--launch-timeout",
                    "120",
                ]
                if tool == "initcheck":
                    command += ["--initcheck-address-space", "all"]
                command += [sys.executable, "-m", "pytest", "-m", "gpu", "-q"]
                code, log = execute(command, path, tool, env, 1200)
                result = safety_result(tool, code, log, expected)
                meta["results"][tool] = result
                write_json(path / "metadata.json", meta)
                print("SAFETY_END", tool, result["status"], flush=True)
                if result["status"] != "passed":
                    print("Safety gate did not pass; preserve evidence", flush=True)
                    break
    if action == "all" and any(
        meta["results"].get(tool, {}).get("status") != "passed"
        for tool in ("memcheck", "racecheck", "initcheck", "synccheck")
    ):
        meta.update(complete=True, all_requested_passed=False)
        meta["results"]["hardware"] = dict(status="not_run", reason="Safety gate did not pass")
        write_json(path / "metadata.json", meta)
        return 1
    if action in {"all", "hardware"}:
        command = [sys.executable, __file__, "worker", "--config", config_path]
        code, _ = execute(
            [*command, "--output-root", str(path / "capture")], path, "capture", env, 600
        )
        meta["results"]["capture"] = dict(
            status="passed" if code == 0 else "failed", exit_code=code
        )
        disassembler = gpu_tools.find_tool("nvdisasm")
        sass = []
        if disassembler and code == 0:
            execute([disassembler, "--version"], path, "nvdisasm-version", env, 30)
            for i, cubin in enumerate(sorted((path / "capture").glob("*/*/kernel.cubin"))):
                status, log = execute([disassembler, str(cubin)], path, "sass-" + str(i), env, 60)
                sass.append(
                    dict(
                        cubin=str(cubin),
                        exit_code=status,
                        disassembly_log="sass-" + str(i) + ".log",
                        sha256=hashlib.sha256(log.encode()).hexdigest(),
                    )
                )
        write_json(path / "sass-manifest.json", sass)
        meta["results"]["sass"] = dict(
            status="passed"
            if sass and all(x["exit_code"] == 0 for x in sass)
            else "unavailable"
            if not disassembler
            else "failed",
            count=len(sass),
        )
        ncu = gpu_tools.find_tool("ncu")
        if not ncu:
            meta["results"]["nsight"] = dict(status="unavailable")
        else:
            execute([ncu, "--version"], path, "ncu-version", env, 30)
            sections = [
                "LaunchStats",
                "Occupancy",
                "ComputeWorkloadAnalysis",
                "MemoryWorkloadAnalysis",
                "InstructionStats",
            ]
            flags = [x for name in sections for x in ["--section", name]]
            code, log = execute(
                [
                    ncu,
                    "--target-processes",
                    "all",
                    "--nvtx",
                    "--nvtx-include",
                    "triton_lab/",
                    *flags,
                    "--export",
                    str(path / "hardware"),
                    *command,
                    "--profile",
                    "--output-root",
                    str(path / "profile-worker"),
                ],
                path,
                "nsight",
                env,
                900,
            )
            report = path / "hardware.ncu-rep"
            result = profiler_result(code, log, report.is_file() and report.stat().st_size > 0)
            result["requested_sections"] = sections
            if result["status"] == "collected":
                export, _ = execute(
                    [ncu, "--import", str(report), "--csv", "--page", "raw"],
                    path,
                    "nsight-metrics",
                    env,
                    60,
                )
                result["csv_export_exit_code"] = export
                if export:
                    result["status"] = "failed"
            meta["results"]["nsight"] = result
    meta["complete"] = True
    meta["all_requested_passed"] = all(
        r["status"] in {"passed", "collected"} for r in meta["results"].values()
    )
    if action in {"all", "safety"}:
        meta["all_requested_passed"] &= all(
            meta["results"].get(tool, {}).get("status") == "passed"
            for tool in ("memcheck", "racecheck", "initcheck", "synccheck")
        )
    write_json(path / "metadata.json", meta)
    print("QUALIFICATION_COMPLETE", path, json.dumps(meta["results"]), flush=True)
    return 0 if meta["all_requested_passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["all", "safety", "hardware", "worker"])
    parser.add_argument("--config", default="configs/study/qualification.json")
    parser.add_argument("--output-root")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "worker":
            worker(args.config, output_root(args.output_root), args.profile)
            sys.exit(0)
        sys.exit(run(args.action, args.config, output_root(args.output_root)))
    except (RuntimeError, ValueError, OSError) as exc:
        print("error:", exc, file=sys.stderr)
        sys.exit(2)
