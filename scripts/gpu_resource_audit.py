"""Capture fixed/selected GEMM resources, run memcheck, attempt bounded Nsight profiling.

No performance selection, evaluation-set access, installation or security changes.
Every external tool has a timeout. Missing tools/permissions are never passes.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import signal

import torch

from triton_ops_lab.artifacts import environment, new_run, output_root, write_json
from triton_ops_lab.benchmarks.cases import make_callable, make_inputs, oracle
from triton_ops_lab.benchmarks.config import load_config
from triton_ops_lab.ops import require_cuda
from triton_ops_lab.resource_audit import profiler_result, sanitizer_result
from triton_ops_lab.validation import compare

FIXED = dict(bm=32, bn=64, bk=32, warps=4, stages=2)


def selected_config(case):
    # Frozen from fourth-round Graph development results, never from eval.
    small = case["id"].split("-")[1] in {"tail", "square"}
    return (
        dict(bm=32, bn=32, bk=32, warps=4, stages=2)
        if small
        else dict(bm=64, bn=64, bk=64, warps=4, stages=3)
    )


def worker(config_path, path, profile_only=False):
    require_cuda()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    cfg = load_config(config_path)
    results = []
    for case in cfg["cases"]:
        if profile_only and case["id"] != "fused-large-float16":
            continue
        args, model = make_inputs(case, "cuda", seed=cfg["seed"])
        expected = oracle(case, args, model)
        for label, tile in (("fixed", FIXED), ("selected", selected_config(case))):
            for backend in (
                ("triton_fused",) if profile_only else ("triton_fused", "triton_prepared")
            ):
                target = path / (case["id"] + "-" + label + "-" + backend)
                target.mkdir(parents=True, exist_ok=False)
                if not profile_only:
                    os.environ["TRITON_LAB_CAPTURE_DIR"] = str(target)
                fn = make_callable(case, args, model, backend, config=tile)
                try:
                    accuracy = compare(fn(), expected, case["op"], case["dtype"])
                finally:
                    os.environ.pop("TRITON_LAB_CAPTURE_DIR", None)
                if not accuracy["passed"]:
                    raise RuntimeError(
                        f"Correctness failed: {case['id']} {label} {backend}: {accuracy}"
                    )
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push("triton_lab")
                second = fn()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
                replay_accuracy = compare(second, expected, case["op"], case["dtype"])
                if not replay_accuracy["passed"]:
                    raise RuntimeError("Second invocation failed correctness")
                # Capture resource availability is checked independently from numerical correctness.
                resources = list(target.glob("*/resources.json"))
                if not profile_only and len(resources) != 1:
                    raise RuntimeError(f"Expected one compiled fused kernel: {target}")
                if not profile_only:
                    resource = json.loads(resources[0].read_text())
                    if not {"ptx", "cubin"}.issubset(resource["asm_files"]):
                        raise RuntimeError("PTX/cubin capture incomplete")
                    if (
                        any(resource[key] is None for key in ("registers", "spills"))
                        or resource["compiler_metadata"].get("shared") is None
                    ):
                        raise RuntimeError("Compiler resource metadata unavailable")
                results.append(
                    dict(
                        case=case,
                        configuration=label,
                        tile=tile,
                        backend=backend,
                        accuracy=accuracy,
                        resource_files=[str(f) for f in resources],
                    )
                )
                write_json(path / "worker-results.json", dict(complete=False, results=results))
    write_json(
        path / "worker-results.json",
        dict(complete=True, environment=environment(), results=results),
    )


def find_tool(name):
    found = shutil.which(name)
    if found:
        return found
    patterns = [
        f"cuda*/bin/{name}",
        f"cuda*/compute-sanitizer/{name}",
        f"cuda*/nsight-compute*/{name}",
    ]
    for pattern in patterns:
        for file in sorted(Path("/usr/local").glob(pattern), reverse=True):
            if file.is_file() and os.access(file, os.X_OK):
                return str(file)
    return None


def execute(command, path, label, env, timeout):
    log_file = path / (label + ".log")
    with log_file.open("w") as log:
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            log.write(f"\nAUDIT_TIMEOUT after {timeout}s\n")
            code = 124
    result = dict(command=command, exit_code=code, log=str(log_file), timeout_seconds=timeout)
    write_json(path / (label + ".command.json"), result)
    print(label, "exit=", code, flush=True)
    return code, log_file.read_text()


def audit(config_path, root):
    require_cuda()
    path = new_run(root, "gpu-resource-audit")
    tools = {name: find_tool(name) for name in ("compute-sanitizer", "ncu", "nvcc")}
    # Multiprocess compilation can stall Python shutdown under Compute Sanitizer.
    # This changes compilation scheduling for instrumentation only, not kernels or benchmarks.
    compile_threads = os.environ.get("TORCHINDUCTOR_COMPILE_THREADS", "1")
    meta = dict(
        complete=False,
        environment=environment(),
        tools=tools,
        results={},
        instrumentation_environment={"TORCHINDUCTOR_COMPILE_THREADS": compile_threads},
    )
    write_json(path / "metadata.json", meta)
    env = dict(os.environ, TRITON_CACHE_DIR=str(path / "triton-cache"))
    env["TORCHINDUCTOR_COMPILE_THREADS"] = compile_threads
    env.pop("TRITON_LAB_CAPTURE_DIR", None)
    for name, tool in tools.items():
        if tool:
            execute([tool, "--version"], path, name + "-version", env, 30)
    capture_path = path / "capture"
    command = [sys.executable, __file__, "worker", "--config", config_path]
    code, _ = execute([*command, "--output-root", str(capture_path)], path, "capture", env, 600)
    meta["results"]["capture"] = dict(status="passed" if code == 0 else "failed", exit_code=code)
    write_json(path / "metadata.json", meta)
    tool = tools["compute-sanitizer"]
    if tool:
        code, log = execute(
            [
                tool,
                "--tool",
                "memcheck",
                "--error-exitcode",
                "1",
                "--target-processes",
                "all",
                "--launch-timeout",
                "120",
                sys.executable,
                "-m",
                "pytest",
                "-m",
                "gpu",
                "-q",
            ],
            path,
            "memcheck",
            env,
            900,
        )
        meta["results"]["memcheck"] = sanitizer_result(code, log)
    else:
        meta["results"]["memcheck"] = dict(
            status="unavailable", reason="compute-sanitizer not installed"
        )
    write_json(path / "metadata.json", meta)
    tool = tools["ncu"]
    if tool:
        report = path / "nsight.ncu-rep"
        code, log = execute(
            [
                tool,
                "--target-processes",
                "all",
                "--set",
                "basic",
                "--section",
                "LaunchStats",
                "--section",
                "Occupancy",
                "--nvtx",
                "--nvtx-include",
                "triton_lab/",
                "--export",
                str(path / "nsight"),
                *command,
                "--profile-only",
                "--output-root",
                str(path / "profile-worker"),
            ],
            path,
            "nsight",
            env,
            300,
        )
        result = profiler_result(code, log, report.is_file() and report.stat().st_size > 0)
        if result["status"] == "collected":
            export_code, _ = execute(
                [tool, "--import", str(report), "--csv", "--page", "raw"],
                path,
                "nsight-metrics",
                env,
                60,
            )
            result["csv_export_exit_code"] = export_code
            if export_code:
                result["status"] = "failed"
        meta["results"]["nsight"] = result
    else:
        meta["results"]["nsight"] = dict(status="unavailable", reason="ncu not installed")
    meta["complete"] = True
    meta["all_tools_succeeded"] = all(
        x["status"] in {"passed", "collected"} for x in meta["results"].values()
    )
    write_json(path / "metadata.json", meta)
    print(path, meta["results"], flush=True)
    return 0 if meta["all_tools_succeeded"] else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["run", "worker"])
    p.add_argument("--config", default="configs/gemm-tune-events.json")
    p.add_argument("--output-root")
    p.add_argument("--profile-only", action="store_true")
    args = p.parse_args()
    try:
        if args.action == "worker":
            worker(args.config, output_root(args.output_root), args.profile_only)
            sys.exit(0)
        sys.exit(audit(args.config, output_root(args.output_root)))
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
