"""Run via uv run --frozen python scripts/gpu_tools.py --help.

No remote access or paid-resource lifecycle changes. Capture/profile are intentionally
separate from bench so instrumentation does not contaminate benchmark timing.
"""

import argparse
import os
import shutil
import subprocess
import sys

from triton_ops_lab.artifacts import environment, new_run, output_root, write_json
from triton_ops_lab.benchmarks.cases import make_callable, make_inputs, oracle
from triton_ops_lab.benchmarks.config import load_config
from triton_ops_lab.ops import require_cuda
from triton_ops_lab.validation import compare


def worker(config_path, case_id):
    require_cuda()
    cfg = load_config(config_path)
    matching = [c for c in cfg["cases"] if c["id"] == case_id]
    if len(matching) != 1:
        raise ValueError("Select an exact case id from --config")
    case = matching[0]
    args, model = make_inputs(case, "cuda", seed=cfg["seed"])
    fn = make_callable(case, args, model, "triton_fixed")
    expected = oracle(case, args, model)
    result = compare(fn(), expected, case["op"], case["dtype"])
    if not result["passed"]:
        raise RuntimeError(f"Correctness failed: {result}")
    # Explicit range keeps profiling focused on warmed Triton execution.
    import torch

    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("triton_lab")
    fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    print(result)


def main():
    p = argparse.ArgumentParser(
        description="GPU validation / sanitizer / profiler / compiled IR capture"
    )
    p.add_argument("action", choices=["check", "sanitize", "profile", "capture", "worker"])
    p.add_argument("--config", default="configs/smoke.json")
    p.add_argument("--case", default="gemm-epilogue-float16")
    p.add_argument("--output-root")
    args = p.parse_args()
    try:
        require_cuda()
        if args.action == "worker":
            worker(args.config, args.case)
            return 0
        root = output_root(args.output_root)
        tool = {"sanitize": "compute-sanitizer", "profile": "ncu"}.get(args.action)
        if tool and not shutil.which(tool):
            raise RuntimeError(f"{tool} is not installed; see docs/project.md")
        path = new_run(root, args.action)
        env = dict(os.environ, TRITON_LAB_OUTPUT_ROOT=str(root))
        env.pop("TRITON_LAB_CAPTURE_DIR", None)
        worker_cmd = [
            sys.executable,
            __file__,
            "worker",
            "--config",
            args.config,
            "--case",
            args.case,
        ]
        if args.action == "check":
            cmd = [sys.executable, "-m", "pytest", "-m", "gpu", "-q"]
        elif args.action == "sanitize":
            cmd = [
                tool,
                "--tool",
                "memcheck",
                "--error-exitcode",
                "1",
                sys.executable,
                "-m",
                "pytest",
                "-m",
                "gpu",
                "-q",
            ]
        elif args.action == "profile":
            cmd = [
                tool,
                "--target-processes",
                "all",
                "--set",
                "full",
                "--nvtx",
                "--nvtx-include",
                "triton_lab/",
                "--export",
                str(path / "profile"),
                *worker_cmd,
            ]
        else:
            env["TRITON_LAB_CAPTURE_DIR"] = str(path / "compiled")
            # Separate empty cache ensures capture includes compilation for this invocation.
            env["TRITON_CACHE_DIR"] = str(path / "triton-cache")
            cmd = worker_cmd
        meta = {
            "environment": environment(),
            "command": cmd,
            "output_directory": str(path),
            "status": "running",
        }
        write_json(path / "metadata.json", meta)
        with (path / "execution.log").open("w") as log:
            completed = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        meta.update(
            status="passed" if completed.returncode == 0 else "failed",
            exit_code=completed.returncode,
        )
        write_json(path / "metadata.json", meta)
        print(path)
        if completed.returncode:
            print(
                "GPU tool failed; inspect execution.log (including ERR_NVGPUCTRPERM if profiling permissions are absent)",
                file=sys.stderr,
            )
        return completed.returncode
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
