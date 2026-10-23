import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch


def project_root():
    return Path(__file__).resolve().parents[2]


def output_root(explicit=None):
    value = explicit or os.environ.get("TRITON_LAB_OUTPUT_ROOT")
    if value:
        result = Path(value).expanduser().resolve()
    else:
        project = project_root()
        workspace = project.parent.parent
        if project.parent.name != "labs" or not (workspace / "storage" / "README.md").is_file():
            raise ValueError("Set --output-root or TRITON_LAB_OUTPUT_ROOT for a standalone clone")
        result = workspace / "storage" / "runs" / "triton"
    if result == project_root() or project_root() in result.parents:
        raise ValueError("Experiment outputs must be outside the source repository")
    return result


def new_run(root, label):
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:64]
    path = Path(root) / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + safe
        + "-"
        + uuid.uuid4().hex[:8]
    )
    path.mkdir(parents=True, exist_ok=False)
    return path


def command_output(args, cwd=None):
    try:
        r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def device_conditions():
    fields = [
        "pstate",
        "clocks.current.sm",
        "clocks.current.memory",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "utilization.gpu",
        "memory.used",
    ]
    raw = command_output(
        ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"]
    )
    return dict(
        fields=fields,
        raw_csv=raw,
        note="Run-endpoint observations only; clocks/power are not fixed and this is not a per-sample trace",
    )


def environment():
    root = project_root()
    lock = root / "uv.lock"
    # New/uncommitted labs still need an exact executable-source identity.
    source_hash = hashlib.sha256()
    for folder, pattern in (("src", "*.py"), ("scripts", "*.py"), ("configs", "*.json")):
        for file in sorted((root / folder).rglob(pattern)):
            source_hash.update(str(file.relative_to(root)).encode())
            source_hash.update(b"\0")
            source_hash.update(file.read_bytes())
    packages = {}
    for name in ("torch", "triton", "numpy", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = None
    runtime = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu = {
            "name": p.name,
            "capability": list(torch.cuda.get_device_capability()),
            "total_memory": p.total_memory,
            "count": torch.cuda.device_count(),
        }
        boot_id = Path("/proc/sys/kernel/random/boot_id")
        runtime = {
            "hostname_sha256": hashlib.sha256(platform.node().encode()).hexdigest(),
            "boot_id_sha256": hashlib.sha256(boot_id.read_bytes()).hexdigest()
            if boot_id.is_file()
            else None,
        }
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
        "git_commit": command_output(["git", "rev-parse", "HEAD"], root),
        "git_dirty": bool(command_output(["git", "status", "--porcelain"], root)),
        "source_sha256": source_hash.hexdigest(),
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        "cuda_build": torch.version.cuda,
        "gpu": gpu,
        "runtime_identity": runtime,
        "driver": command_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


FIELDS = [
    "case_id",
    "op",
    "shape",
    "dtype",
    "backend",
    "status",
    "reason",
    "selected_config",
    "first_call_ms",
    "construction_ms",
    "selection_ms",
    "policy_decisions",
    "tuning_ms",
    "p20_ms",
    "median_ms",
    "p80_ms",
    "min_ms",
    "max_ms",
    "peak_allocated_bytes",
    "extra_peak_bytes",
    "max_abs",
    "relative_l2",
    "max_scaled_relative",
    "atol",
    "rtol",
    "tflops",
    "samples_ms",
    "wall_median_ms",
    "wall_samples_ms",
    "timing_mode",
    "compile_audit",
    "candidates",
    "kernel_names",
]


def write_rows(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False, allow_nan=False)
                    if isinstance(v, (list, dict))
                    else v
                    for k, v in row.items()
                }
            )
