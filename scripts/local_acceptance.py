"""Re-run local gates and store raw evidence outside the source repository.

Run with uv run --frozen python scripts/local_acceptance.py.
No GPU or cloud work is started. Expected CUDA failures are required on macOS.
"""

import json
import subprocess
from pathlib import Path

from triton_ops_lab.artifacts import environment, new_run, output_root, project_root, write_json


def main():
    root = project_root()
    path = new_run(output_root(), "local-acceptance")
    results = []

    def execute(label, args, expected=0):
        result = subprocess.run(args, cwd=root, capture_output=True, text=True)
        (path / (label + ".stdout.txt")).write_text(result.stdout)
        (path / (label + ".stderr.txt")).write_text(result.stderr)
        entry = {
            "label": label,
            "command": args,
            "exit_code": result.returncode,
            "expected_exit_code": expected,
            "passed": result.returncode == expected,
        }
        results.append(entry)
        write_json(
            path / "summary.json",
            {"environment": environment(), "checks": results, "complete": False},
        )
        if not entry["passed"]:
            raise RuntimeError(f"{label} failed; evidence: {path}")
        return result.stdout

    uv = ["uv", "run", "--frozen"]
    execute("sync", ["uv", "sync", "--frozen"])
    execute("ruff", [*uv, "ruff", "check", "."])
    execute("format", [*uv, "ruff", "format", "--check", "src", "scripts", "tests"])
    execute("syntax", [*uv, "python", "-m", "compileall", "-q", "src", "scripts", "tests"])
    execute("pytest", [*uv, "pytest", "-q", "--junitxml", str(path / "pytest.xml")])
    for command in [None, "doctor", "check", "bench", "analyze"]:
        execute(
            "help-" + (command or "root"),
            [*uv, "triton-lab", *([command] if command else []), "--help"],
        )
    for script in ["gpu_tools", "synthetic_demo"]:
        execute("help-" + script, [*uv, "python", "scripts/" + script + ".py", "--help"])
    execute("doctor", [*uv, "triton-lab", "doctor"])
    cpu = json.loads(execute("cpu-check", [*uv, "triton-lab", "check", "--device", "cpu"]))
    for name in ["smoke", "tune", "eval"]:
        execute(
            "dry-run-" + name,
            [*uv, "triton-lab", "bench", "--config", f"configs/{name}.json", "--dry-run"],
        )
    execute("cuda-check-refusal", [*uv, "triton-lab", "check", "--device", "cuda"], expected=2)
    execute(
        "cuda-bench-refusal",
        [*uv, "triton-lab", "bench", "--config", "configs/smoke.json"],
        expected=2,
    )
    execute("gpu-tools-refusal", [*uv, "python", "scripts/gpu_tools.py", "capture"], expected=2)
    synthetic = execute("synthetic", [*uv, "python", "scripts/synthetic_demo.py"]).strip()
    execute("synthetic-refusal", [*uv, "triton-lab", "analyze", synthetic], expected=2)
    analysis = execute(
        "analysis", [*uv, "triton-lab", "analyze", synthetic, "--allow-synthetic"]
    ).strip()
    assert (Path(analysis) / "report.md").is_file()
    assert (Path(analysis) / "metrics-float16.png").is_file()
    write_json(
        path / "summary.json",
        {
            "environment": environment(),
            "checks": results,
            "complete": True,
            "cpu_check": cpu["output"],
            "synthetic_run": synthetic,
            "analysis": analysis,
            "gpu_verified": False,
        },
    )
    print(path)


if __name__ == "__main__":
    main()
