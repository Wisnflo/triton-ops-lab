import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from triton_ops_lab.analysis.report import load_results, report, speedups
from triton_ops_lab.artifacts import new_run, output_root, project_root, write_json, write_rows
from triton_ops_lab.benchmarks.config import assert_disjoint, load_config
from triton_ops_lab.benchmarks.runner import summarize
from triton_ops_lab.benchmarks.selection import candidates
from triton_ops_lab.cli import main


def synthetic_run(path):
    path.mkdir(exist_ok=True)
    write_json(
        path / "metadata.json",
        {
            "schema_version": 1,
            "kind": "synthetic",
            "complete": True,
            "environment": {"gpu": "FAKE GPU"},
            "protocol": "TEST ONLY",
        },
    )
    rows = []
    for backend, ms in [("torch", 2.0), ("triton_fixed", 1.0), ("triton_pruned", 4.0)]:
        rows.append(
            dict(
                case_id="fake-case",
                op="matmul",
                shape=[16, 16, 16],
                dtype="float16",
                backend=backend,
                status="passed",
                median_ms=ms,
                p20_ms=ms * 0.9,
                p80_ms=ms * 1.1,
                min_ms=ms * 0.8,
                max_ms=ms * 1.2,
                extra_peak_bytes=1024,
            )
        )
    rows.append(
        dict(
            case_id="fake-failure",
            op="matmul",
            shape=[17, 17, 17],
            dtype="float16",
            backend="triton_fixed",
            status="incorrect",
            reason="intentional TEST fixture",
        )
    )
    write_rows(path / "metrics.csv", rows)
    return path


def test_configs():
    root = project_root() / "configs"
    tune, evaluation = load_config(root / "tune.json"), load_config(root / "eval.json")
    assert_disjoint(tune, evaluation)
    load_config(root / "smoke.json")
    for mode in ("events", "graphs"):
        cfg = load_config(root / ("mm-compat-" + mode + ".json"))
        assert_disjoint(cfg, evaluation)
        assert "compile_mm_compat" in cfg["backends"]
        cfg = load_config(root / ("gemm-tune-" + mode + ".json"))
        assert_disjoint(cfg, evaluation)
        assert cfg["candidate_budget"] == 6
        assert "triton_prepared_search" in cfg["backends"]
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(tune, tune)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(backends=["unknown"]),
        lambda c: c.update(candidate_budget=0),
        lambda c: c["cases"][0].update(dtype="float32"),
        lambda c: c["cases"][0].update(shape=[-1]),
        lambda c: c["cases"].append(c["cases"][0]),
        lambda c: c["timing"].update(samples=1),
        lambda c: c["timing"].update(mode="unknown"),
        lambda c: c["cases"][0].update(stride=2),
    ],
)
def test_invalid_configs(tmp_path, mutation):
    cfg = copy.deepcopy(load_config(project_root() / "configs/smoke.json"))
    mutation(cfg)
    write_json(tmp_path / "bad.json", cfg)
    with pytest.raises(ValueError):
        load_config(tmp_path / "bad.json")


def test_candidates():
    for op, shape in [
        ("matmul", [17, 23, 19]),
        ("attention", [1, 2, 65, 128]),
        ("conv2d", [1, 3, 7, 9, 5, 3]),
    ]:
        fixed = candidates(op, shape)
        search = candidates(op, shape, "search", 8)
        pruned = candidates(op, shape, "pruned", 8)
        assert len(fixed) == 1
        assert all(c in search for c in pruned)
        assert 1 <= len(pruned) <= len(search)
        assert len(candidates(op, shape, "search", 2)) == 2


def test_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_LAB_OUTPUT_ROOT", str(tmp_path / "env"))
    assert output_root() == tmp_path / "env"
    assert output_root(tmp_path / "explicit") == tmp_path / "explicit"
    assert new_run(tmp_path, "x") != new_run(tmp_path, "x")
    with pytest.raises(ValueError, match="outside"):
        output_root(project_root() / "runs")
    monkeypatch.delenv("TRITON_LAB_OUTPUT_ROOT")
    import triton_ops_lab.artifacts as artifacts

    monkeypatch.setattr(artifacts, "project_root", lambda: tmp_path / "clone")
    with pytest.raises(ValueError, match="standalone"):
        output_root()


def test_stats_and_reporting(tmp_path):
    assert summarize([1, 2, 3])["median_ms"] == 2
    with pytest.raises(ValueError):
        summarize([1, float("nan"), 3])
    run = synthetic_run(tmp_path / "synthetic")
    with pytest.raises(ValueError, match="Synthetic"):
        load_results(run)
    meta, rows = load_results(run, True)
    assert sorted(v for _, v in speedups(rows)) == [0.5, 2.0]
    report_dir = report(run, True)
    text = (report_dir / "report.md").read_text()
    assert "SYNTHETIC" in text and "0.500x" in text and "intentional TEST" in text
    assert (report_dir / "metrics-float16.png").stat().st_size > 1000
    assert (report_dir / "metrics-float16.svg").stat().st_size > 1000
    # Changed precision/shape cannot acquire another row's torch baseline.
    rows[1]["dtype"] = "bfloat16"
    assert len(speedups(rows)) == 1


def test_report_ignores_inherited_notebook_backend(tmp_path, monkeypatch):
    # Must also work in a fresh interpreter before matplotlib has been imported.
    run = synthetic_run(tmp_path / "synthetic")
    monkeypatch.setenv("MPLBACKEND", "module://missing_colab_backend")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from triton_ops_lab.analysis.report import report; import sys; report(sys.argv[1], True)",
            str(run),
        ],
        check=True,
    )


def test_reject_corrupt_timings(tmp_path):
    run = synthetic_run(tmp_path)
    text = (run / "metrics.csv").read_text().replace("2.0", "nan")
    (run / "metrics.csv").write_text(text)
    with pytest.raises(ValueError):
        load_results(run, True)


def test_no_triton_import():
    code = "import sys; import triton_ops_lab; from triton_ops_lab.cli import parser; parser(); assert 'triton' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cli_dry_run(tmp_path, capsys):
    assert (
        main(
            [
                "bench",
                "--config",
                str(project_root() / "configs/smoke.json"),
                "--dry-run",
                "--output-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["dry_run"]
    assert not list(tmp_path.iterdir())


def test_cpu_smoke_cli(tmp_path, capsys):
    assert main(["check", "--device", "cpu", "--output-root", str(tmp_path)]) == 0
    message = json.loads(capsys.readouterr().out)
    assert message["passed"] and message["cases"] == 10
    assert (Path(message["output"]) / "checks.json").exists()


def test_explicit_cuda_failure(tmp_path, monkeypatch, capsys):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert main(["check", "--device", "cuda", "--output-root", str(tmp_path)]) == 2
    assert (
        main(
            [
                "bench",
                "--config",
                str(project_root() / "configs/smoke.json"),
                "--output-root",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert "no CPU fallback" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_notebook_is_thin_and_unexecuted():
    notebook = json.loads((project_root() / "notebooks/colab.ipynb").read_text())
    for cell in notebook["cells"]:
        assert cell["id"]
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            text = "".join(cell["source"])
            compile(text, "colab-cell", "exec")
            assert "@triton.jit" not in text


def test_capture_blocks_benchmark(tmp_path, monkeypatch):
    from triton_ops_lab.benchmarks.runner import run

    monkeypatch.setenv("TRITON_LAB_CAPTURE_DIR", str(tmp_path / "capture"))
    with pytest.raises(ValueError, match="capture"):
        run(load_config(project_root() / "configs/smoke.json"), tmp_path)
    assert not list(tmp_path.iterdir())


def test_smoke_failure_prevents_benchmark(tmp_path, monkeypatch):
    """Control-flow test with mocked CUDA readiness; not kernel or performance evidence."""
    import triton_ops_lab.benchmarks.runner as runner
    import triton_ops_lab.checks as checks

    monkeypatch.setattr(runner, "require_cuda", lambda: None)
    monkeypatch.setattr(
        checks, "check", lambda _: [{"passed": False, "reason": "manufactured failure"}]
    )
    monkeypatch.setattr(
        runner, "make_inputs", lambda *a, **kw: pytest.fail("benchmark ran after failed smoke")
    )
    path, passed = runner.run(load_config(project_root() / "configs/smoke.json"), tmp_path)
    assert not passed
    metadata = json.loads((path / "metadata.json").read_text())
    assert not metadata["complete"] and metadata["all_applicable_passed"] is False
    assert not (path / "metrics.csv").exists()


def test_plot_only_pairs_matching_shapes():
    rows = [
        {
            "case_id": "same",
            "op": "matmul",
            "shape": "[1,2,3]",
            "dtype": "float16",
            "backend": "torch",
            "status": "passed",
            "median_ms": 2,
        },
        {
            "case_id": "same",
            "op": "matmul",
            "shape": "[3,2,1]",
            "dtype": "float16",
            "backend": "triton_fixed",
            "status": "passed",
            "median_ms": 1,
        },
    ]
    assert speedups(rows) == []


def test_tile_config_validation():
    from triton_ops_lab.ops import _validate_config

    for op, shape in [
        ("matmul", [17, 23, 19]),
        ("attention", [1, 2, 35, 32]),
        ("conv2d", [1, 3, 7, 9, 5, 3]),
    ]:
        for cfg in candidates(op, shape, "search"):
            _validate_config(cfg, op, "triton")
    with pytest.raises(ValueError, match="only supported"):
        _validate_config({"bm": 32}, "matmul", "reference")
    with pytest.raises(ValueError):
        _validate_config(dict(bm=7, bn=32, bk=32, warps=4, stages=2), "matmul", "triton")
    with pytest.raises(ValueError):
        _validate_config(dict(bm=32, bn=32, warps=4, stages=0), "attention", "triton")


def test_environment_source_identity():
    from triton_ops_lab.artifacts import environment

    info = environment()
    assert len(info["source_sha256"]) == 64
    assert len(info["lock_sha256"]) == 64
    assert info["source_sha256"] == environment()["source_sha256"]


def test_wrong_candidate_never_timed(tmp_path, monkeypatch):
    """Mock CUDA boundaries to test gating, not GPU kernels or measured performance."""
    import torch
    import triton_ops_lab.benchmarks.runner as runner
    import triton_ops_lab.checks as checks

    cfg = load_config(project_root() / "configs/smoke.json")
    cfg["cases"] = [cfg["cases"][0]]
    cfg["backends"] = ["triton_fixed"]
    monkeypatch.setattr(runner, "require_cuda", lambda: None)
    monkeypatch.setattr(checks, "check", lambda _: [{"passed": True}])
    monkeypatch.setattr(runner, "make_inputs", lambda *a, **kw: ((), None))
    monkeypatch.setattr(runner, "oracle", lambda *a: torch.zeros(1))
    monkeypatch.setattr(runner, "make_callable", lambda *a: lambda: torch.ones(1))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        runner, "measure", lambda *a: pytest.fail("Incorrect candidate reached timer")
    )
    path, passed = runner.run(cfg, tmp_path)
    assert not passed
    import csv

    with (path / "metrics.csv").open() as f:
        row = next(csv.DictReader(f))
    assert row["status"] == "incorrect" and row["median_ms"] == ""


def test_initialization_error_is_recorded(tmp_path, monkeypatch):
    import triton_ops_lab.benchmarks.runner as runner
    import triton_ops_lab.checks as checks

    monkeypatch.setattr(runner, "require_cuda", lambda: None)
    monkeypatch.setattr(checks, "check", lambda _: [{"passed": True}])

    def fail(*args, **kwargs):
        raise RuntimeError("manufactured input allocation failure")

    monkeypatch.setattr(runner, "make_inputs", fail)
    with pytest.raises(RuntimeError, match="allocation"):
        runner.run(load_config(project_root() / "configs/smoke.json"), tmp_path)
    meta = json.loads(next(tmp_path.glob("*/metadata.json")).read_text())
    assert not meta["complete"] and "allocation failure" in meta["reason"]
