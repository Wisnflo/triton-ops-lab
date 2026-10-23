import json
from collections import namedtuple
from types import SimpleNamespace

from triton_ops_lab.kernels.capture import capture
from triton_ops_lab.resource_audit import profiler_result, sanitizer_result, safety_result


def test_instrumentation_must_report_success():
    assert sanitizer_result(0, "68 passed")["status"] == "failed"
    assert sanitizer_result(0, "68 passed\nERROR SUMMARY: 2 errors")["status"] == "failed"
    assert sanitizer_result(1, "68 passed\nERROR SUMMARY: 0 errors")["status"] == "failed"
    assert sanitizer_result(0, "ERROR SUMMARY: 0 errors")["status"] == "failed"
    assert (
        sanitizer_result(0, "68 passed\nInternal Sanitizer Error\nERROR SUMMARY: 0 errors")[
            "status"
        ]
        == "failed"
    )
    good = sanitizer_result(0, "68 passed\nERROR SUMMARY: 0 errors")
    assert good["status"] == "passed" and good["pytest_passed"] == 68
    shutdown = sanitizer_result(
        0, "68 passed\nException ignored in atexit callback\nERROR SUMMARY: 0 errors"
    )
    assert (
        shutdown["status"] == "passed_with_host_shutdown_error" and shutdown["host_shutdown_error"]
    )
    assert profiler_result(0, "==WARNING== No kernels were profiled", True)["status"] == "failed"
    assert profiler_result(0, "", False)["status"] == "failed"
    assert profiler_result(1, "ERR_NVGPUCTRPERM", True)["status"] == "permission_denied"


def test_capture_keeps_exact_binary_and_machine_readable_resources(tmp_path, monkeypatch):
    metadata = namedtuple("Metadata", "name shared num_warps num_stages num_ctas cluster_dims")
    compiled = SimpleNamespace(
        asm={"ptx": "kernel text", "cubin": bytes(range(256))},
        n_regs=32,
        n_spills=0,
        metadata=metadata("test_kernel", 8192, 4, 2, 1, (1, 1, 1)),
    )
    monkeypatch.setenv("TRITON_LAB_CAPTURE_DIR", str(tmp_path))
    capture(compiled, "test")
    folder = next(tmp_path.iterdir())
    assert (folder / "kernel.cubin").read_bytes() == bytes(range(256))
    r = json.loads((folder / "resources.json").read_text())
    assert r["registers"] == 32 and r["spills"] == 0
    assert r["compiler_metadata"]["shared"] == 8192
    assert r["asm_files"]["cubin"]["bytes"] == 256


def test_safety_tools_require_the_full_test_count_and_clean_tool_summary():
    log = "96 passed\n========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
    assert safety_result("racecheck", 0, log, 96)["status"] == "passed"
    assert safety_result("racecheck", 0, log, 97)["status"] == "failed"
    assert (
        safety_result(
            "racecheck",
            0,
            log.replace("0 hazards", "1 hazard").replace("0 warnings", "1 warning"),
            96,
        )["status"]
        == "failed"
    )
    assert (
        safety_result(
            "racecheck",
            0,
            "96 passed\nERROR SUMMARY: 0 errors\nRACECHECK SUMMARY: 1 hazard displayed (0 errors, 1 warning)",
            96,
        )["status"]
        == "failed"
    )
    assert (
        safety_result(
            "initcheck",
            0,
            "96 passed\n========= ERROR SUMMARY: 0 errors\n========= WARNING: incomplete tracking",
            96,
        )["status"]
        == "passed_with_tool_warning"
    )
