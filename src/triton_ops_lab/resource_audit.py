"""Evidence gates for instrumented GPU tools; no CUDA imports or timing claims."""

import re


def sanitizer_result(returncode, log):
    summaries = [int(x) for x in re.findall(r"ERROR SUMMARY:\s*(\d+) errors?", log)]
    tests = re.findall(r"(\d+) passed", log)
    rejected = any(
        message in log
        for message in (
            "Internal Sanitizer Error",
            "No attachable process",
            "Target application terminated before first instrumented API call",
        )
    )
    passed = (
        returncode == 0
        and bool(summaries)
        and all(x == 0 for x in summaries)
        and bool(tests)
        and int(tests[-1]) > 0
        and not rejected
    )
    shutdown_error = "Exception ignored in atexit callback" in log
    return dict(
        status="passed_with_host_shutdown_error"
        if passed and shutdown_error
        else "passed"
        if passed
        else "failed",
        exit_code=returncode,
        error_summaries=summaries,
        pytest_passed=int(tests[-1]) if tests else None,
        host_shutdown_error=shutdown_error,
        note="memcheck only; not racecheck/initcheck/synccheck or a proof for all inputs",
    )


def profiler_result(returncode, log, report_exists):
    if "ERR_NVGPUCTRPERM" in log:
        status = "permission_denied"
    elif returncode != 0 or "No kernels were profiled" in log or not report_exists:
        status = "failed"
    else:
        status = "collected"
    return dict(
        status=status, exit_code=returncode, note="Instrumented metrics are not benchmark times"
    )


def safety_result(tool, returncode, log, expected_tests):
    result = sanitizer_result(returncode, log)
    if tool == "racecheck":
        hazards = re.findall(
            r"RACECHECK SUMMARY:\s*(\d+) hazards? displayed \((\d+) errors?,\s*(\d+) warnings?\)",
            log,
        )
        result["race_summaries"] = [[int(v) for v in group] for group in hazards]
        result["status"] = "failed"
        if (
            hazards
            and all(all(int(v) == 0 for v in group) for group in hazards)
            and returncode == 0
            and "Internal Sanitizer Error" not in log
        ):
            result["status"] = (
                "passed_with_host_shutdown_error" if result["host_shutdown_error"] else "passed"
            )
    warnings = re.findall(r"^=+\s*(?:WARNING|Warning)[: ].*", log, re.MULTILINE)
    result["tool_warnings"] = warnings
    if result["pytest_passed"] != expected_tests or expected_tests <= 0:
        result["status"] = "failed"
    elif warnings and result["status"] == "passed":
        result["status"] = "passed_with_tool_warning"
    result["tool"] = tool
    result["expected_gpu_tests"] = expected_tests
    result["note"] = (
        "Instrumented GPU tests; only the selected sanitizer tool and its configured address spaces are covered"
    )
    return result
