import gc
import os
import time

import numpy as np
import torch

from ..artifacts import (
    device_conditions,
    environment,
    new_run,
    project_root,
    write_json,
    write_rows,
)
from ..ops import require_cuda
from ..validation import compare
from ..policy import FrozenPolicy
from .cases import applicable, make_callable, make_inputs, oracle
from .selection import GEMM_ABLATIONS, RULE_BACKENDS, RULE_SEARCH_BACKENDS, candidates


def summarize(samples):
    a = np.asarray(samples, dtype=float)
    if len(a) < 3 or not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError("Timing samples must be >=3 finite positive values")
    return dict(
        p20_ms=float(np.quantile(a, 0.2)),
        median_ms=float(np.median(a)),
        p80_ms=float(np.quantile(a, 0.8)),
        min_ms=float(a.min()),
        max_ms=float(a.max()),
    )


def measure(fn, timing, expected=None, case=None):
    mode = timing.get("mode", "cuda_event")
    for _ in range(timing["warmup"]):
        fn()
    torch.cuda.synchronize()
    graph = None
    if mode == "cuda_graph":
        if expected is None or case is None:
            raise ValueError("CUDA Graph timing requires a replay correctness oracle")
        # Warm the allocator/library state on a side stream before capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(timing["warmup"]):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(timing["repeats"]):
                captured_output = fn()
        graph.replay()
        torch.cuda.synchronize()
        accuracy = compare(captured_output, expected, case["op"], case["dtype"])
        if not accuracy["passed"]:
            raise RuntimeError(f"CUDA Graph replay correctness failed: {accuracy}")
    samples, wall_samples = [], []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # Materialize event handles outside measured samples.
    start.record()
    end.record()
    end.synchronize()
    for _ in range(timing["samples"]):
        begin = time.perf_counter()
        start.record()
        if graph is None:
            for _ in range(timing["repeats"]):
                fn()
        else:
            graph.replay()
        end.record()
        end.synchronize()
        wall_samples.append((time.perf_counter() - begin) * 1000 / timing["repeats"])
        samples.append(start.elapsed_time(end) / timing["repeats"])
    return {
        **summarize(samples),
        "samples_ms": samples,
        "wall_median_ms": summarize(wall_samples)["median_ms"],
        "wall_samples_ms": wall_samples,
        "timing_mode": mode,
    }


def flops(case):
    op, s = case["op"], case["shape"]
    if op == "matmul":
        m, n, k = s
        return 2 * m * n * k
    if op == "attention":
        b, h, n, d = s
        # Useful matmul FLOPs, excludes softmax and masked causal pairs.
        pairs = n * (n + 1) // 2 if case.get("causal") else n * n
        return 4 * b * h * pairs * d
    if op == "conv2d":
        b, c, h, w, oc, k = s
        p, stride = case.get("padding", 0), case.get("stride", 1)
        return (
            2
            * b
            * oc
            * ((h + 2 * p - k) // stride + 1)
            * ((w + 2 * p - k) // stride + 1)
            * c
            * k
            * k
        )
    return None


def observed_kernels(fn):
    # Separate from timing. Record actual dispatched operators/device kernels for auditing.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    names = {
        event.name
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA or "scaled_dot_product" in event.name
    }
    return sorted(names)


def run(config, root):
    if os.environ.get("TRITON_LAB_CAPTURE_DIR"):
        raise ValueError("Disable TRITON_LAB_CAPTURE_DIR before timing; capture changes overhead")
    require_cuda()  # Fail before creating a misleading empty performance run on Mac.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False
    path = new_run(root, config["name"])
    metadata = {
        "schema_version": 1,
        "kind": "gpu_measured",
        "environment": environment(),
        "config": config,
        "output_directory": str(path),
        "complete": False,
        "protocol": (
            "cuda-graph-replay-v2"
            if config["timing"].get("mode") == "cuda_graph"
            else "cuda-events-with-wall-v2"
        ),
        "compile_policy": "fullgraph-static-single-graph-reset-per-case-v2",
        "compile_mm_compat_policy": "explicit-opaque-external-fp32-mm-v1",
        "precision": "fp32-accum-no-tf32",
        "device_conditions_start": device_conditions(),
        "compilation_environment": {
            k: os.environ.get(k)
            for k in (
                "TORCHINDUCTOR_COMPILE_THREADS",
                "TRITON_CACHE_DIR",
                "TORCHINDUCTOR_CACHE_DIR",
            )
        },
    }
    write_json(path / "metadata.json", metadata)
    rows = []
    try:
        policy = None
        if any(x in RULE_BACKENDS for x in config["backends"]):
            begin = time.perf_counter()
            policy = FrozenPolicy.load(project_root() / "configs/policy-v1.json")
            metadata["frozen_policy"] = dict(
                sha256=policy.sha256,
                load_ms=(time.perf_counter() - begin) * 1000,
                training_environment=policy.data["training_environment"],
            )
        from ..checks import check

        preflight = check("cuda")
        write_json(path / "preflight.json", preflight)
        if not all(r["passed"] for r in preflight):
            metadata["all_applicable_passed"] = False
            metadata["reason"] = "GPU smoke correctness gate failed"
            return path, False
        for case in config["cases"]:
            args, model = make_inputs(case, "cuda", seed=config["seed"])
            expected = oracle(case, args, model)
            for backend in config["backends"]:
                row = {
                    "case_id": case["id"],
                    "op": case["op"],
                    "shape": case["shape"],
                    "dtype": case["dtype"],
                    "backend": backend,
                }
                if not applicable(case, backend):
                    row.update(status="unsupported", reason="backend does not apply to operator")
                    rows.append(row)
                    write_rows(path / "metrics.csv", rows)
                    continue
                strategy = {
                    "triton_search": "search",
                    "triton_pruned": "pruned",
                    "triton_prepared_search": "search",
                    "triton_rule_search": "rule_search",
                    "triton_prepared_rule_search": "rule_search",
                }.get(backend, "fixed")
                tuning_start = time.perf_counter()
                configs = (
                    candidates(case["op"], case["shape"], strategy, config["candidate_budget"])
                    if backend.startswith("triton_") and strategy != "rule_search"
                    else [None]
                )
                if backend in GEMM_ABLATIONS:
                    configs = [dict(GEMM_ABLATIONS[backend])]
                if backend in RULE_BACKENDS:
                    if case["op"] in {"matmul", "attention", "conv2d"}:
                        ranking = policy.rank(case)
                        configs = (
                            [ranking[0]["config"]]
                            if backend not in RULE_SEARCH_BACKENDS
                            else policy.shortlist(case)[: config["candidate_budget"]]
                        )
                        row["policy_decisions"] = dict(ranking=ranking, policy_sha256=policy.sha256)
                    else:
                        configs = [None]
                row["selection_ms"] = (time.perf_counter() - tuning_start) * 1000
                attempts, best = [], None
                try:
                    audit = None
                    for cfg in configs:
                        attempt = {"config": cfg}
                        try:
                            setup_begin = time.perf_counter()
                            fn = (
                                make_callable(case, args, model, backend, cfg, policy=policy)
                                if policy is not None and backend in RULE_BACKENDS
                                else make_callable(case, args, model, backend, cfg)
                            )
                            construction_ms = (time.perf_counter() - setup_begin) * 1000
                            audit = getattr(fn, "compile_audit", None)
                            torch.cuda.synchronize()
                            begin = time.perf_counter()
                            actual = fn()
                            if audit is not None:
                                audit.require_single_graph()
                            torch.cuda.synchronize()
                            first_ms = (time.perf_counter() - begin) * 1000
                            accuracy = compare(actual, expected, case["op"], case["dtype"])
                            del actual
                            attempt.update(
                                first_call_ms=first_ms,
                                construction_ms=construction_ms,
                                accuracy=accuracy,
                            )
                            if not accuracy["passed"]:
                                attempt["status"] = "incorrect"
                            else:
                                measured = measure(fn, config["timing"], expected, case)
                                if audit is not None:
                                    audit.require_single_graph()
                                attempt.update(status="passed", median_ms=measured["median_ms"])
                                if best is None or measured["median_ms"] < best[0]:
                                    best = (
                                        measured["median_ms"],
                                        fn,
                                        cfg,
                                        accuracy,
                                        first_ms,
                                        construction_ms,
                                    )
                        except Exception as exc:
                            attempt.update(status="failed", reason=str(exc)[:2000])
                        if audit is not None:
                            row["compile_audit"] = audit.snapshot()
                        attempts.append(attempt)
                    row.update(
                        candidates=attempts,
                        tuning_ms=(time.perf_counter() - tuning_start) * 1000
                        if strategy != "fixed"
                        else 0,
                    )
                    if best is None:
                        row.update(
                            status="incorrect"
                            if any(a["status"] == "incorrect" for a in attempts)
                            else "failed",
                            reason=(
                                attempts[0].get("reason", "Correctness failed; see candidates")
                                if len(attempts) == 1
                                else "No candidate passed correctness; see candidates"
                            ),
                        )
                    else:
                        _, fn, cfg, accuracy, first_ms, construction_ms = best
                        if hasattr(fn, "kernel_plan"):
                            row["policy_decisions"] = dict(
                                plan=fn.kernel_plan, policy_sha256=policy.sha256
                            )
                        # Independent final timing avoids reporting the winning search sample.
                        measured = measure(fn, config["timing"], expected, case)
                        gc.collect()
                        torch.cuda.synchronize()
                        allocated = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        value = fn()
                        torch.cuda.synchronize()
                        peak = torch.cuda.max_memory_allocated()
                        del value
                        if audit is not None:
                            audit.require_single_graph()
                            row["compile_audit"] = audit.snapshot()
                        row.update(
                            status="passed",
                            selected_config=cfg,
                            first_call_ms=first_ms,
                            construction_ms=construction_ms,
                            peak_allocated_bytes=peak,
                            extra_peak_bytes=peak - allocated,
                            **measured,
                            **{k: v for k, v in accuracy.items() if k not in {"passed", "reason"}},
                        )
                        count = flops(case)
                        if count:
                            row["tflops"] = count / (measured["median_ms"] * 1e9)
                        try:
                            row["kernel_names"] = observed_kernels(fn)
                        except RuntimeError as exc:
                            row["kernel_names"] = ["profiling unavailable: " + str(exc)[:500]]
                        if audit is not None:
                            audit.require_single_graph()
                            row["compile_audit"] = audit.snapshot()
                except Exception as exc:
                    row.update(status="failed", reason=str(exc)[:2000])
                    # A late audit/profiling failure must not retain usable timings.
                    for key in (
                        "median_ms",
                        "p20_ms",
                        "p80_ms",
                        "min_ms",
                        "max_ms",
                        "samples_ms",
                        "wall_median_ms",
                        "wall_samples_ms",
                        "tflops",
                    ):
                        row.pop(key, None)
                rows.append(row)
                write_rows(path / "metrics.csv", rows)
            del expected, args, model
            torch.cuda.empty_cache()
        metadata["complete"] = True
        metadata["all_applicable_passed"] = all(
            r["status"] in {"passed", "unsupported"} for r in rows
        )
    except Exception as exc:
        metadata["reason"] = f"{type(exc).__name__}: {exc}"
        metadata["all_applicable_passed"] = False
        raise
    finally:
        metadata["device_conditions_end"] = device_conditions()
        write_json(path / "metadata.json", metadata)
    return path, metadata["all_applicable_passed"]
