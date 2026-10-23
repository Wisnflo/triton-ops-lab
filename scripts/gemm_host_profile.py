"""Instrument warmed Python submission separately from benchmark timing.

cProfile overhead and driver waits are included. These are diagnostic host-call
costs, not uninstrumented CPU time or GPU kernel latency. No cloud lifecycle work.
"""

import argparse
import cProfile
import pstats
import time

import torch

from triton_ops_lab.artifacts import environment, new_run, output_root, write_json
from triton_ops_lab.benchmarks.cases import make_callable, make_inputs, oracle
from triton_ops_lab.benchmarks.config import load_config
from triton_ops_lab.ops import require_cuda
from triton_ops_lab.validation import compare


def profile(config_path, case_id, calls, root):
    require_cuda()
    cfg = load_config(config_path)
    case = next(c for c in cfg["cases"] if c["id"] == case_id)
    if case["op"] != "matmul" or not 1 <= calls <= 1000:
        raise ValueError("Select a matmul case and 1..1000 calls")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    args, model = make_inputs(case, "cuda", seed=cfg["seed"])
    expected = oracle(case, args, model)
    path = new_run(root, "host-profile")
    meta = dict(
        environment=environment(),
        case=case,
        calls=calls,
        complete=False,
        note="Instrumented warmed host submission; not benchmark latency",
        results={},
    )
    write_json(path / "metadata.json", meta)
    for backend in ("torch", "triton_fused", "triton_prepared"):
        fn = make_callable(case, args, model, backend)
        accuracy = compare(fn(), expected, case["op"], case["dtype"])
        if not accuracy["passed"]:
            raise RuntimeError(f"{backend} correctness failed: {accuracy}")
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        profiler = cProfile.Profile()
        started = time.perf_counter()
        profiler.enable()
        for _ in range(calls):
            fn()
        profiler.disable()
        host_ms = (time.perf_counter() - started) * 1000 / calls
        torch.cuda.synchronize()
        profiler.dump_stats(str(path / (backend + ".prof")))
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        with (path / (backend + ".txt")).open("w") as output:
            stats.stream = output
            stats.print_stats(50)
        selected = []
        for (filename, line, name), (primitive, count, own, cumulative, _) in stats.stats.items():
            if name in {
                "_validate",
                "_validate_matmul",
                "require_cuda",
                "check",
                "tensor_signature",
            }:
                selected.append(
                    dict(
                        file=filename,
                        line=line,
                        function=name,
                        calls=count,
                        primitive_calls=primitive,
                        self_ms=own * 1000,
                        cumulative_ms=cumulative * 1000,
                    )
                )
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as trace:
            fn()
            torch.cuda.synchronize()
        trace.export_chrome_trace(str(path / (backend + "-trace.json")))
        meta["results"][backend] = dict(
            instrumented_host_call_ms=host_ms, selected_functions=selected, accuracy=accuracy
        )
        write_json(path / "metadata.json", meta)
    meta["complete"] = True
    write_json(path / "metadata.json", meta)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gemm-tune-events.json")
    parser.add_argument("--case", default="fused-tail-float16")
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    print(profile(args.config, args.case, args.calls, output_root(args.output_root)))
