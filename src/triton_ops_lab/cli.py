import argparse
import json
import sys

from .artifacts import environment, new_run, output_root, write_json


def parser():
    p = argparse.ArgumentParser(
        description="Triton operator research: CPU preparation / explicit CUDA measurements"
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Inspect local environment without allocating GPU workloads")
    check = sub.add_parser("check", help="Numerical correctness smoke checks")
    check.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    check.add_argument("--output-root")
    bench = sub.add_parser("bench", help="CUDA benchmarking; use --dry-run on Mac")
    bench.add_argument("--config", required=True)
    bench.add_argument("--dry-run", action="store_true")
    bench.add_argument("--output-root")
    analysis = sub.add_parser("analyze", help="Analyze a single run; no cross-hardware aggregation")
    analysis.add_argument("run")
    analysis.add_argument("--allow-synthetic", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            from .ops import UnsupportedBackend, require_cuda

            info = environment()
            local_ready = (
                sys.version_info[:2] == (3, 12)
                and all(info["packages"][name] for name in ("torch", "numpy", "matplotlib"))
                and info["lock_sha256"] is not None
            )
            try:
                require_cuda()
                cuda = {"ready": True}
            except UnsupportedBackend as exc:
                cuda = {"ready": False, "reason": str(exc)}
            print(
                json.dumps(
                    {
                        "environment": info,
                        "local_ready": local_ready,
                        "cuda": cuda,
                        "gpu_kernels": "draft; requires real GPU correctness and profiling",
                    },
                    indent=2,
                )
            )
            return 0 if local_ready else 2
        elif args.command == "check":
            from .checks import check

            rows = check(args.device)
            path = new_run(output_root(args.output_root), "check-" + args.device)
            passed = all(r["passed"] for r in rows)
            write_json(
                path / "checks.json",
                {
                    "device": args.device,
                    "environment": environment(),
                    "passed": passed,
                    "results": rows,
                },
            )
            print(json.dumps({"passed": passed, "cases": len(rows), "output": str(path)}, indent=2))
            return 0 if passed else 1
        elif args.command == "bench":
            from .benchmarks.config import load_config
            from .benchmarks.selection import candidates

            cfg = load_config(args.config)
            root = output_root(args.output_root)
            if args.dry_run:
                print(
                    json.dumps(
                        {
                            "dry_run": True,
                            "output_root": str(root),
                            "config": cfg,
                            "search_candidates": {
                                c["id"]: len(
                                    candidates(
                                        c["op"], c["shape"], "search", cfg["candidate_budget"]
                                    )
                                )
                                for c in cfg["cases"]
                            },
                        },
                        indent=2,
                    )
                )
            else:
                from .benchmarks.runner import run

                path, passed = run(cfg, root)
                print(json.dumps({"output": str(path), "all_applicable_passed": passed}, indent=2))
                return 0 if passed else 1
        elif args.command == "analyze":
            from .analysis.report import report

            print(report(args.run, args.allow_synthetic))
    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
