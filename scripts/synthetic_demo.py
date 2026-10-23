"""Create explicitly synthetic data ONLY to exercise report generation."""

import argparse

from triton_ops_lab.artifacts import new_run, output_root, write_json, write_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root")
    args = p.parse_args()
    path = new_run(output_root(args.output_root), "SYNTHETIC-analysis-test")
    write_json(
        path / "metadata.json",
        {
            "schema_version": 1,
            "kind": "synthetic",
            "complete": True,
            "environment": {"gpu": "FAKE GPU — NO MEASUREMENTS"},
            "protocol": "SYNTHETIC TEST ONLY",
        },
    )
    rows = []
    for name, scale in [("fixture-A", 1), ("fixture-B", 2)]:
        for backend, ms in [("torch", 2.0), ("triton_fixed", 1.0), ("triton_pruned", 3.0)]:
            t = ms * scale
            rows.append(
                dict(
                    case_id=name,
                    op="matmul",
                    shape=[16, 16, 16],
                    dtype="float16",
                    backend=backend,
                    status="passed",
                    median_ms=t,
                    p20_ms=t * 0.9,
                    p80_ms=t * 1.1,
                    min_ms=t * 0.8,
                    max_ms=t * 1.2,
                    extra_peak_bytes=scale * 1024**2,
                )
            )
    rows.append(
        dict(
            case_id="fixture-failure",
            op="matmul",
            shape=[17, 17, 17],
            dtype="float16",
            backend="triton_fixed",
            status="incorrect",
            reason="manufactured test failure",
        )
    )
    write_rows(path / "metrics.csv", rows)
    print(path)


if __name__ == "__main__":
    main()
