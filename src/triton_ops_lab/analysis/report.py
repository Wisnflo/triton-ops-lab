import csv
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

from ..artifacts import new_run


def load_results(run, allow_synthetic=False):
    run = Path(run)
    meta = json.loads((run / "metadata.json").read_text())
    if meta.get("schema_version") != 1 or meta.get("kind") not in {"gpu_measured", "synthetic"}:
        raise ValueError("Unknown result schema/kind")
    if meta["kind"] == "synthetic" and not allow_synthetic:
        raise ValueError(
            "Synthetic data is not GPU evidence; pass --allow-synthetic for analysis testing"
        )
    if meta.get("kind") == "gpu_measured" and not meta.get("environment", {}).get("gpu"):
        raise ValueError("Measured GPU results must identify the GPU")
    with (run / "metrics.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("No result rows")
    seen = set()
    for row in rows:
        key = (row["case_id"], row["dtype"], row["backend"])
        if key in seen:
            raise ValueError("Duplicate case/backend rows")
        seen.add(key)
        if row["status"] not in {"passed", "incorrect", "failed", "unsupported"}:
            raise ValueError("Invalid measurement status")
        if row["status"] == "passed":
            for name in ("median_ms", "p20_ms", "p80_ms", "min_ms", "max_ms"):
                row[name] = float(row[name])
                if not math.isfinite(row[name]) or row[name] <= 0:
                    raise ValueError("Passed timings must be finite and positive")
            if (
                not row["min_ms"]
                <= row["p20_ms"]
                <= row["median_ms"]
                <= row["p80_ms"]
                <= row["max_ms"]
            ):
                raise ValueError("Inconsistent timing quantiles")
    return meta, rows


def speedups(rows):
    # case id alone is insufficient: enforce identical op/shape/dtype too.
    def key(r):
        return r["case_id"], r["op"], r["shape"], r["dtype"]

    baseline = {
        key(r): r["median_ms"] for r in rows if r["backend"] == "torch" and r["status"] == "passed"
    }
    return [
        (r, baseline[key(r)] / r["median_ms"])
        for r in rows
        if r["status"] == "passed" and r["backend"] != "torch" and key(r) in baseline
    ]


def report(run, allow_synthetic=False):
    run = Path(run).resolve()
    meta, rows = load_results(run, allow_synthetic)
    # All generated products stay with their run, never inside the source tree.
    from ..artifacts import project_root

    if run == project_root() or project_root() in run.parents:
        raise ValueError("Analysis outputs must be outside source repository")
    out = new_run(run, "analysis")
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "triton-lab-matplotlib")
    )
    # Colab's notebook backend is absent in the isolated uv interpreter. Set the
    # backend before importing matplotlib, which validates MPLBACKEND on import.
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    synthetic = meta["kind"] == "synthetic"
    banner = (
        "SYNTHETIC — TEST DATA, NOT GPU RESULTS"
        if synthetic
        else "GPU measured — correctness-gated"
    )
    counts = Counter(r["status"] for r in rows)
    applicable = len(rows) - counts["unsupported"]
    lines = [
        f"# {banner}",
        "",
        f"Source: `{run}`",
        "",
        f"Complete: {meta.get('complete', False)}",
        "",
        f"Environment: `{json.dumps(meta.get('environment', {}), ensure_ascii=False)}`",
        "",
        f"Protocol: `{meta.get('protocol')}`",
        "",
        f"Compile policy: `{meta.get('compile_policy', 'legacy; no fullgraph guarantee')}`",
        "",
        "compile_mm_compat, when present, is an explicitly opaque external FP32-output GEMM plus compiled surrounding operations. It is a distinct baseline from native compile.",
        "",
        f"Status counts: `{dict(counts)}`",
        "",
        f"Pass rate (excludes unsupported): {counts['passed']}/{applicable}",
        "",
        "Each row is compared only to its same-case, same-dtype torch baseline.",
        "",
        "| Case | dtype | backend | event median ms | wall median ms | p20–p80 ms | speedup vs torch | extra peak bytes |",
        "|---|---|---|---:|---:|---|---:|---:|",
    ]
    ratios = {(r["case_id"], r["dtype"], r["backend"]): v for r, v in speedups(rows)}
    passed = [r for r in rows if r["status"] == "passed"]
    for r in passed:
        ratio = ratios.get((r["case_id"], r["dtype"], r["backend"]))
        ratio_text = f"{ratio:.3f}" if ratio is not None else "—"
        wall = f"{float(r['wall_median_ms']):.5f}" if r.get("wall_median_ms") else "—"
        lines.append(
            f"| {r['case_id']} | {r['dtype']} | {r['backend']} | {r['median_ms']:.5f} | {wall} | {r['p20_ms']:.5f}–{r['p80_ms']:.5f} | {ratio_text} | {r.get('extra_peak_bytes', '')} |"
        )
    lines += [
        "",
        "Event timings include any host submission gaps. Wall timings include synchronization.",
        "CUDA Graph protocol replays static captured calls; it excludes Python dispatch and allocation overhead.",
        "Memory peaks and kernel names are measured on ordinary calls outside graph replay.",
        "Never pool different protocols or describe graph replay as ordinary API latency.",
    ]
    lines += [
        "",
        "## Construction and search cost",
        "",
        "Construction is the selected callable's factory wall time; first_call_ms is separate.",
        "Search includes candidate construction/compilation/checks/sampling, and may use warm caches.",
        "Prepared backends bind input objects and launch parameters; per-call metadata guards and fresh outputs remain.",
        "",
        "| Case | backend | construction ms | search ms | selected config |",
        "|---|---|---:|---:|---|",
    ]
    for r in passed:
        if r.get("construction_ms"):
            lines.append(
                f"| {r['case_id']} | {r['backend']} | {float(r['construction_ms']):.3f} | {float(r.get('tuning_ms') or 0):.3f} | {r.get('selected_config', '')} |"
            )
    lines += ["", "## Regressions (<1x)", ""]
    lines += [
        f"- {r['case_id']} / {r['dtype']} / {r['backend']}: {v:.3f}x"
        for r, v in speedups(rows)
        if v < 1
    ] or ["None in the available paired rows."]
    lines += ["", "## Failed / unsupported / incorrect", ""]
    lines += [
        f"- {r['case_id']} / {r['backend']}: {r['status']} — {r.get('reason', '')}"
        for r in rows
        if r["status"] != "passed"
    ] or ["None."]
    if synthetic:
        lines += [
            "",
            "These numbers were manufactured to test reporting. No performance conclusion is valid.",
        ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    if passed:
        # One figure per dtype prevents mixed-precision pooled comparisons.
        for dtype in sorted({r["dtype"] for r in passed}):
            group = [r for r in passed if r["dtype"] == dtype]
            fig, axes = plt.subplots(1, 3, figsize=(16, max(4, len(group) * 0.28)))
            labels = [r["case_id"] + " / " + r["backend"] for r in group]
            medians = [r["median_ms"] for r in group]
            axes[0].barh(labels, medians)
            axes[0].set_xlabel("Latency (ms)")
            axes[1].barh(
                labels,
                [
                    ratios.get(
                        (r["case_id"], dtype, r["backend"]),
                        1 if r["backend"] == "torch" else float("nan"),
                    )
                    for r in group
                ],
            )
            axes[1].axvline(1, color="black", linestyle="--")
            axes[1].set_xlabel("Speedup vs same-case torch")
            axes[2].barh(labels, [float(r.get("extra_peak_bytes") or 0) / 2**20 for r in group])
            axes[2].set_xlabel("Extra peak allocated memory (MiB)")
            for ax in axes[1:]:
                ax.tick_params(axis="y", labelleft=False)
            fig.suptitle(banner + " / " + dtype)
            fig.tight_layout()
            fig.savefig(out / f"metrics-{dtype}.png", dpi=140)
            fig.savefig(out / f"metrics-{dtype}.svg")
            plt.close(fig)
    return out
