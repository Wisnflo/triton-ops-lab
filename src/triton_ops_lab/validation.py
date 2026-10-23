"""Predeclared correctness criteria; never inferred from measured speedups."""

import torch

# Each pair is (atol, rtol); elementwise checks AND finite outputs are required.
TOLERANCES = {
    "float64": {"default": (1e-9, 1e-8)},
    "float32": {"default": (2e-5, 2e-4), "transformer": (1e-4, 5e-4)},
    "float16": {"default": (2e-2, 2e-2), "softmax": (2e-3, 1e-2), "attention": (8e-3, 2e-2)},
    "bfloat16": {"default": (8e-2, 5e-2), "softmax": (8e-3, 3e-2), "attention": (4e-2, 5e-2)},
}


def compare(actual, expected, op="default", dtype=None):
    name = dtype or str(actual.dtype).removeprefix("torch.")
    atol, rtol = TOLERANCES[name].get(op, TOLERANCES[name]["default"])
    if actual.shape != expected.shape:
        return {"passed": False, "reason": "shape mismatch", "atol": atol, "rtol": rtol}
    a, e = actual.double(), expected.double()
    if not torch.isfinite(a).all() or not torch.isfinite(e).all():
        return {"passed": False, "reason": "nonfinite result", "atol": atol, "rtol": rtol}
    diff = (a - e).abs()
    # Stable denominators prevent near-zero reference entries from exploding the metric.
    rms = torch.mean(e.square()).sqrt().item()
    return {
        "passed": bool(torch.all(diff <= atol + rtol * e.abs()).item()),
        "max_abs": diff.max().item(),
        "relative_l2": (
            torch.linalg.vector_norm(a - e) / torch.linalg.vector_norm(e).clamp_min(1e-12)
        ).item(),
        "max_scaled_relative": (diff / e.abs().clamp_min(max(rms * 1e-3, 1e-12))).max().item(),
        "atol": atol,
        "rtol": rtol,
    }
