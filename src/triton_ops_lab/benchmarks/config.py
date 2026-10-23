import json
from pathlib import Path

from .selection import GEMM_ABLATIONS, RULE_BACKENDS

OPS = {"vector_add", "softmax", "matmul", "attention", "conv2d", "transformer", "cnn"}
BACKENDS = {
    "torch",
    "compile",
    "compile_mm_compat",
    "triton_basic",
    "triton_grouped",
    "triton_fused",
    "triton_fixed",
    "triton_search",
    "triton_pruned",
    "triton_prepared",
    "triton_prepared_search",
    "triton_rule",
    "triton_rule_search",
    "sdpa_math",
    "sdpa_flash",
    "sdpa_efficient",
}
BACKENDS.update(GEMM_ABLATIONS)
BACKENDS.update(RULE_BACKENDS)
CASE_KEYS = {
    "id",
    "op",
    "shape",
    "dtype",
    "bias",
    "activation",
    "causal",
    "stride",
    "padding",
    "transpose",
}


def identity(case):
    # Defaults cannot disguise overlap as different JSON spellings.
    result = {"op": case["op"], "shape": case["shape"]}
    for key, default in {
        "matmul": {"bias": False, "activation": None, "transpose": False},
        "attention": {"causal": False},
        "conv2d": {"stride": 1, "padding": 0},
    }.get(case["op"], {}).items():
        result[key] = case.get(key, default)
    return json.dumps(result, sort_keys=True)


def validate_case(case):
    if not isinstance(case, dict) or set(case) - CASE_KEYS:
        raise ValueError("Unknown case fields")
    if not isinstance(case.get("id"), str) or not case["id"]:
        raise ValueError("Each case requires a nonempty id")
    op, shape = case.get("op"), case.get("shape")
    if (
        op not in OPS
        or not isinstance(shape, list)
        or not all(type(n) is int and n > 0 for n in shape)
    ):
        raise ValueError("Invalid operator or positive integer shape")
    rank = {
        "vector_add": 1,
        "softmax": 2,
        "matmul": 3,
        "attention": 4,
        "conv2d": 6,
        "transformer": 3,
        "cnn": 4,
    }[op]
    if len(shape) != rank:
        raise ValueError(f"{op} requires {rank} shape fields")
    if case.get("dtype") not in {"float16", "bfloat16"}:
        raise ValueError("GPU benchmark dtype must be float16 or bfloat16")
    allowed = {
        "matmul": {"bias", "activation", "transpose"},
        "attention": {"causal"},
        "conv2d": {"stride", "padding"},
    }
    if (set(case) - {"id", "op", "shape", "dtype"}) - allowed.get(op, set()):
        raise ValueError("Parameters do not apply to this operator")
    if any(type(case[k]) is not bool for k in ("bias", "causal", "transpose") if k in case):
        raise ValueError("bias/causal/transpose must be boolean")
    if case.get("activation") not in {None, "gelu"}:
        raise ValueError("Only tanh GELU supported")
    if op == "attention" and shape[-1] not in {32, 64, 128}:
        raise ValueError("Unsupported head dimension")
    if op == "transformer" and shape[-1] not in {64, 128, 256}:
        raise ValueError("Transformer has 2 heads of dimension 32/64/128")
    if op == "softmax" and shape[-1] > 16384:
        raise ValueError("Softmax supports <= 16384 columns")
    if op == "conv2d":
        _, _, h, w, _, kernel = shape
        s, p = case.get("stride", 1), case.get("padding", 0)
        if (
            type(s) is not int
            or type(p) is not int
            or kernel not in {1, 3}
            or s not in {1, 2}
            or p not in {0, 1}
        ):
            raise ValueError("Unsupported convolution parameters")
        if min(h, w) + 2 * p < kernel:
            raise ValueError("Invalid convolution output")


def load_config(path):
    data = json.loads(Path(path).read_text())
    if set(data) != {"name", "seed", "backends", "timing", "candidate_budget", "cases"}:
        raise ValueError("Config fields must be name/seed/backends/timing/candidate_budget/cases")
    if not isinstance(data["name"], str) or type(data["seed"]) is not int:
        raise ValueError("Invalid config name or seed")
    if (
        not isinstance(data["backends"], list)
        or not data["backends"]
        or not set(data["backends"]) <= BACKENDS
    ):
        raise ValueError("Unsupported benchmark backend")
    if len(set(data["backends"])) != len(data["backends"]):
        raise ValueError("Duplicate backends")
    timing = data["timing"]
    if set(timing) - {"warmup", "samples", "repeats", "mode"} or not {
        "warmup",
        "samples",
        "repeats",
    } <= set(timing):
        raise ValueError("timing requires warmup/samples/repeats and optional mode")
    if timing.get("mode", "cuda_event") not in {"cuda_event", "cuda_graph"}:
        raise ValueError("timing mode must be cuda_event or cuda_graph")
    if (
        any(type(timing[k]) is not int or timing[k] < 1 for k in ("warmup", "samples", "repeats"))
        or data["timing"]["samples"] < 3
    ):
        raise ValueError("Positive timing counts and >=3 samples required")
    if type(data["candidate_budget"]) is not int or not 1 <= data["candidate_budget"] <= 8:
        raise ValueError("candidate_budget must be 1..8")
    if not isinstance(data["cases"], list) or not data["cases"]:
        raise ValueError("No cases")
    for case in data["cases"]:
        validate_case(case)
    if len({c["id"] for c in data["cases"]}) != len(data["cases"]):
        raise ValueError("Duplicate case ids")
    return data


def assert_disjoint(tune, evaluation):
    overlap = {identity(c) for c in tune["cases"]} & {identity(c) for c in evaluation["cases"]}
    if overlap:
        raise ValueError("Tuning and evaluation shapes overlap")
