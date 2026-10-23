"""Bounded configuration search. Shape filtering is a hypothesis, not a speed claim."""


def gemm_ablations():
    """One-factor experiments, never a pool from which to select an eval winner."""
    base = dict(bm=64, bn=64, bk=32, warps=4, stages=2, group=8)
    options = {"triton_ablation_base": base}
    for field, values in {
        "bm": [32, 128],
        "bn": [32, 128],
        "bk": [64],
        "warps": [8],
        "stages": [1, 3, 4],
        "group": [1, 4, 16],
    }.items():
        for value in values:
            options[f"triton_ablation_{field}{value}"] = {**base, field: value}
    return options


GEMM_ABLATIONS = gemm_ablations()
RULE_BACKENDS = {
    "triton_rule",
    "triton_rule_search",
    "triton_prepared_rule",
    "triton_prepared_rule_search",
}
RULE_SEARCH_BACKENDS = {"triton_rule_search", "triton_prepared_rule_search"}
PREPARED_RULE_BACKENDS = {"triton_prepared_rule", "triton_prepared_rule_search"}


def candidates(op, shape, strategy="fixed", budget=8):
    if strategy not in {"fixed", "search", "pruned"} or not 1 <= budget <= 8:
        raise ValueError("Invalid selection strategy/budget")
    if op not in {"matmul", "attention", "conv2d"}:
        return [None]
    if op == "attention":
        options = [
            dict(bm=m, bn=n, warps=w, stages=s)
            for m, n, w, s in [
                (32, 32, 4, 2),
                (64, 32, 4, 2),
                (32, 64, 4, 2),
                (64, 64, 4, 3),
                (64, 64, 8, 2),
            ]
        ]
        extent = shape[2]
    else:
        options = [
            dict(bm=m, bn=n, bk=k, warps=w, stages=s)
            for m, n, k, w, s in [
                (32, 64, 32, 4, 2),
                (32, 32, 32, 4, 2),
                (64, 64, 32, 4, 2),
                (64, 128, 32, 8, 3),
                (128, 64, 32, 8, 3),
                (64, 64, 64, 4, 3),
            ]
        ]
        if op == "conv2d":
            # Match the public Conv default, rather than silently using GEMM's BN=64.
            options[0], options[1] = options[1], options[0]
        extent = shape[0] if op == "matmul" else shape[0] * shape[2] * shape[3]
    if strategy == "fixed":
        return options[:1]
    if strategy == "pruned":
        # Keep smallest viable tile even for tiny/non-divisible shapes.
        options = [c for c in options if c["bm"] <= max(32, extent)]
        if op == "attention" and shape[-1] == 128:
            options = [c for c in options if c["bm"] * c["bn"] <= 2048]
    return options[:budget]
