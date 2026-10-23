"""Frozen nearest-shape candidate ranking, fitted exclusively on development data.

This is a transparent experimental heuristic, not a claim of a new learning
algorithm. It has no online learning, evaluation-file access or timing calls.
"""

import hashlib
import json
import math
from pathlib import Path

POLICY_SCHEMA = 1
NEIGHBORS = 3
SHORTLIST = 2


def features(case):
    op, shape = case["op"], case["shape"]
    if not all(type(x) is int and x > 0 for x in shape):
        raise ValueError("Policy requires positive integer dimensions")
    log = math.log2
    if op == "matmul":
        m, n, k = shape
        return [
            log(m),
            log(n),
            log(k),
            0.5 * bool(case.get("bias")),
            0.5 * (case.get("activation") == "gelu"),
            0.5 * bool(case.get("transpose")),
            (m % 32) / 32,
            (n % 32) / 32,
            (k % 32) / 32,
        ]
    if op == "attention":
        b, h, n, d = shape
        return [log(n), log(d), log(b * h), float(bool(case.get("causal"))), (n % 32) / 32]
    if op == "conv2d":
        b, c, h, w, oc, kernel = shape
        stride, padding = case.get("stride", 1), case.get("padding", 0)
        oh, ow = (h + 2 * padding - kernel) // stride + 1, (w + 2 * padding - kernel) // stride + 1
        if min(oh, ow) <= 0:
            raise ValueError("Invalid convolution extent")
        return [
            log(b * oh * ow),
            log(oc),
            log(c * kernel * kernel),
            log(kernel),
            0.5 * log(stride),
            0.5 * padding,
            ((b * oh * ow) % 32) / 32,
            (oc % 32) / 32,
        ]
    raise ValueError("Policy only selects matmul, attention and conv2d kernels")


def config_key(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


class FrozenPolicy:
    def __init__(self, data, digest=None):
        if data.get("schema_version") != POLICY_SCHEMA or data.get("purpose") != "development_only":
            raise ValueError("Only frozen development-trained policies are accepted")
        if data.get("neighbors") != NEIGHBORS or data.get("shortlist") != SHORTLIST:
            raise ValueError("Policy algorithm/hyperparameters changed")
        self.data, self.sha256 = data, digest

    @classmethod
    def load(cls, filename):
        content = Path(filename).read_bytes()
        data = json.loads(content)
        if (
            data.get("feature_implementation_sha256")
            != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ):
            raise ValueError("Frozen policy feature implementation does not match this source")
        return cls(data, hashlib.sha256(content).hexdigest())

    def rank(self, case):
        vector = features(case)
        pool_key = case["op"] + ":" + case["dtype"]
        if pool_key not in self.data["groups"]:
            raise ValueError("Frozen policy has no coverage for this operator/dtype")
        group = self.data["groups"][pool_key]
        donors = []
        for sample in group["samples"]:
            if len(sample["features"]) != len(vector):
                raise ValueError("Incompatible policy feature schema")
            distance = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(vector, sample["features"], strict=True))
            )
            donors.append((distance, sample))
        donors.sort(key=lambda x: (x[0], x[1]["identity"]))
        if not donors:
            raise ValueError("No development donors")
        selected = donors[:NEIGHBORS]
        exact = [x for x in selected if x[0] < 1e-12]
        if exact:
            selected = exact
        weight_sum = sum(1 / (0.25 + d) ** 2 for d, _ in selected)
        scores = []
        for index, cfg in enumerate(group["pool"]):
            score = (
                sum(sample["relative_costs"][index] / (0.25 + d) ** 2 for d, sample in selected)
                / weight_sum
            )
            if not math.isfinite(score) or score <= 0:
                raise ValueError("Invalid frozen candidate cost")
            scores.append(dict(config=dict(cfg), predicted_relative_cost=score))
        scores.sort(key=lambda x: (x["predicted_relative_cost"], config_key(x["config"])))
        return scores

    def choose(self, case):
        return self.rank(case)[0]["config"]

    def shortlist(self, case):
        result = [x["config"] for x in self.rank(case)[:SHORTLIST]]
        default = self.data["groups"][case["op"] + ":" + case["dtype"]]["pool"][0]
        if default not in result:
            result.append(dict(default))
        return result
