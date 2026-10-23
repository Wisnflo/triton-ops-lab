"""Verify separate, untimed observations without replacing original timing rows."""

import hashlib
import json
from pathlib import Path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Dispatch supplement references an invalid artifact")
    return path


def load_dispatch_supplement(study, state):
    study = Path(study)
    root = study.parent / "dispatch-supplement"
    path = root / "kernel-observation-supplement.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if (
        data.get("schema") != "missing-dispatch-observation-v1"
        or not data.get("complete")
        or data["purpose"] != state["suite"]["purpose"]
        or data["suite_status_sha256"] != _sha(study / "suite-status.json")
        or data["policy_sha256"] != state["frozen_policy_sha256"]
        or _sha(_file(root, data["script"])) != data["script_sha256"]
        or data["planned_at"] <= state["ended"]
        or not data["entries"]
    ):
        raise ValueError("Dispatch supplement identity or lifecycle differs")
    fields = (
        "job",
        "case",
        "backend",
        "seed",
        "selected_config",
        "original_metrics",
        "original_metrics_sha256",
        "original_metadata_sha256",
    )
    if data["plan"] != [{k: x[k] for k in fields} for x in data["entries"]]:
        raise ValueError("Dispatch observations differ from their declared plan")
    entries = {}
    for item in data["entries"]:
        key = (item["job"], item["case"]["id"], item["backend"])
        if key in entries:
            raise ValueError("Duplicate dispatch supplement entry")
        entries[key] = item
    return dict(
        root=root,
        entries=entries,
        used=set(),
        manifest=str(path),
        manifest_sha256=_sha(path),
        note="Separate untimed dispatch observations; original rows and timings are unchanged.",
    )


def verify_dispatch_observation(supplement, file, metadata, job, case, row):
    key = (job["label"], case["id"], row["backend"])
    if supplement is None or key not in supplement["entries"]:
        raise ValueError("Actual dispatched kernels were not recorded")
    item = supplement["entries"][key]
    if (
        row["kernel_names"] != "[]"
        or case["op"] != "attention"
        or not row["backend"].startswith("triton_")
        or item["case"] != case
        or item["seed"] != metadata["config"]["seed"]
        or item["selected_config"] != json.loads(row["selected_config"])
        or item["environment"] != metadata["environment"]
        or item["original_metrics"] != str(file.relative_to(file.parents[3]))
        or item["original_metrics_sha256"] != _sha(file)
        or item["original_metadata_sha256"] != _sha(file.parent / "metadata.json")
        or not item["accuracy"]["passed"]
        or not 1 <= len(item["attempts"]) <= 3
    ):
        raise ValueError("Dispatch supplement does not identify the unchanged original callable")
    selected = None
    for attempt in item["attempts"]:
        trace = _file(supplement["root"], attempt["trace"])
        if _sha(trace) != attempt["trace_sha256"] or not attempt["accuracy"]["passed"]:
            raise ValueError("Dispatch trace identity or correctness differs")
        events = json.loads(trace.read_text())["traceEvents"]
        kernels = [e for e in events if e.get("cat") == "kernel"]
        if any(
            e.get("ph") != "X" or e.get("dur", 0) <= 0 or e.get("args", {}).get("device") != 0
            for e in kernels
        ):
            raise ValueError("Dispatch trace does not contain valid GPU kernel events")
        names = sorted({e["name"] for e in kernels})
        if names != attempt["kernel_names"]:
            raise ValueError("Recorded kernel names differ from the raw GPU trace")
        if selected is not None:
            raise ValueError("A successful observation was replaced by a later attempt")
        if names:
            if not any("_attention" in n for n in names):
                raise ValueError("Expected Triton attention dispatch was not observed")
            selected = names
    if selected is None:
        raise ValueError("Supplemental profiling still contains no dispatched GPU kernel")
    supplement["used"].add(key)
    return selected
