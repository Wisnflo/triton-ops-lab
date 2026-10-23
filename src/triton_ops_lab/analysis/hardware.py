"""Cross-check instrumented launch counters against explicit compiler captures."""

import ast
import csv
import json
import math
from pathlib import Path


def expected_grid(case, tile):
    def ceil(x, b):
        return (x + b - 1) // b

    shape = case["shape"]
    if case["op"] == "matmul":
        return (ceil(shape[0], tile["bm"]) * ceil(shape[1], tile["bn"]), 1, 1)
    if case["op"] == "attention":
        return (ceil(shape[2], tile["bm"]), shape[0] * shape[1], 1)
    b, _, h, w, oc, k = shape
    stride, pad = case.get("stride", 1), case.get("padding", 0)
    oh, ow = (h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1
    return (ceil(b * oh * ow, tile["bm"]), ceil(oc, tile["bn"]), 1)


def audit_hardware(path):
    path = Path(path)
    manifest = json.loads((path / "profile-worker/worker-results.json").read_text())
    if not manifest["complete"] or len(manifest["results"]) != 6:
        raise ValueError("Expected fixed/rule profiles for three operators")
    text = (path / "nsight-metrics.log").read_text()
    lines = text.splitlines()
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if next(csv.reader([line]), [])[:2] == ["ID", "Process ID"]
        ),
        None,
    )
    if start is None:
        raise ValueError("Missing raw Nsight CSV header")
    raw = list(csv.DictReader(lines[start:]))
    if len(raw) != 7 or raw[0]["ID"] != "":
        raise ValueError("Expected one units row and exactly six launches")
    units = raw[0]
    result = []
    for item, row in zip(manifest["results"], raw[1:], strict=True):
        case, tile = item["case"], item["tile"]
        expected_name = {"matmul": "_gemm", "attention": "_attention", "conv2d": "_conv"}[
            case["op"]
        ]
        if row["Kernel Name"] != expected_name:
            raise ValueError("Unexpected profiled kernel order or dispatch")
        grid = tuple(ast.literal_eval(row["Grid Size"]))
        block = tuple(ast.literal_eval(row["Block Size"]))
        if grid != expected_grid(case, tile) or block != (tile["warps"] * 32, 1, 1):
            raise ValueError("Profiled launch geometry does not match selected configuration")
        directory = path / "capture" / (case["id"] + "-" + item["configuration"] + "-triton_fixed")
        files = list(directory.glob("*/resources.json"))
        if len(files) != 1:
            raise ValueError("Matching compiler resource capture missing")
        resource = json.loads(files[0].read_text())

        def number(key):
            value = float(row[key].replace(",", ""))
            if not math.isfinite(value):
                raise ValueError("Non-finite hardware counter")
            return value

        registers = number("launch__registers_per_thread")
        shared = number("launch__shared_mem_per_block_dynamic")
        unit = units["launch__shared_mem_per_block_dynamic"]
        scale = {"byte/block": 1, "Kbyte/block": 1000}.get(unit)
        if scale is None:
            raise ValueError("Unknown shared memory counter unit: " + unit)
        if registers != resource["registers"] or not math.isclose(
            shared * scale, resource["compiler_metadata"]["shared"], abs_tol=0.01
        ):
            raise ValueError("Profiled registers/shared memory differ from compiler capture")
        occupancy = number("sm__warps_active.avg.pct_of_peak_sustained_active")
        compute = {
            key: dict(value=number(key), unit=units[key])
            for key in (
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "sm__instruction_throughput.avg.pct_of_peak_sustained_elapsed",
            )
            if key in row
        }
        if not compute:
            raise ValueError("Missing named compute activity counter")
        memory = {}
        instructions = {}
        for key, value in row.items():
            try:
                parsed = float(value.replace(",", ""))
            except (ValueError, AttributeError):
                continue
            if not math.isfinite(parsed):
                continue
            counter = dict(value=parsed, unit=units[key])
            if key.startswith(("dram__", "lts__", "l1tex__")):
                memory[key] = counter
            if "inst_executed" in key and key.startswith(("sm__", "smsp__")):
                instructions[key] = counter
        if not memory or not instructions or not any(x["value"] > 0 for x in instructions.values()):
            raise ValueError("Missing memory or instruction counter evidence")
        result.append(
            dict(
                case=case,
                configuration=item["configuration"],
                tile=tile,
                kernel=row["Kernel Name"],
                grid=grid,
                block=block,
                registers=registers,
                dynamic_shared_bytes=shared * scale,
                compiler_spills=resource["spills"],
                achieved_occupancy_percent=occupancy,
                compute_counters=compute,
                memory_counters=memory,
                instruction_counters=instructions,
            )
        )
    return dict(
        complete=True,
        launches=result,
        note="Instrumented metrics explain resources and access behavior; they are not steady-state benchmark times or a causal proof of a bottleneck.",
    )
