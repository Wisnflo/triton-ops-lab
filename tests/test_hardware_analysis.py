import csv
import json

import pytest

from triton_ops_lab.analysis.hardware import audit_hardware, expected_grid


def test_launch_geometry_respects_operator_axes_and_tails():
    tile = dict(bm=64, bn=64, warps=4)
    assert expected_grid(dict(op="matmul", shape=[129, 193, 257]), tile) == (12, 1, 1)
    assert expected_grid(dict(op="attention", shape=[1, 2, 2048, 128]), tile) == (32, 2, 1)
    assert expected_grid(
        dict(op="conv2d", shape=[1, 16, 33, 31, 65, 3], stride=2, padding=1), tile
    ) == (5, 2, 1)


@pytest.mark.parametrize(
    "compute_metric",
    [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__instruction_throughput.avg.pct_of_peak_sustained_elapsed",
    ],
)
def test_hardware_audit_requires_counters_and_actual_launch_match(tmp_path, compute_metric):
    tile = dict(bm=64, bn=64, warps=4)
    rows = []
    manifest = []
    fields = [
        "ID",
        "Process ID",
        "Kernel Name",
        "Grid Size",
        "Block Size",
        "launch__registers_per_thread",
        "launch__shared_mem_per_block_dynamic",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        compute_metric,
        "dram__bytes_read.sum",
        "smsp__inst_executed.sum",
    ]
    units = dict.fromkeys(fields, "")
    units["launch__shared_mem_per_block_dynamic"] = "Kbyte/block"
    for i, (op, shape, name) in enumerate(
        [
            ("matmul", [129, 193, 257], "_gemm"),
            ("attention", [1, 2, 2048, 128], "_attention"),
            ("conv2d", [1, 16, 32, 32, 32, 3], "_conv"),
        ]
    ):
        case = dict(id=op, op=op, shape=shape, dtype="float16")
        for label in ("fixed", "rule"):
            manifest.append(dict(case=case, tile=tile, configuration=label))
            rows.append(
                dict(
                    zip(
                        fields,
                        [
                            str(len(rows)),
                            "1",
                            name,
                            str(expected_grid(case, tile)),
                            str((128, 1, 1)),
                            "60",
                            "6.144",
                            "40",
                            "33",
                            "1000",
                            "2000",
                        ],
                        strict=True,
                    )
                )
            )
            folder = tmp_path / "capture" / (op + "-" + label + "-triton_fixed") / "kernel"
            folder.mkdir(parents=True)
            (folder / "resources.json").write_text(
                json.dumps(dict(registers=60, spills=0, compiler_metadata=dict(shared=6144)))
            )
    (tmp_path / "profile-worker").mkdir()
    (tmp_path / "profile-worker/worker-results.json").write_text(
        json.dumps(dict(complete=True, results=manifest))
    )

    def save():
        with (tmp_path / "nsight-metrics.log").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows([units, *rows])

    save()
    audited = audit_hardware(tmp_path)["launches"]
    assert len(audited) == 6
    assert audited[0]["compute_counters"] == {compute_metric: dict(value=33, unit="")}
    rows[-1]["Grid Size"] = "(999,1,1)"
    save()
    with pytest.raises(ValueError, match="geometry"):
        audit_hardware(tmp_path)
    rows[-1]["Grid Size"] = str(expected_grid(manifest[-1]["case"], tile))
    rows[-1]["smsp__inst_executed.sum"] = "n/a"
    save()
    with pytest.raises(ValueError, match="instruction"):
        audit_hardware(tmp_path)
