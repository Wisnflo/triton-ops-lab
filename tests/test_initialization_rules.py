import xml.etree.ElementTree as ET

import pytest

from triton_ops_lab.analysis.initialization import SHARED_READ, audit_report_rules


def files(tmp_path):
    raw = ET.fromstring(
        "<ComputeSanitizerOutput><record><kind>Initcheck</kind>"
        f"<what><text>{SHARED_READ}</text><address>0x10</address><size>16</size></what>"
        "<where><func>flash_fwd_kernel</func><pc>0x1990</pc></where>"
        "<hostStack><frame><func>pytorch_flash::run_flash_fwd&lt;32&gt;</func>"
        "<module>/env/torch/lib/libtorch_cuda.so</module></frame></hostStack>"
        "</record></ComputeSanitizerOutput>"
    )
    rule = ET.fromstring(ET.tostring(raw))
    rule[0].remove(rule[0].find("hostStack"))
    rule[0].find("what").remove(rule[0].find("what/address"))
    broader = ET.fromstring(ET.tostring(rule))
    broader[0].find("where").remove(broader[0].find("where/pc"))
    paths = [tmp_path / (name + ".xml") for name in ("raw", "rule", "broader")]
    for tree, path in zip((raw, rule, broader), paths, strict=True):
        ET.ElementTree(tree).write(path)
    return paths


def test_rules_preserve_concrete_vendor_instruction_provenance(tmp_path):
    paths = files(tmp_path)
    result = audit_report_rules(*paths)
    assert result["raw_reports"] == result["unique_rules"] == 1
    assert result["instruction_offsets"] == [0x1990]
    # ElementTree exports can retain an identical record with different tail whitespace.
    tree = ET.parse(paths[1])
    tree.getroot().append(ET.fromstring(ET.tostring(tree.getroot()[0])))
    tree.write(paths[1])
    duplicate = audit_report_rules(*paths)
    assert duplicate["rule_records"] == 2 and duplicate["unique_rules"] == 1


@pytest.mark.parametrize("replacement", [".*", "_gemm", "^flash_fwd_kernel$"])
def test_rules_reject_wildcards_and_unreproduced_project_functions(tmp_path, replacement):
    paths = files(tmp_path)
    paths[1].write_text(paths[1].read_text().replace("flash_fwd_kernel", replacement))
    with pytest.raises(ValueError, match="function"):
        audit_report_rules(*paths)


def test_rules_reject_unseen_locations_and_global_reads(tmp_path):
    paths = files(tmp_path)
    original = paths[1].read_text()
    paths[1].write_text(original.replace("0x1990", "0x2000"))
    with pytest.raises(ValueError, match="locations"):
        audit_report_rules(*paths)
    paths[1].write_text(original.replace("__shared__", "__global__"))
    with pytest.raises(ValueError, match="report type"):
        audit_report_rules(*paths)
