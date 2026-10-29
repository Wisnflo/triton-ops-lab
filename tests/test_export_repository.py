import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


spec = importlib.util.spec_from_file_location(
    "export_repository", Path(__file__).resolve().parents[1] / "tools/export_repository.py"
)
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


@pytest.mark.parametrize(
    "name,raw",
    [
        ("README.md", b"github" + b"_pat_" + b"x" * 40),
        ("README.md", b"person" + b"@" + b"example.test"),
        ("README.md", b"/Users/" + b"someone" + b"/private"),
        ("docs/data.zip", b"PK"),
        ("local-data/manifest.json", b"{}"),
        (
            "notebooks/run.ipynb",
            json.dumps({"cells": [{"outputs": [{"text": "private trace"}]}]}).encode(),
        ),
    ],
)
def test_private_material_is_rejected(name, raw):
    with pytest.raises(ValueError):
        publisher.validate_file(name, raw)


def test_redaction_keeps_artifact_links_explicitly_private():
    home = "/Users/" + "someone" + "/workspace/results.json"
    notebook = "https://colab.research.google.com/" + "drive/" + "private-id"
    usage = "12.34" + " C" + "U"
    source = f"# History\n\n[raw]({home})\n\n[notebook]({notebook})\n\nBalance: {usage}\n"
    result = publisher.sanitize_document("docs/history.md", source)
    publisher.validate_file("docs/history.md", result.encode())
    assert "[raw](project.md#7-结果管理与审计)" in result
    assert "12.34" not in result and "private-id" not in result


def test_privacy_failure_writes_no_partial_snapshot(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "README.md").write_text("# Example\n\nSafe text.\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "data.zip").write_bytes(b"PK")
    subprocess.run(["git", "add", "README.md", "docs/data.zip"], cwd=tmp_path, check=True)
    output = tmp_path / "local-data/publish/rejected"
    with pytest.raises(ValueError):
        publisher.export(tmp_path, output)
    assert not output.exists()
