"""Opt-in compiled IR capture. Never enabled during performance timing."""

import os
import uuid
import hashlib

from ..artifacts import output_root, write_json


def capture(compiled, label):
    directory = os.environ.get("TRITON_LAB_CAPTURE_DIR")
    if not directory:
        return
    root = output_root(directory)
    folder = root / (label + "-" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    files = {}
    for kind, content in compiled.asm.items():
        target = folder / ("kernel." + kind)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(str(content))
        files[kind] = {
            "file": target.name,
            "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    metadata = getattr(compiled, "metadata", None)
    fields = {}
    for name in ("name", "shared", "num_warps", "num_stages", "num_ctas", "cluster_dims"):
        value = getattr(metadata, name, None)
        if value is None or isinstance(value, (str, int, float, bool, tuple, list)):
            fields[name] = value
    write_json(
        folder / "resources.json",
        {
            "registers": getattr(compiled, "n_regs", None),
            "spills": getattr(compiled, "n_spills", None),
            "metadata": str(metadata),
            "compiler_metadata": fields,
            "asm_files": files,
            "note": "Capture only; resource properties depend on locked Triton version",
        },
    )
