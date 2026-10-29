"""Build a reviewed source snapshot without private evidence or local Git history."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT_FILES = {
    ".gitignore",
    ".python-version",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "LICENSE",
    "NOTICE",
}
SOURCE_DIRS = {"src", "scripts", "tests", "configs", "docs", "notebooks", "tools", ".github"}
PRIVATE_SUFFIXES = {
    ".zip",
    ".bundle",
    ".pem",
    ".key",
    ".csv",
    ".tsv",
    ".log",
    ".har",
    ".prof",
    ".cubin",
    ".ptx",
    ".sass",
    ".ncu-rep",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
    ".pickle",
    ".sqlite",
    ".db",
}
PATTERNS = {
    "credential": re.compile(
        r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        r"sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}|"
        r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"
    ),
    "personal_path": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
    "private_notebook": re.compile(r"colab\.research\.google\.com/drive/[A-Za-z0-9_-]+"),
    "email": re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "url_credentials": re.compile(r"https?://[^\s/]+:[^\s/]+@"),
    "account_usage": re.compile(r"\b\d+(?:\.\d+)?\s*CU\b"),
}


def validate_file(name, raw):
    path = Path(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not (name in ROOT_FILES or path.parts[0] in SOURCE_DIRS)
        or path.suffix.lower() in PRIVATE_SUFFIXES
        or any(p in {"local-data", ".git", ".venv", "__pycache__"} for p in path.parts)
        or path.name.startswith(".env")
    ):
        raise ValueError(f"Private or unapproved export path: {name}")
    text = raw.decode("utf-8")
    for label, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            if label == "email" and match.group().endswith("@users.noreply.github.com"):
                continue
            raise ValueError(f"Potential {label} in {name}; values intentionally omitted")
    if path.suffix == ".ipynb":
        notebook = json.loads(text)
        if any(c.get("outputs") or c.get("execution_count") is not None for c in notebook["cells"]):
            raise ValueError(f"Executed notebook cannot be exported: {name}")


def sanitize_document(name, text):
    reference = (
        "docs/project.md#7-结果管理与审计" if name == "README.md" else "project.md#7-结果管理与审计"
    )

    def link(match):
        label, target = match.groups()
        if re.search(r"colab\.research\.google\.com/drive/", target):
            return label
        if target.startswith("/Users/") or "../storage/" in target:
            return f"[{label}]({reference})"
        return match.group()

    text = re.sub(r"\[([^\]]+)\]\(([^)\n]+)\)", link, text)
    text = re.sub(r"/(?:Users|home)/[A-Za-z0-9_.-]+/[^\s`\n)]+", "<local-project-path>", text)
    text = re.sub(
        r"https://colab\.research\.google\.com/drive/[A-Za-z0-9_-]+", "<private-notebook>", text
    )
    return "\n\n".join(
        paragraph
        for paragraph in text.split("\n\n")
        if not re.search(r"\bCU\b|Gemini|余额|订阅", paragraph)
    )


def export(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    private = root / "local-data" / "publish"
    if not output.is_relative_to(private) or output == private or output.exists():
        raise ValueError("Use a new directory below local-data/publish")
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    rendered, changes = {}, []
    for name in filter(None, files):
        source = root / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Invalid tracked export file: {name}")
        raw = source.read_bytes()
        exported = sanitize_document(name, raw.decode()).encode() if source.suffix == ".md" else raw
        validate_file(name, exported)
        rendered[name] = exported
        if raw != exported:
            changes.append(name)
    # Nothing is written until every candidate has passed the privacy gate.
    repo = output / "repository"
    for name, raw in rendered.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    report = dict(
        schema="source-publication-review-v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        file_count=len(rendered),
        bytes=sum(map(len, rendered.values())),
        redacted_documents=changes,
        original_git_history_included=False,
        raw_evidence_included=False,
        privacy_checks_passed=True,
        files=[
            dict(path=n, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            for n, raw in sorted(rendered.items())
        ],
    )
    (output / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    return repo, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo, report = export(Path(__file__).resolve().parents[1], args.output)
    print(
        json.dumps(
            dict(
                repository=str(repo),
                files=report["file_count"],
                bytes=report["bytes"],
                privacy_checks_passed=True,
            ),
            indent=2,
        )
    )
