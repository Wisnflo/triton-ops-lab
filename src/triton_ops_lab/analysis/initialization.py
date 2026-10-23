"""Validate vendor report rules without declaring the original application safe."""

import hashlib
import ast
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from ..resource_audit import safety_result
from .safety import VENDOR_WINOGRAD, vendor_race_attribution


SHARED_READ = "Uninitialized __shared__ memory read of size 16 bytes"
FLASH_FUNCTION = "flash_fwd_kernel"


def _fields(element, expected):
    if element is None or element.attrib or [x.tag for x in element] != expected:
        raise ValueError("Unexpected initialization rule fields")


def _signature(record, *, raw=False, broader=False):
    if not raw:
        _fields(record, ["kind", "what", "where"])
        _fields(record.find("what"), ["text", "size"])
        _fields(record.find("where"), ["func"] if broader else ["func", "pc"])
    if (
        record.findtext("kind") != "Initcheck"
        or record.findtext("what/text") != SHARED_READ
        or record.findtext("what/size") != "16"
        or record.findtext("where/func") != FLASH_FUNCTION
    ):
        raise ValueError("Rule covers an unverified report type or function")
    if broader:
        return ("Initcheck", SHARED_READ, 16, FLASH_FUNCTION)
    pc = record.findtext("where/pc", "")
    if not re.fullmatch(r"0x[0-9a-fA-F]+", pc):
        raise ValueError("Rule lacks a concrete reproduced instruction address")
    if raw:
        frames = record.findall("hostStack/frame")
        if not any(
            "pytorch_flash::run_flash_fwd<" in frame.findtext("func", "")
            and frame.findtext("module", "").endswith("/torch/lib/libtorch_cuda.so")
            for frame in frames
        ):
            raise ValueError("Reproduction does not identify the PyTorch launch library")
    return ("Initcheck", SHARED_READ, 16, FLASH_FUNCTION, int(pc, 16))


def audit_report_rules(raw_xml, rule_xml, broader_xml):
    """Require literal rules derived entirely from retained framework-only reports.

    The broader rule is used only for activation/negative controls. This function
    certifies rule provenance, not execution success or absence of vendor errors.
    """
    documents = []
    for path in (raw_xml, rule_xml, broader_xml):
        root = ET.fromstring(Path(path).read_bytes())
        if root.tag != "ComputeSanitizerOutput" or root.attrib:
            raise ValueError("Unexpected initialization XML root")
        if not len(root) or any(x.tag != "record" for x in root):
            raise ValueError("Missing or unexpected initialization XML records")
        documents.append(list(root))
    raw, rules, broader = documents
    observed = {_signature(r, raw=True) for r in raw}
    selected = [_signature(r) for r in rules]
    if set(selected) != observed:
        raise ValueError("Rules differ from reproduced instruction locations")
    if len(broader) != 1 or _signature(broader[0], broader=True) != next(iter(observed))[:-1]:
        raise ValueError("The negative-control rule is not the vendor-only broader rule")
    return dict(
        raw_reports=len(raw),
        rule_records=len(selected),
        unique_rules=len(set(selected)),
        function=FLASH_FUNCTION,
        address_space="shared",
        access_bytes=16,
        instruction_offsets=sorted(x[-1] for x in observed),
        sha256={
            label: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for label, path in zip(
                ("raw", "rules", "broader_control"),
                (raw_xml, rule_xml, broader_xml),
                strict=True,
            )
        },
        note="Rule provenance only. Original application and vendor failures remain separate evidence.",
    )


def audit_attributed_safety(root, manifest, policy, project):
    """Audit project checks with disclosed, reproduced vendor report exceptions.

    All kernels remain instrumented by initcheck. The original application run
    remains failed; this does not certify the excluded vendor accesses as safe.
    """
    root, project = Path(root), Path(project)

    def read(path):
        return json.loads(Path(path).read_text())

    raw = root / manifest["original_initcheck"]
    vendor = root / manifest["vendor_reproduction"]
    attributed = root / manifest["attributed"]
    controls = root / manifest["controls"]
    folders = dict(
        original_initcheck=raw, vendor_reproduction=vendor, attributed=attributed, controls=controls
    )
    metadata = {k: read(p / "metadata.json") for k, p in folders.items()}
    current = metadata["attributed"]
    for label, meta in metadata.items():
        if meta["policy_sha256"] != policy.sha256:
            raise ValueError("Safety evidence used a different frozen policy")
        script = root / manifest[label + "_script"]
        if hashlib.sha256(script.read_bytes()).hexdigest() != meta["source_script_sha256"]:
            raise ValueError("Safety execution source identity differs: " + label)
        # Machine identities are retained separately. Safety may be supplemented
        # on another L4 with otherwise identical code, libraries and driver.
        if {k: v for k, v in meta["environment"].items() if k != "runtime_identity"} != {
            k: v for k, v in current["environment"].items() if k != "runtime_identity"
        }:
            raise ValueError("Safety source, hardware or software differs across sessions")
    if not all(metadata[k]["complete"] for k in ("original_initcheck", "attributed", "controls")):
        raise ValueError("Safety execution has not completed")
    package = current["tool_package"]
    if (
        package["sha256"] != "bf5303fe49d8ab871876213a7a2cf9ebe844406a293dbd05cd327f52c976a726"
        or package["executable_sha256"]
        != "fc3b9d2abad07999a56631a6be69c320461ebdb80a5ab0044775c44d301c9488"
        or package["url"]
        != "https://developer.download.nvidia.com/compute/cuda/redist/cuda_sanitizer_api/linux-x86_64/cuda_sanitizer_api-linux-x86_64-13.4.92-archive.tar.xz"
        or metadata["controls"]["tool_package"] != package
    ):
        raise ValueError("Sanitizer executable/package differs from the pinned tool")

    def run(folder, label):
        command = read(folder / (label + ".command.json"))
        log = (folder / (label + ".log")).read_text()
        if command["command"][0] != package["executable"] or any(
            x in log for x in ("Internal Sanitizer Error", "Exception ignored in atexit callback")
        ):
            raise ValueError("Safety tool did not run normally: " + label)
        args = command["command"]
        if "--target-processes" not in args or args[args.index("--target-processes") + 1] != "all":
            raise ValueError("Safety command does not include child processes")
        if "--suppressions" in args and label not in {
            "compact-literal",
            "broader-vendor-positive",
            "suppression-negative",
            "shared-with-broader-vendor-rule",
            "initcheck-attributed",
        }:
            raise ValueError("Unexpected report exception: " + label)
        if any(x.startswith(("--kernel-name", "--launch-skip", "--launch-count")) for x in args):
            if label != "racecheck-project":
                raise ValueError("Safety evidence omits kernel instrumentation: " + label)
        return command, log

    def clean(folder, label, tool, record):
        cmd, log = run(folder, label)
        actual = safety_result(tool, cmd["exit_code"], log, 96)
        if actual != record or actual["status"] != "passed":
            raise ValueError("Full safety execution is not clean: " + label)
        if cmd["command"][cmd["command"].index("--tool") + 1] != tool:
            raise ValueError("Wrong safety tool: " + label)
        return cmd["command"]

    original_cmd, original_log = run(raw, "initcheck")
    original = safety_result("initcheck", original_cmd["exit_code"], original_log, 96)
    for flag, value in (("--tool", "initcheck"), ("--initcheck-address-space", "all")):
        if (
            flag not in original_cmd["command"]
            or original_cmd["command"][original_cmd["command"].index(flag) + 1] != value
        ):
            raise ValueError("Original initialization command did not check the full scope")
    if (
        original != metadata["original_initcheck"]["results"]["initcheck"]
        or original["status"] != "failed"
        or original["pytest_passed"] != 96
        or not any(x > 0 for x in original["error_summaries"])
        or "--suppressions" in original_cmd["command"]
    ):
        raise ValueError("Original unfiltered initialization failure was not retained")
    raw_xml = vendor / "flash-only-all.xml"
    cmd, log = run(vendor, "flash-only-all")
    if (
        cmd["command"][cmd["command"].index("--tool") + 1] != "initcheck"
        or cmd["command"][cmd["command"].index("--initcheck-address-space") + 1] != "all"
    ):
        raise ValueError("Framework reproduction used the wrong tool or address space")
    raw_records = ET.parse(raw_xml).findall("record")
    if (
        cmd["exit_code"] != 1
        or log.count("FLASH_ONLY_COMPLETE") != 4
        or re.findall(r"ERROR SUMMARY:\s*(\d+) errors?", log) != [str(len(raw_records))]
    ):
        raise ValueError("Framework reproduction omits reports or numerical checks")
    archive = root / manifest["input_archive"]
    if (
        hashlib.sha256(archive.read_bytes()).hexdigest()
        != current["reused_input"]["archive_sha256"]
        or hashlib.sha256(raw_xml.read_bytes()).hexdigest() != current["reused_input"]["xml_sha256"]
    ):
        raise ValueError("Reused raw reproduction identity differs")
    program = root / manifest["framework_repro_source"]
    imports = [
        n
        for n in ast.walk(ast.parse(program.read_text()))
        if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    if not imports or any(
        (
            isinstance(n, ast.Import)
            and any(a.name != "torch" and not a.name.startswith("torch.") for a in n.names)
        )
        or (isinstance(n, ast.ImportFrom) and (not n.module or not n.module.startswith("torch.")))
        for n in imports
    ):
        raise ValueError("Reproduction imports non-framework code")
    repro_meta = read(program.parent / "metadata.json")
    if (
        hashlib.sha256(program.read_bytes()).hexdigest()
        != repro_meta["results"]["flash-only"]["source_sha256"]
    ):
        raise ValueError("Framework-only reproduction source identity differs")
    rules = audit_report_rules(
        raw_xml, attributed / "vendor-reports-literal.xml", controls / "vendor-function-only.xml"
    )

    def suppression(args, file):
        if args[args.index("--tool") + 1] != "initcheck":
            raise ValueError("Wrong tool in initialization report control")
        if (
            args.count("--suppressions") != 1
            or Path(args[args.index("--suppressions") + 1]).name != file
        ):
            raise ValueError("Wrong report rules in initialization command")
        if args[args.index("--initcheck-address-space") + 1] != "all":
            raise ValueError("Incomplete initialization address-space coverage")

    for folder, label, rule in (
        (attributed, "compact-literal", "vendor-reports-literal.xml"),
        (controls, "broader-vendor-positive", "vendor-function-only.xml"),
    ):
        command, log = run(folder, label)
        suppression(command["command"], rule)
        if (
            command["exit_code"]
            or log.count("FLASH_ONLY_COMPLETE") != 4
            or re.findall(r"ERROR SUMMARY:\s*(\d+) errors?", log) != ["0"]
        ):
            raise ValueError("Vendor report rules did not activate")
    command, log = run(attributed, "suppression-negative")
    suppression(command["command"], "vendor-reports-literal.xml")
    if (
        command["exit_code"] != 1
        or "SUPPRESSION_NEGATIVE_COMPLETE" not in log
        or "at _suppression_negative+" not in log
        or "Uninitialized __global__ memory read" not in log
    ):
        raise ValueError("Global-memory fault control was not detected")
    names = set()
    for file in (project / "src/triton_ops_lab/kernels").glob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, ast.FunctionDef) and any(
                isinstance(d, ast.Attribute)
                and d.attr == "jit"
                and isinstance(d.value, ast.Name)
                and d.value.id == "triton"
                for d in node.decorator_list
            ):
                names.add(node.name)
    if set(metadata["controls"]["project_kernel_names"]) != names or any(
        FLASH_FUNCTION in n for n in names
    ):
        raise ValueError("Fault controls do not cover every actual project kernel name")
    for label in ("shared-memcheck", "shared-unfiltered", "shared-with-broader-vendor-rule"):
        command, log = run(controls, label)
        if "SHARED_CONTROL_COMPLETE 6" not in log:
            raise ValueError("Shared-memory fault controls did not complete")
        if label == "shared-memcheck":
            if command["command"][command["command"].index("--tool") + 1] != "memcheck":
                raise ValueError("Wrong memory-bounds tool in shared fault control")
            if command["exit_code"] or re.findall(r"ERROR SUMMARY:\s*(\d+) errors?", log) != ["0"]:
                raise ValueError("Shared fault control is not within valid memory bounds")
        elif (
            command["exit_code"] != 1
            or not names <= set(re.findall(r"at (_[A-Za-z_]+)\+0x", log))
            or SHARED_READ not in log
        ):
            raise ValueError("Same-type shared fault was hidden by report rules")
        if label == "shared-with-broader-vendor-rule":
            suppression(command["command"], "vendor-function-only.xml")
        elif "--suppressions" in command["command"]:
            raise ValueError("Unfiltered fault control contains report exceptions")
    cmd = clean(
        attributed, "initcheck-attributed", "initcheck", current["results"]["initcheck-attributed"]
    )
    suppression(cmd, "vendor-reports-literal.xml")
    safety = metadata["controls"]["results"]
    for tool in ("memcheck", "synccheck"):
        cmd = clean(controls, tool, tool, safety[tool])
        if "--suppressions" in cmd:
            raise ValueError("Unexpected report suppression outside initcheck")
    race_scope = dict(global_racecheck_status="passed")
    if safety["racecheck"]["status"] == "passed":
        clean(controls, "racecheck", "racecheck", safety["racecheck"])
    else:
        cmd, log = run(controls, "racecheck")
        repro_cmd, repro_log = run(controls, "vendor-race-only")
        if (
            cmd["exit_code"] != 1
            or repro_cmd["exit_code"] != 1
            or safety_result("racecheck", cmd["exit_code"], log, 96) != safety["racecheck"]
        ):
            raise ValueError("Race attribution did not retain the unfiltered failure")
        race_scope = vendor_race_attribution(log, repro_log)
        tree = ast.parse((controls / "vendor-race-only.py").read_text())
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        if not imports or any(
            not isinstance(n, ast.Import)
            or any(a.name not in {"torch", "torch.nn.functional"} for a in n.names)
            for n in imports
        ):
            raise ValueError("Race reproduction imports project or non-framework code")
        cmd = clean(controls, "racecheck-project", "racecheck", safety["racecheck-project"])
        filters = [(a, cmd[i + 1]) for i, a in enumerate(cmd[:-1]) if a.startswith("--kernel-name")]
        if filters != [("--kernel-name-exclude", "kne=" + VENDOR_WINOGRAD)]:
            raise ValueError("Race filter excludes an unverified kernel")
    race_note = race_scope.pop("note", None)
    return dict(
        **race_scope,
        race_attribution_note=race_note,
        global_initcheck_status="failed",
        project_initcheck_status="passed_with_disclosed_vendor_report_rules",
        original_initcheck=original,
        rules=rules,
        project_kernel_names=sorted(names),
        qualification_environments={k: v["environment"] for k, v in metadata.items()},
        vendor_shared_initialization_certified=False,
        note="All project kernels remain instrumented; global and same-type shared fault controls remain detectable. Vendor failures and unprinted original events are not certified or declared false positives.",
    )
