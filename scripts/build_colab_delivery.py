"""Create a private, self-contained notebook from a clean Git commit.

Run with uv run --frozen python scripts/build_colab_delivery.py.
Only tracked committed files travel; the generated notebook remains in storage.
"""

import argparse
import base64
import hashlib
import subprocess
import textwrap

from triton_ops_lab.artifacts import new_run, output_root, project_root, write_json


def build(root, suite="baseline-audit"):
    project = project_root()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=project).strip():
        raise RuntimeError("Commit/review changes before delivery; a clean checkout is required")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()
    path = new_run(root, "colab-" + suite)
    bundle = path / "source.bundle"
    subprocess.run(["git", "bundle", "create", str(bundle), "HEAD"], cwd=project, check=True)
    data = bundle.read_bytes()
    payload = "\n".join(textwrap.wrap(base64.b64encode(data).decode(), 100))
    cells = []

    def cell(kind, source, **metadata):
        c = dict(
            cell_type=kind,
            id=f"delivery-{len(cells)}",
            metadata=metadata,
            source=source.splitlines(True),
        )
        if kind == "code":
            compile(source, "colab-cell", "exec")
            c.update(execution_count=None, outputs=[])
        cells.append(c)

    cell(
        "markdown",
        f"# Triton {suite} · {commit[:7]}\n"
        "Select L4/A100. Frozen uv environment. Correctness gates precede timing. "
        "Event and CUDA Graph protocols must be reported separately. "
        "Three runs use rotated backend order on tuning shapes. "
        "Download evidence before deleting the runtime.\n",
    )
    setup = (
        f"""#@title Restore exact source and install locked environment
import base64, hashlib, json, os, subprocess, sys, time, shutil
from pathlib import Path
COMMIT = {commit!r}
SUITE = {suite!r}
PROJECT = Path('/content/triton-ops-lab')
OUTPUT = Path('/content/triton-'+SUITE+'-evidence')
OUTPUT.mkdir(exist_ok=True)
os.environ['TRITON_LAB_OUTPUT_ROOT'] = str(OUTPUT)
os.environ['MPLBACKEND'] = 'Agg'
if SUITE.startswith('study-'):
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
payload = base64.b64decode("""
        + "'''\n"
        + payload
        + "\n''')\n"
        + f"""
assert hashlib.sha256(payload).hexdigest() == {hashlib.sha256(data).hexdigest()!r}
bundle = Path('/content/triton-source.bundle')
bundle.write_bytes(payload)
if not PROJECT.exists():
    subprocess.run(['git','clone',str(bundle),str(PROJECT)],check=True)
    subprocess.run(['git','checkout','--detach',COMMIT],cwd=PROJECT,check=True)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()==COMMIT
assert not subprocess.check_output(['git','status','--porcelain'],cwd=PROJECT,text=True).strip()
subprocess.run([sys.executable,'-m','pip','install','uv==0.12.15'],check=True)
def run(label, args, allowed=(0,), stream=False):
    start=time.time()
    if stream:
        with (OUTPUT/(label+'.log')).open('w') as log:
            process=subprocess.Popen(args,cwd=PROJECT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:
                log.write(line); log.flush(); print(line.rstrip(),flush=True)
            code=process.wait()
        tail=(OUTPUT/(label+'.log')).read_text()[-3500:]
    else:
        result=subprocess.run(args,cwd=PROJECT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (OUTPUT/(label+'.log')).write_text(result.stdout)
        code,tail=result.returncode,result.stdout[-3500:]
    (OUTPUT/(label+'.status.json')).write_text(json.dumps(dict(returncode=code,seconds=time.time()-start,commit=COMMIT)))
    print(label, 'exit=',code, tail, flush=True)
    if code not in allowed: raise RuntimeError(label+' failed; retain evidence')
run('install',['uv','sync','--frozen'])
run('doctor',['uv','run','--frozen','triton-lab','doctor'])
print('SETUP_COMPLETE')
"""
    )
    cell("code", setup, cellView="form")
    cell(
        "code",
        """run('cuda-correctness',['uv','run','--frozen','triton-lab','check','--device','cuda'])
run('gpu-tests',['uv','run','--frozen','python','scripts/gpu_tools.py','check'])
print('GPU_CORRECTNESS_COMPLETE')
""",
    )
    cell(
        "code",
        """assert json.loads((OUTPUT/'cuda-correctness.status.json').read_text())['returncode']==0
assert json.loads((OUTPUT/'gpu-tests.status.json').read_text())['returncode']==0
for repetition in range(3):
    for mode in ('events','graphs'):
        cfg=json.loads((PROJECT/('configs/'+SUITE+'-'+mode+'.json')).read_text())
        backends=cfg['backends']
        shift=(repetition*len(backends))//3
        cfg['backends']=backends[shift:]+backends[:shift]
        label='audit-'+mode+'-'+str(repetition+1)
        cfg['name']=label
        config_path=OUTPUT/(label+'.json')
        config_path.write_text(json.dumps(cfg,indent=2))
        # Exit 1 means recorded failed rows; retain them and continue the audit.
        # Exit 2/unexpected setup errors stop the stage.
        run(label,['uv','run','--frozen','triton-lab','bench','--config',str(config_path)],allowed=(0,1))
print('BASELINE_AUDIT_COMPLETE')
if SUITE == 'gemm-tune':
    for case_id in ('fused-tail-float16','fused-large-float16'):
        run('host-profile-'+case_id,['uv','run','--frozen','python','scripts/gemm_host_profile.py','--case',case_id])
""",
    )
    if suite == "gpu-resource":
        source = """assert json.loads((OUTPUT/'cuda-correctness.status.json').read_text())['returncode']==0
assert json.loads((OUTPUT/'gpu-tests.status.json').read_text())['returncode']==0
run('resource-audit',['uv','run','--frozen','python','scripts/gpu_resource_audit.py','run'],allowed=(0,1))
print('RESOURCE_AUDIT_COMPLETE; inspect individual tool statuses before claiming validation')
"""
        compile(source, "resource-audit-cell", "exec")
        cells[-1]["source"] = source.splitlines(True)
        cells[0]["source"] = [
            f"# Triton GPU resources and memcheck · {commit[:7]}\n",
            "L4. Exact uv environment. Development shapes only; no performance measurement. "
            "Missing tools/permissions are not passes. Download evidence before runtime deletion.\n",
        ]
    if suite in {
        "study-development",
        "study-evaluation",
        "study-replication",
        "study-cost",
        "study-qualification",
    }:
        phase = suite.removeprefix("study-")
        source = """assert json.loads((OUTPUT/'cuda-correctness.status.json').read_text())['returncode']==0
assert json.loads((OUTPUT/'gpu-tests.status.json').read_text())['returncode']==0
"""
        if phase == "qualification":
            source += "run('qualification',['uv','run','--frozen','python','scripts/qualify_gpu.py','all'],allowed=(0,1),stream=True)\n"
        else:
            source += f"run({phase!r},['uv','run','--frozen','python','scripts/run_study.py','--suite','configs/study/{phase}-suite.json'],stream=True)\n"
        compile(source, "study-development-cell", "exec")
        cells[-1]["source"] = source.splitlines(True)
        cells[0]["source"] = [
            f"# Triton technical {phase} · {commit[:7]}\n",
            "L4/A100, locked source/environment. Separate timing protocols. "
            "Development learns configurations; later phases require a frozen policy and never update it. "
            "Live job progress and raw evidence; no paper/report generation.\n",
        ]
    cell(
        "code",
        """from google.colab import files
for metrics in sorted(OUTPUT.glob('*/metrics.csv')):
    run('analysis-'+metrics.parent.name,['uv','run','--frozen','triton-lab','analyze',str(metrics.parent)])
print((OUTPUT/'doctor.log').read_text())
for log in sorted(OUTPUT.glob('*/execution.log')):
    print(log, log.read_text()[-2000:])
if SUITE.startswith('study-'):
    import zipfile
    omitted=[]
    retained=[]
    for file in sorted(OUTPUT.rglob('*')):
        if not file.is_file(): continue
        rel=file.relative_to(OUTPUT)
        if {'triton-cache','inductor-cache','controlled-cache'} & set(rel.parts):
            omitted.append(dict(path=str(rel),bytes=file.stat().st_size))
        else:
            retained.append(file)
    inventory=OUTPUT/'archive-inventory.json'
    inventory.write_text(json.dumps(dict(
        omitted_cache_files=len(omitted),omitted_cache_bytes=sum(x['bytes'] for x in omitted),
        omitted=omitted,note='Only rebuildable compiler caches excluded. Explicit capture IR, cubin, SASS and profiling evidence retained.'
    ),indent=2))
    retained.append(inventory)
    archive='/content/triton-'+SUITE+'-evidence.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for file in retained: z.write(file,file.relative_to(OUTPUT))
else:
    archive=shutil.make_archive('/content/triton-'+SUITE+'-evidence','zip',OUTPUT)
print('Evidence bytes:',Path(archive).stat().st_size)
files.download(archive)
""",
    )
    notebook = path / f"triton-{suite}-{commit[:7]}.ipynb"
    write_json(
        notebook,
        dict(
            cells=cells,
            metadata={
                "kernelspec": {"display_name": "Python 3", "name": "python3"},
                "accelerator": "GPU",
            },
            nbformat=4,
            nbformat_minor=5,
        ),
    )
    write_json(
        path / "manifest.json",
        dict(
            commit=commit,
            bundle_sha256=hashlib.sha256(data).hexdigest(),
            notebook_sha256=hashlib.sha256(notebook.read_bytes()).hexdigest(),
            gpu_verified=False,
        ),
    )
    return notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root")
    parser.add_argument(
        "--suite",
        choices=[
            "baseline-audit",
            "mm-compat",
            "gemm-tune",
            "gpu-resource",
            "study-development",
            "study-evaluation",
            "study-replication",
            "study-cost",
            "study-qualification",
        ],
        default="baseline-audit",
    )
    args = parser.parse_args()
    print(build(output_root(args.output_root), args.suite))
