"""Generates SnatchGuard_Train.ipynb: the same one-command pipeline as the GPU box, on Google Colab (free tier).

Free Colab gives a T4 for a few hours at a time, so the run is split by what each part needs:
  1. GPU runtime: data, training, perception caches, learned scorer (`run_all --until learned`)
  2. any runtime (CPU is enough): rule tuning, fusion, report, so no GPU time is spent on CPU work
  3. GPU runtime, optional: the simulated-night check, then the report again
Everything a later session needs (done-markers, logs, checkpoints, models, caches, results) lives on Google Drive
under My Drive/snatchguard/state; only the videos and training images are rebuilt on a fresh runtime, and the
training images only while training is unfinished (training/run_all.py: LOCAL_OUTPUTS).
"""

import json
from pathlib import Path

MD, CODE = "markdown", "code"
cells = [
    (MD, """# SnatchGuard on Google Colab (free tier)

Same pipeline as `scripts/windows/train_all.ps1`: Snatch 1.0 → training sets → pseudo-labels → fine-tuning →
perception caches → learned scorer → cross-validated tuning and fusion → **acceptance report**
(recall/precision vs 90 %), with the quick profile (YOLO26n).

Put in `My Drive/snatchguard/`: `snatchguard_code.zip` and `snatch_web_dataset.zip` (or its `.part*` pieces).
Every result is written to `My Drive/snatchguard/state/`, so a disconnect loses at most the epoch in progress:
**after a disconnect, run the setup cell and then the part you were in again**; finished steps are skipped."""),
    (MD, "## Setup (every new runtime)"),
    (CODE, """import os, shutil, subprocess, zipfile
from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/snatchguard'
STATE, WEB = f'{DRIVE}/state', '/content/snatch_web_dataset'
os.makedirs(STATE, exist_ok=True)
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip() if shutil.which('nvidia-smi') else ''
print('GPU:', gpu or 'none (fine for part 2 only)')

# code: unpacked from Drive on every run of this cell, so an updated zip takes effect
zipfile.ZipFile(f'{DRIVE}/snatchguard_code.zip').extractall('/content/snatchguard')
%cd /content/snatchguard
!pip -q install -e ".[edge,central,dev,train,learned]"

# pipeline state on Drive: done-markers, logs, checkpoints, models, caches, results, generated configs
PERSIST = ['models', 'runtime/pipeline', 'runtime/train_runs', 'runtime/cache', 'runtime/features', 'runtime/learned',
           'runtime/tuning', 'runtime/tuning_baseline', 'runtime/fusion_dev']
GENERATED = ['configs/edge.gpu.yaml', 'configs/edge.baseline.yaml', 'configs/rules.tuned.yaml', 'configs/fusion.tuned.yaml']
for p in PERSIST:
    os.makedirs(f'{STATE}/{p}', exist_ok=True)
    if not os.path.islink(p):
        if os.path.isdir(p):
            shutil.copytree(p, f'{STATE}/{p}', dirs_exist_ok=True)
            shutil.rmtree(p)
        os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
        os.symlink(f'{STATE}/{p}', p)
os.makedirs(f'{STATE}/configs', exist_ok=True)
for f in GENERATED:
    if not os.path.islink(f):
        if os.path.exists(f):
            shutil.move(f, f'{STATE}/{f}')
        os.symlink(f'{STATE}/{f}', f)

# external test set (never trained or tuned on)
if not os.path.exists(f'{WEB}/labels.csv'):
    whole = f'{DRIVE}/snatch_web_dataset.zip'
    parts = sorted(n for n in os.listdir(DRIVE) if n.startswith('snatch_web_dataset.zip.part'))
    src = whole if os.path.exists(whole) else '/content/snatch_web_dataset.zip'
    if not os.path.exists(whole) and parts:
        with open(src, 'wb') as out:
            for n in parts:
                with open(f'{DRIVE}/{n}', 'rb') as f:
                    shutil.copyfileobj(f, out)
    if os.path.exists(src):
        zipfile.ZipFile(src).extractall(WEB)
print('web dataset:', 'yes' if os.path.exists(f'{WEB}/labels.csv') else 'NOT FOUND (the external test is skipped)')

def snatch_videos():
    \"\"\"Snatch 1.0 (12.5 GB from Zenodo) onto this runtime's disk, unless it is already here.\"\"\"
    if subprocess.run(['python', 'scripts/snatch_dataset.py', 'verify'], capture_output=True).returncode == 0:
        return print('Snatch 1.0: present')
    get_ipython().system('apt-get -qq install -y aria2 > /dev/null')     # Zenodo is slow per connection
    get_ipython().system('python scripts/snatch_dataset.py download --to /content/snatch_zips')
    get_ipython().system('python scripts/snatch_dataset.py extract --zips /content/snatch_zips')
    shutil.rmtree('/content/snatch_zips', ignore_errors=True)

# Hours of progress output streamed into the page can freeze the browser tab, and that loses the runtime: the full
# output goes to /content/run.log (each step's own log is on Drive, under state/runtime/pipeline/logs)
QUIET = '>> /content/run.log 2>&1; tail -n 20 /content/run.log'

def step(name, args=''):
    \"\"\"One pipeline step, skipped when Drive says it is done.\"\"\"
    if os.path.exists(f'runtime/pipeline/{name}.done'):
        return print('done:', name)
    get_ipython().system(f'python -m training.run_all --profile quick --web {WEB} --only {name} {args} {QUIET}')
    if not os.path.exists(f'runtime/pipeline/{name}.done'):
        raise SystemExit(f'{name} failed: see runtime/pipeline/logs/{name}.log')"""),
    (MD, """## Part 1: GPU (`Runtime → Change runtime type → T4 GPU`)

Data, training, perception caches and the learned scorer: about 5–6 h on a T4. A free session may end before
that; run the setup cell and this cell again in the next session and it continues (training resumes from its
last epoch)."""),
    (CODE, """snatch_videos()
!python -m training.run_all --profile quick --web {WEB} --until learned {QUIET}"""),
    (MD, """## Part 2: tuning and report (a CPU runtime is enough)

Rule tuning, rules + learned-scorer fusion and the acceptance report only replay the cached perception, so
switch to a CPU runtime (`Runtime → Change runtime type → CPU`) to keep the GPU allowance for training."""),
    (CODE, """for name in ('tune', 'fusion', 'report'):
    step(name)
!sed -n 1,60p runtime/tuning/acceptance_report.md"""),
    (MD, "## Part 3 (optional, GPU): the same held-out footage as simulated IR and low-light night"),
    (CODE, """snatch_videos()
step('night')
!python -m training.run_all --profile quick --web {WEB} --only report {QUIET}
!sed -n 1,60p runtime/tuning/acceptance_report.md"""),
]


def to_nb(cells):
    out = []
    for kind, src in cells:
        lines = src.split("\n")
        cell = {"cell_type": kind, "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
        if kind == CODE:
            cell.update({"execution_count": None, "outputs": []})
        out.append(cell)
    return {"cells": out, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


if __name__ == "__main__":
    path = Path(__file__).with_name("SnatchGuard_Train.ipynb")
    path.write_text(json.dumps(to_nb(cells), indent=1), encoding="utf-8")
    print(f"wrote {path}")
