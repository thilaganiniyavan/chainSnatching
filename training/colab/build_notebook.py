"""Generates SnatchGuard_Train.ipynb: the same one-command pipeline as the GPU box, on Google Colab.

The main route is a local NVIDIA GPU (docs/WINDOWS_GPU_GUIDE.md). Colab is the backup: free sessions
disconnect, so the quick profile is used and results are copied to Google Drive at the end.
"""

import json
from pathlib import Path

MD, CODE = "markdown", "code"
cells = [
    (MD, """# SnatchGuard on Google Colab (backup route)

Same pipeline as `scripts/windows/train_all.ps1`: Snatch 1.0 → training sets → pseudo-labels → fine-tuning →
perception caches → cross-validated tuning → night check → **acceptance report** (recall/precision vs 90 %).

`Runtime → Change runtime type → GPU`. The quick profile takes about 4–6 h on a T4; if the session
disconnects, run all cells again (finished steps are skipped while the runtime's disk survives)."""),
    (CODE, """!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/snatchguard'
!mkdir -p {DRIVE}"""),
    (MD, "## Code: upload `snatchguard_code.zip` to `My Drive/snatchguard/` (or clone your GitHub repo with a token)"),
    (CODE, """import getpass, os, subprocess, zipfile
ZIP, REPO = f'{DRIVE}/snatchguard_code.zip', 'thilaganiniyavan/snatchguard'
if not os.path.exists('/content/snatchguard'):
    if os.path.exists(ZIP):
        zipfile.ZipFile(ZIP).extractall('/content/snatchguard')
    else:
        token = getpass.getpass('GitHub token (read-only): ')
        subprocess.run(['git', 'clone', '--depth', '1', f'https://{token}@github.com/{REPO}.git', '/content/snatchguard'], check=True)
        del token
%cd /content/snatchguard
!pip -q install -e ".[edge,central,dev,train]\""""),
    (MD, "## Snatch 1.0 (12.5 GB) and the full pipeline"),
    (CODE, """!python scripts/snatch_dataset.py download --to /content/snatch_zips
!python -m training.run_all --profile quick --zips /content/snatch_zips --baseline"""),
    (MD, "## Results to Google Drive"),
    (CODE, """!mkdir -p {DRIVE}/results
!cp -r runtime/tuning runtime/tuning_baseline {DRIVE}/results/ 2>/dev/null; true
!cp models/det_finetuned.pt models/pose_finetuned.pt configs/edge.gpu.yaml configs/rules.tuned.yaml {DRIVE}/results/
!head -40 runtime/tuning/acceptance_report.md"""),
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
