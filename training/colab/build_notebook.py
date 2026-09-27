"""Generates SnatchGuard_Train.ipynb (keeps the notebook reviewable as plain Python)."""

import json
from pathlib import Path

MD, CODE = "markdown", "code"
cells = [
    (MD, """# SnatchGuard: night-robust perception training (Google Colab)

Only **perception** is trained here: the YOLO26 detector (person / bicycle / motorcycle) and the YOLO26 pose model.
The snatch decision on the edge stays **rule-based**.

**Before you start:** `Runtime → Change runtime type → T4 GPU` (or better).

Training checkpoints and exported models are written to Google Drive, so if Colab disconnects,
just run all cells again: datasets are rebuilt and training **resumes** from the last epoch.

| Step | Time on a T4 (approx.) |
|---|---|
| Build datasets | 30–60 min (downloads) |
| Pseudo-label your CCTV (optional, recommended) | 20–40 min |
| Train detector (40 epochs) | 3–5 h |
| Train pose (25 epochs) | 2–3 h |"""),
    (CODE, """!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/snatchguard'
!mkdir -p {DRIVE}"""),
    (MD, """## 1. Get the code
**Option A (simplest):** upload `snatchguard_code.zip` to `My Drive/snatchguard/` before running this cell.

**Option B:** if the zip isn't there, the cell clones the private GitHub repo. Create a fine-grained token with
*read-only Contents* access to the `snatchguard` repo and paste it when asked (it isn't saved)."""),
    (CODE, """import getpass, os, subprocess, zipfile
ZIP = f'{DRIVE}/snatchguard_code.zip'
REPO = 'thilaganiniyavan/snatchguard'   # only used for option B
if not os.path.exists('/content/snatchguard'):
    if os.path.exists(ZIP):
        zipfile.ZipFile(ZIP).extractall('/content/snatchguard')
        print('code from', ZIP)
    else:
        token = getpass.getpass('GitHub token (read-only): ')
        subprocess.run(['git', 'clone', '--depth', '1', f'https://{token}@github.com/{REPO}.git',
                        '/content/snatchguard'], check=True)
        del token
%cd /content/snatchguard
!ls"""),
    (CODE, """!pip -q install -e ".[edge]" huggingface_hub
import ultralytics; print('ultralytics', ultralytics.__version__)"""),
    (MD, """## 2. Build the datasets
* **ExDark** (real low-light photos), **LLVIP** (real infrared surveillance), **VisDrone** (elevated street views with riders
  and motorcycles), **COCO** street images.
* Daytime images get synthetic **IR / low-light** variants and **CCTV degradation** (blur, heavy JPEG, low resolution).
* Separate `val_day` / `val_night` sets so night accuracy is reported on its own.

Datasets and downloads live on Colab's local disk (fast, large). If the runtime is reset this step is simply
re-run (about 30-60 min); training checkpoints are on Drive and are never lost."""),
    (CODE, """!python training/prepare_dataset.py --out /content/datasets --cache /content/cache \\
    --coco-images 12000 --visdrone-images 6000 --llvip-images 6000 --pose-images 15000"""),
    (MD, """## 3. (Recommended) Teach the model your cameras: teacher → student pseudo-labels
A large YOLO26x labels frames from real CCTV recordings; the small edge model then trains on them.
Below we use the **Snatch 1.0** normal videos (real Hyderabad CCTV). `s_2.avi` and `s_3.avi` are held out because the
false-alarm benchmark uses them. Replace with footage from **your own cameras** as soon as you have it: that is the
single biggest accuracy gain."""),
    (CODE, """!python scripts/snatch_dataset.py get-all --category Normal --max 8 --max-mb 300
!python training/pseudo_label.py --videos "data/samples/snatch1.0/Chain Snatching Videos/Normal" --out /content/datasets/sg_det \\
    --every 2.0 --max-per-video 300 --exclude s_2.avi s_3.avi --night-aug"""),
    (MD, "## 4. Train the detector (resumes automatically after a disconnect)"),
    (CODE, """!python training/train.py det --data /content/datasets/sg_det --epochs 40 --imgsz 640 --batch 32 \\
    --project {DRIVE}/runs --export-to {DRIVE}/export"""),
    (MD, "## 5. Day vs night accuracy: pretrained vs fine-tuned (put this table in the report)"),
    (CODE, """!python training/train.py compare --kind det --data /content/datasets/sg_det --imgsz 640 \\
    --weights {DRIVE}/runs/det/weights/best.pt --project {DRIVE}/runs
!cat {DRIVE}/runs/compare_det.json"""),
    (MD, "## 6. Train the pose model"),
    (CODE, """!python training/train.py pose --data /content/datasets/sg_pose --epochs 25 --imgsz 640 --batch 32 \\
    --project {DRIVE}/runs --export-to {DRIVE}/export
!python training/train.py compare --kind pose --data /content/datasets/sg_pose --imgsz 640 \\
    --weights {DRIVE}/runs/pose/weights/best.pt --project {DRIVE}/runs"""),
    (MD, """## 7. Bring the models to the laptop / edge box
Download `MyDrive/snatchguard/export/det/best.pt` and `export/pose/best.pt`, then on the laptop:

```bash
python scripts/fetch_models.py --from-colab path/to/det/best.pt path/to/pose/best.pt
# then set models.detector / models.pose in configs/edge.yaml to the printed paths
python -m snatchguard.evaluation.benchmark --only snatch_bike    # before/after comparison
```"""),
    (CODE, "!ls -R {DRIVE}/export | head -40"),
]


def to_nb(cells):
    out = []
    for kind, src in cells:
        lines = src.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == CODE:
            cell.update({"execution_count": None, "outputs": []})
        out.append(cell)
    return {"cells": out, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


if __name__ == "__main__":
    path = Path(__file__).with_name("SnatchGuard_Train.ipynb")
    path.write_text(json.dumps(to_nb(cells), indent=1))
    print(f"wrote {path}")
