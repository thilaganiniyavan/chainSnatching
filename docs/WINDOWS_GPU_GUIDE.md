# Training, evaluation and the demo on Windows 11 + NVIDIA GPU (RTX 4060)

This takes you from a fresh Windows PC to the acceptance report and a live two-machine demo.
Everything heavy runs on the GPU box; the laptop only runs the dashboard.

| what | where | time (RTX 4060) |
|---|---|---|
| setup | GPU box | 15 min |
| Snatch 1.0 download (12.5 GB) | GPU box | 1–3 h, depends on your connection |
| full pipeline (`train_all.ps1`) | GPU box | about 12–18 h (`-Profile quick`: about 4–6 h) |
| demo | GPU box + laptop | live |

Disk: keep **150 GB free** (dataset zips + videos ≈ 25 GB, training sets ≈ 40 GB, perception caches, runs).

## 1. Setup (once)

1. Install the latest **NVIDIA driver** (GeForce Experience / Studio driver). Check in PowerShell: `nvidia-smi`.
2. Install **Python 3.12** from python.org (tick *Add python.exe to PATH*) and **Git for Windows**.
3. Get the code: `git clone https://github.com/thilaganiniyavan/snatchguard.git` (or unzip the code zip),
   then in PowerShell inside that folder:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1
```

It creates `.venv`, installs the CUDA build of PyTorch that works with your driver, installs
SnatchGuard, prints your GPU, and runs the tests. All tests should pass.

## 2. Snatch 1.0 (once)

```powershell
.\.venv\Scripts\python.exe scripts\snatch_dataset.py download --to D:\snatch_zips   # resumes after drops
```
(or download the 7 zips yourself from https://zenodo.org/records/10812645 into one folder).
The pipeline unpacks them in the next step.

## 3. Everything: training → evaluation → acceptance report

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\train_all.ps1 -Zips D:\snatch_zips -Baseline
```

* `-Baseline` also measures the *pretrained* models first, so the report can show before → after.
* `-Profile quick` gives a first result in a few hours with smaller models and data.
* The PC is kept awake while it runs. If it stops (reboot, power cut, error), **run the same command
  again**: finished steps are skipped and training resumes from its last epoch.
* Logs: `runtime\pipeline\logs\<step>.log`. Progress markers: `runtime\pipeline\*.done`.
* Re-run from a step: `... -Extra "--from","cache"` (e.g. after changing `configs\rules.yaml`).

What the steps do, in order:

| step | what happens | output |
|---|---|---|
| check | GPU, disk, packages | |
| data | unpack + verify all 64 labelled videos | `data\samples\snatch1.0\` |
| models | pretrained YOLO26 detector, pose, YOLO26x teacher | `models\` |
| baseline | (with `-Baseline`) caches + tuning with pretrained models | `runtime\tuning_baseline\` |
| datasets | ExDark, LLVIP, VisDrone, COCO + night / CCTV variants | `data\datasets\` |
| pseudo | teacher labels on the 15 *pseudo-fold* normal videos | added to `sg_det` |
| train_det | fine-tune the detector (YOLO26s, 960 px) | `runtime\train_runs\det\` |
| compare | day vs night mAP, pretrained vs fine-tuned | `runtime\train_runs\compare_det.json` |
| train_pose | fine-tune the pose model | `runtime\train_runs\pose\` |
| install | copy weights, write `configs\edge.gpu.yaml` | `models\*_finetuned.pt` |
| cache | perception once over all evaluation videos | `runtime\cache\` |
| tune | cross-validated rule tuning against 90 % / 90 % | `runtime\tuning\` |
| report | **acceptance report**, deploys the tuned rules | `runtime\tuning\acceptance_report.md` |

### How the 90 % is measured (so the number holds up in front of a panel)

* **Recall**: of the labelled snatch events (motorcycle and on-foot clips), how many got an alert at the
  right moment. **Precision**: of all alerts (on snatch clips *and* on 3.1 h of normal CCTV), how many
  were real. Both are event-level and include 95 % confidence intervals.
* The data is split three ways and nothing is scored on data it was fitted to:
  the detector learns from the `pseudo` fold of normal videos only; rule thresholds are tuned on fold A
  and scored on fold B and vice versa (cross-validation); the reported number is the pooled
  held-out score.
* "Alert" means anything that notifies the dashboard (CONFIRMED or SUSPECT). Use
  `python -m snatchguard.evaluation.tune --level confirmed` for the stricter CONFIRMED-only figure.

If the report says **NOT MET**, it lists every missed clip and every false alarm, which tells you
whether to add data, adjust labels, or change a rule. Re-run from `tune` (fast) or `cache`.

## 4. The two-machine demo

**Laptop** (dashboard; same Wi-Fi/LAN as the GPU box):

```bash
SNATCHGUARD_API_KEYS=my-edge-key SNATCHGUARD_DASHBOARD_PASSWORD=choose-one scripts/laptop/run_central.sh
```
It prints the laptop's address, e.g. `http://192.168.1.20:8000`. Open it in a browser
(user `operator`). If the GPU box can't connect, run `sudo ufw allow 8000/tcp` on the laptop.

**GPU box** (edge node):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\run_edge.ps1 -Central http://192.168.1.20:8000 -ApiKey my-edge-key
```
By default the camera is the **demo playlist** (`configs\playlists\demo.txt`): normal Hyderabad CCTV
from the held-out fold with real snatch clips in between, played in real time as one live camera.
Alerts with snapshot, clip and evidence appear on the dashboard within a few seconds.

Real camera instead: add `-Source rtsp://user:pass@camera-ip/stream` (or a webcam index such as `0`).
Night simulation on a day video: edit the camera in `configs\edge.gpu.yaml` (`simulate_night: ir`).

## Troubleshooting

| symptom | fix |
|---|---|
| `PyTorch cannot see a CUDA GPU` | update the NVIDIA driver, re-run `setup.ps1` |
| CUDA out of memory in training | already AutoBatch; close other GPU apps, or use `-Profile quick` |
| `step 'datasets' failed` with LLVIP | the run continues without it; check `runtime\pipeline\logs\datasets.log` |
| the edge says `delivery ... failed, will retry` | laptop IP/port/firewall; alerts are queued and sent when it connects |
| dashboard asks for a login | user `operator`, password = `SNATCHGUARD_DASHBOARD_PASSWORD` on the laptop |
| scripts blocked | run them with `powershell -ExecutionPolicy Bypass -File ...` as shown |
