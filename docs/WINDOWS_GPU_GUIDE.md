# Training, evaluation and the demo on Windows 11 + NVIDIA GPU (RTX 4060)

This takes you from a fresh Windows PC to the acceptance report and a live two-machine demo.
Everything heavy runs on the GPU box; the laptop only runs the dashboard.

| what | where | time (RTX 4060) |
|---|---|---|
| setup | GPU box | 15 min |
| Snatch 1.0 download (12.5 GB) | GPU box | 1–3 h, depends on your connection |
| full pipeline (`train_all.ps1`) | GPU box | about 13–20 h (`-Profile quick`: about 5–7 h) |
| demo | GPU box + laptop | live |

Disk: keep **80 GB free**. The run uses about 60 GB: Snatch 1.0 videos 12 GB (+12.5 GB if the original zips
stay on the same disk), downloaded and prepared training sets ~25 GB, Python + CUDA libraries ~7 GB,
models / caches / reports a few GB.

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

Already have the videos on another computer or a USB drive (zips, the extracted *Snatch 1.0* folder, or
loose files)? Point the importer at it; it recognises every labelled video by name:

```powershell
.\.venv\Scripts\python.exe scripts\snatch_dataset.py import --from E:\snatch   # add --move to save disk space
```
It ends with `64/64 labelled videos present`. Then run step 3 without `-Zips`.

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
* `-Web D:\snatch_web_dataset` adds the external test (25 snatch + 11 normal clips from other cities and
  cameras; copy the whole `snatch_web_dataset` folder, it needs `labels.csv` and `clips\`). It is only
  scored, never trained or tuned on. Default: `%USERPROFILE%\snatch_web_dataset` if that folder exists.
* `-Tiles 2x2` also runs the detector on 4 overlapping tiles of every frame, so small, distant bikes are
  seen at a usable size. Off by default: it costs 5 detector passes per frame (the `cache` step takes
  roughly 3x longer), and on the laptop (YOLO26n, 512 px) it helped 1 of 3 tested bike clips (23_0.mp4:
  motorcycle found in 12 instead of 7 of 24 frames, tracked in 4 instead of 0), with no change on 17_0
  and 2_0. Turn it on after a run if `python -m snatchguard.evaluation.funnel` shows bikes being lost at
  detection: `-Tiles 2x2 -Extra "--from","install"`.

**Already ran an earlier version?** The training settings and the pipeline changed, so re-run from the
pseudo-labels (the downloaded datasets are kept; training starts fresh, an older run folder is moved to
`runtime\train_runs\det_previous_<date>`):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\train_all.ps1 -Extra "--from","pseudo"
```

What the steps do, in order:

| step | what happens | output |
|---|---|---|
| check | GPU, disk, packages | |
| data | unpack + verify all 64 labelled videos | `data\samples\snatch1.0\` |
| models | pretrained YOLO26 detector, pose, YOLO26x teacher | `models\` |
| baseline | (with `-Baseline`) caches + tuning with pretrained models | `runtime\tuning_baseline\` |
| datasets | ExDark, LLVIP, VisDrone, COCO + night / CCTV variants | `data\datasets\` |
| pseudo | teacher labels on the 15 *pseudo-fold* normal videos; 3 whole videos become a real-CCTV validation set | added to `sg_det` |
| train_det | fine-tune the detector (YOLO26s, 960 px, 50 epochs) | `runtime\train_runs\det\` |
| compare | day / night / real-CCTV mAP, pretrained vs fine-tuned | `runtime\train_runs\compare_det.json` |
| train_pose | fine-tune the pose model | `runtime\train_runs\pose\` |
| install | copy weights, write `configs\edge.gpu.yaml` | `models\*_finetuned.pt` |
| cache | perception once over all evaluation videos | `runtime\cache\` |
| learned | learned snatch scorer: video features, cross-validated scores, deployed model (30–60 min) | `models\learned\`, `runtime\learned\` |
| tune | cross-validated rule tuning against 90 % / 90 % | `runtime\tuning\` |
| fusion | cross-validated tuning of how rule alerts and the learned score are combined (minutes) | `runtime\tuning\fusion.json`, `configs\fusion.tuned.yaml` |
| night | the same held-out footage as simulated IR and low-light night, scored with the tuned rules | `runtime\tuning\night.json` |
| report | external test on the web dataset (with `-Web`), then the **acceptance report**; deploys the tuned rules | `runtime\tuning\acceptance_report.md`, `runtime\tuning\external.json` |

### How the 90 % is measured (so the number holds up in front of a panel)

* **Recall**: of the labelled snatch events (motorcycle and on-foot clips), how many got an alert at the
  right moment. **Precision**: of all alerts (on snatch clips *and* on 3.1 h of normal CCTV), how many
  were real. Both are event-level and include 95 % confidence intervals.
* The data is split three ways and nothing is scored on data it was fitted to:
  the detector learns from the `pseudo` fold of normal videos only; rule thresholds, the learned scorer
  and the fusion thresholds are fitted on fold A and scored on fold B and vice versa (cross-validation);
  the reported number is the pooled held-out score. The web dataset is a fourth, fully external check.
* The real-CCTV detector validation (`compare_det.json`, `cctv`) is scored against the YOLO26x teacher's
  labels, so it measures agreement with a much larger model on real footage, not ground truth.
* The precision / recall that Ultralytics prints while training (`Box(P R mAP50 ...)`) is the detector on
  its validation images, per box. It is not the event-level snatch precision / recall of the report.
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
