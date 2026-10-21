# SnatchGuard: run the training and the accuracy test on a GPU laptop

This trains the system's detection models on your NVIDIA GPU and then measures whether it reaches
the target: **at least 90 % of real chain snatchings caught (recall) and at least 90 % of alerts
being real snatchings (precision)**, on footage it has never been trained or tuned on.

One script does everything. It runs for about **13–20 hours**; you mostly just leave it running.

---

## What you need

| | |
|---|---|
| Computer | Windows 10/11 laptop or PC with an **NVIDIA GPU** (RTX 3060/4060 or better, 8 GB+ VRAM) |
| Disk | **80 GB free** (uses ~60 GB: videos 12 GB, training data ~25 GB, Python/GPU libraries ~7 GB, results; ~12 GB less with `--move` in step 3) |
| Internet | to install packages and download training datasets (~10 GB) |
| Time | 15 min setup + 13–20 h unattended run |
| Files | `snatchguard_code.zip` (from your friend), the **Snatch 1.0 videos** you already have, and (optional) the `snatch_web_dataset` folder |

---

## Step 1: Install the basics (once)

1. **NVIDIA driver**: install the latest Game Ready or Studio driver from nvidia.com.
   Check it: open PowerShell and type `nvidia-smi`. You should see your GPU name.
2. **Python 3.12**: download from https://www.python.org/downloads/ . In the installer, **tick
   "Add python.exe to PATH"**.

## Step 2: Unpack the code and run setup

1. Unzip `snatchguard_code.zip` to a short path, for example `C:\snatchguard`.
2. Open **PowerShell** in that folder (in File Explorer: open the folder, type `powershell` in the
   address bar, press Enter).
3. Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1
```

This takes about 10–15 minutes. It installs PyTorch for your GPU and everything else, prints your GPU
name, and runs the self-tests. It should end with `... passed` (no failures) and `Setup done`.

## Step 3: Point it at your Snatch 1.0 videos

You already have the videos. They can be in any form: the 7 zip files
(`Snatch 1.0-20201231T071009Z-001.zip` … `-007.zip`), the unzipped `Snatch 1.0` folder, or the video files
loose in a folder. Note the folder they are in (for example `D:\Snatch 1.0`) and run, in the same
PowerShell window:

```powershell
.\.venv\Scripts\python.exe scripts\snatch_dataset.py import --from "D:\Snatch 1.0"
```

(Use your own folder path; keep the quotes if it has spaces.) It finds every video by name and copies it
into the project (a few minutes). Add `--move` at the end to move instead of copy if disk space is tight.
It **must end with `64/64 labelled videos present`**; if not, it lists the missing file names.

## Step 4: Run everything

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\train_all.ps1 -Baseline -Web "D:\snatch_web_dataset"
```

`-Web` points at the `snatch_web_dataset` folder from your friend (it must contain `labels.csv` and
`clips`). It is an extra test on footage from other cities; leave `-Web ...` out if you don't have it.

**Already ran an earlier version of this?** Unzip the new code over the old folder (keep `data`, `models`
and `runtime`), run `setup.ps1` again, then instead of the command above run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\train_all.ps1 -Web "D:\snatch_web_dataset" -Extra "--from","pseudo"
```

It keeps the downloaded training images and retrains with the corrected settings (about 11–16 hours).

Leave it running (about 13–20 hours). The laptop is kept awake automatically while it runs.
**Keep it plugged in.** You will see progress for each step:

| step | what it does | rough time |
|---|---|---|
| check, data, models | checks the GPU, videos and downloads the base models | minutes |
| baseline | measures accuracy *before* training (for comparison) | 1–2 h |
| datasets | downloads and prepares night / CCTV training images | 30–60 min |
| pseudo | labels real CCTV frames with a large "teacher" model | 20–40 min |
| train_det | trains the person / motorcycle detector | 6–9 h |
| compare | day vs night accuracy, before vs after | minutes |
| train_pose | trains the body-keypoint model | 2–3 h |
| install, cache | installs the trained models and runs them over the test videos | 1–2 h |
| learned | trains the learned snatch scorer on the same videos (cross-validated) | 30–60 min |
| tune, fusion, night | tunes the rules and how they are combined with the learned scorer | 30–60 min |
| report | tests on the web dataset (if given) and writes the final report | 15–30 min |

**If it stops** (error, reboot, power cut, laptop closed): run the **same command again**. Finished
steps are skipped and training continues from where it stopped.

The very last lines show the result, for example:

```
RESULT (cross-validated): recall 92.0% ..., precision 95.8% ... -> PASS
```

## Step 5: Send these files back

Zip them up and send them to your friend:

| file | what it is |
|---|---|
| `runtime\tuning\acceptance_report.md` | **the result**: PASS / NOT MET against 90 %, every video's outcome |
| `runtime\tuning_baseline\acceptance_report.md` | the same test before training (for comparison) |
| `runtime\tuning\external.json`, `runtime\tuning\fusion.json` | web-dataset test; rules + learned scorer results |
| `runtime\train_runs\compare_det.json` | detector accuracy, day vs night vs real CCTV, before vs after |
| `models\det_finetuned.pt` and `models\pose_finetuned.pt` | the trained models |
| `models\learned\` (the whole folder) and `runtime\learned\report.md` | the learned scorer and its report |
| `configs\rules.tuned.yaml`, `configs\fusion.tuned.yaml` and `configs\edge.gpu.yaml` | the tuned settings |
| `runtime\pipeline\logs\` (the whole folder) | logs, in case something needs checking |

---

## If something goes wrong

| what you see | what to do |
|---|---|
| `nvidia-smi` is not recognised | install / update the NVIDIA driver, restart, try again |
| `Python not found` | reinstall Python 3.12 and tick "Add python.exe to PATH" |
| `No CUDA-enabled PyTorch could see the GPU` | update the NVIDIA driver, restart, run `setup.ps1` again |
| `running scripts is disabled on this system` | use the commands exactly as written above (with `-ExecutionPolicy Bypass`) |
| import says fewer than `64/64` | it lists the missing names: check that folder (or zip part) is in the path you gave |
| `CUDA out of memory` | close other programs using the GPU (games, browsers with video) and run the same command again |
| `step '...' failed` | run the same command again; if it fails the same way, send `runtime\pipeline\logs\<step>.log` |
| the PC went to sleep / was restarted | run the same command again; it continues |

Anything else: send a screenshot of the PowerShell window and the matching log file from
`runtime\pipeline\logs\`.

---

## Optional: the live demo with two machines

After the run finishes, this GPU laptop can act as the "edge camera box" and send live alerts to the
dashboard on your friend's laptop (both on the same Wi-Fi):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\run_edge.ps1 -Central http://<friend's-laptop-IP>:8000 -ApiKey <key from your friend>
```

It plays real CCTV clips as a live camera; alerts appear on the dashboard within seconds.
Details: `docs\WINDOWS_GPU_GUIDE.md`.
