# Fine-tune the verifier on a GPU laptop

This retrains SnatchGuard's learned verifier (the video network that checks each close contact between two
people) end to end on an NVIDIA GPU, measures it on footage it never saw, and keeps the best model on the GPU
laptop for the live system. Everything that needs the detection models has already been done on the development
laptop; the GPU laptop only needs PyTorch and two zip files. Nothing is installed outside the project folder.

| | |
|---|---|
| Computer | Windows 10/11 or Linux, **NVIDIA GPU with 4 GB or more** (RTX 2060/3050/3060/4050/4060 or better) |
| Disk | about 12 GB free (PyTorch with CUDA ~6 GB, the data ~1.2 GB, results and models ~1 GB) |
| Internet | for the setup (PyTorch download, ~3 GB) and, on the first run, the two pretrained networks (~250 MB) |
| Time | 15 min setup (longer on a slow connection), then about 2-5 hours unattended (`-SkipSwin`: about 40 % less) |
| Files | `snatchguard_code.zip` and `verifier_data.zip` (from the SnatchGuard team) |

The data is crops of real CCTV clips for research use only: do not share or upload it anywhere, and delete it
after the run.

## Windows

1. **NVIDIA driver**: install the latest Game Ready or Studio driver from nvidia.com. Check it: open PowerShell
   and type `nvidia-smi`; it should print the GPU's name.
2. **Python 3.12** from https://www.python.org/downloads/ ; in the installer tick **"Add python.exe to PATH"**.
3. Unzip `snatchguard_code.zip` to `C:\snatchguard`, and unzip `verifier_data.zip` to `C:\` (it holds a
   `verifier_data` folder, so you get `C:\verifier_data` with `index.json` and `strips` inside; short paths
   without spaces are safest).
   Already have `C:\snatchguard` from the earlier SnatchGuard run? Unzip the new code over it (keep `.venv`).
4. Open PowerShell in `C:\snatchguard` (in File Explorer type `powershell` in the address bar, press Enter) and run:

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1
   ```

   It installs PyTorch for the GPU and the rest (10-15 min) and ends with `Setup done`. Test failures that
   mention missing videos or models do not matter for this job.
5. A 10-minute check that everything works (both networks, a few training steps each):

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\windows\train_verifier.ps1 -Data C:\verifier_data -Quick
   ```

   It should end with `Quick check passed`.
6. The real run (keep the laptop plugged in; it is kept awake automatically):

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\windows\train_verifier.ps1 -Data C:\verifier_data
   ```

   It trains R(2+1)D-18 (11 runs: `external`, `external-probe`, `curve0/33/67`, `s1-cv0` ... `s1-cv4`, `final`),
   then Swin3D-T (8 runs: `external`, `external-probe`, `s1-cv0..4`, `final`); progress lines show the time left
   of each run. Short on time? Add `-SkipSwin`. 4-6 GB GPU? Add `-Batch 8` (it also halves the batch by itself
   when the GPU runs out of memory).
7. When it prints `Done. Send back this one file`, send **`C:\snatchguard\verifier_ft_results.zip`** (a few MB:
   reports, logs, test scores) back to the team. The trained models stay on this laptop:
   `models\verifier\verifier_ft.pt` is the one the live system uses.

## Linux

```bash
unzip snatchguard_code.zip -d ~/snatchguard && unzip verifier_data.zip -d ~
cd ~/snatchguard
bash scripts/train_verifier.sh ~/verifier_data --quick     # creates .venv on first use, then a 10-minute check
bash scripts/train_verifier.sh ~/verifier_data             # the real run (SKIP_SWIN=1 for R(2+1)D-18 only)
```

Send back `~/snatchguard/verifier_ft_results.zip`.

## After training: run it live (Windows)

1. Unzip `snatch_web_test_clips.zip` into `C:\snatchguard` (it creates `data\snatch_web\clips`).
2. Optional, phone alerts (2 minutes, once): in Telegram message **@BotFather**, send `/newbot`, copy the token;
   send your new bot any message; then in PowerShell in `C:\snatchguard`:

   ```powershell
   .\.venv\Scripts\python.exe -m snatchguard.central.notify chat-id --token <token>
   .\.venv\Scripts\python.exe -m snatchguard.central.notify test --token <token> --chat <chat id>
   ```

   The phone should show a test alert.
3. Start it:

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\windows\run_live.ps1 -TelegramToken <token> -TelegramChat <chat id>
   ```

   The dashboard opens in the browser. The demo camera plays the 36 Snatch-Web test clips (25 real snatchings,
   11 normal street scenes, about 30 minutes) with the model trained **without** them
   (`models\verifier\verifier_ft_external.pt`), so what it catches is what it would catch on unseen footage.
   Each alert arrives on the phone with a snapshot; the evidence clip follows as a reply. `CONFIRMED` alerts
   cleared the stricter threshold (1 false alarm per hour on held-out footage), `SUSPECT` ones the 10 per hour one:
   check their clip before acting. Delays (contact -> alert -> phone) are logged in `runtime\central\notify.jsonl`.
4. A real camera instead: `-Source 0` (webcam) or `-Source rtsp://user:pass@camera-ip/stream`; it then uses
   `models\verifier\verifier_ft.pt`, the model trained on all the data.

## If something goes wrong

| what you see | what to do |
|---|---|
| `nvidia-smi` is not recognised | install / update the NVIDIA driver, restart, try again |
| `No CUDA-enabled PyTorch could see the GPU` | update the NVIDIA driver, restart, run `setup.ps1` again |
| `GPU out of memory: starting again with batch 8` | nothing: it retries with smaller batches by itself |
| `swin3d_t failed ... continuing` | nothing: the R(2+1)D-18 results are complete; send the zip as usual |
| it is very slow (more than 6 hours) | close games / browsers using the GPU; add `-Batch 8` or `-SkipSwin` |
| `has no index.json` | `-Data` must point at the folder that contains `index.json` and `strips` |
| the laptop slept or restarted | run the same command again (it starts the runs from the beginning) |
| anything else | send a screenshot and `runtime\verifier_ft\run.log` |

## What it does (for the team)

`python -m snatchguard.verifier.finetune run` trains one pretrained video network (`--arch`): R(2+1)D-18
(Kinetics-400 top-1 67.5 %, native 112 px) or Swin3D-T (77.7 %, trained at 224 px; its window attention takes the
same 12-frame 112 px clips, so the live clip format does not change). Earlier blocks are frozen (batch-norm
statistics fixed); clips around every contact candidate are trained with random time shifts, crops, flips, colour
changes and drawn-on news graphics. Training-pool events have times but no actor labels, so their candidates in
the event window are a bag of which at least one must score high (multiple-instance learning); where a person has
checked the bag (`trainpool bags`), the clips that show the grab are positives outright. News uploads often
circle the grab, so drawn-on graphics are added to 40 % of negative and 10 % of positive training clips. Most
positives are web clips and most normal footage is Snatch 1.0, so each source's negatives (Snatch 1.0, web) are
drawn as often as its share of the positives; otherwise the network can learn "web footage = snatch".

Every test clip is scored twice: its live view alone, and the mean logit over six views (the window shifted by
0.73 s either way, each also mirrored), which the live system reproduces on a GPU by cutting the shifted clips
from its frame buffer (alerts come 0.73 s later). Each protocol fixes, on held-out normal footage (20 % of each
source's normal hours, whole cameras or uploads) and before scoring its test set, two thresholds per scoring:
SUSPECT at 10 and CONFIRMED at 1 false alarm per hour on each source. The report gives events caught and false
alarms per hour at both, with confidence intervals:

* `external`: trained on Snatch 1.0 and the web training pool, tested on the frozen Snatch-Web test set
  (other cities and cameras; never trained or tuned on). Its model is kept (`verifier_ft_external.pt`): it is the
  one to demonstrate on Snatch-Web clips, as every other model has trained on them.
* `external-probe`: the same with the network frozen (only the last layer learns): what fine-tuning adds over
  more data alone.
* `curve0/33/67` (R(2+1)D-18 only): `external` with 0, 1/3 and 2/3 of the pool's events: how recall grows with
  training events.
* `s1-cv0..4`: five folds holding out Snatch 1.0 cameras and event videos.
* `final`: everything except the validation split; `verifier_ft.pt` is the model to deploy.

`python -m snatchguard.verifier.finetune select` then picks the network and the scoring (live view or six views)
to deploy by a rule fixed before any run: the most Snatch 1.0 cross-validation events caught, then the higher mean
clip AUC there; ties keep R(2+1)D-18 and the single view. The Snatch-Web test set plays no part in the choice, so
it stays a held-out result for every candidate. The chosen `final` model is copied to
`models/verifier/verifier_ft.pt` with its choice (`tta_default`) and `models/verifier/selection.json`.

The package is made on the development laptop (perception caches must exist). Strips are named by candidate
number, so after any change to the candidate stage pack into a new, empty folder:

```bash
python -m snatchguard.webdata.trainpool build && python -m snatchguard.webdata.trainpool cache
python -m snatchguard.verifier.finetune pack --out ~/verifier_data
```
