# Fine-tune the verifier on a GPU laptop

This retrains SnatchGuard's learned verifier (the video network that checks each close contact between two
people) end to end on an NVIDIA GPU, then measures it on footage it never saw. Everything that needs the
detection models has already been done on the development laptop; the GPU laptop only needs PyTorch and
two zip files. Nothing is installed outside the project folder.

| | |
|---|---|
| Computer | Windows 10/11 or Linux, **NVIDIA GPU with 4 GB or more** (RTX 2060/3050/3060/4050/4060 or better) |
| Disk | about 12 GB free (PyTorch with CUDA ~6 GB, the data ~1.5 GB, results ~0.5 GB) |
| Internet | only for the setup (PyTorch download, ~3 GB) |
| Time | 15 min setup, then about 1-3 hours unattended (about 1 h on an RTX 4060, 2-3 h on an RTX 3050) |
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
5. A 5-minute check that everything works:

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\windows\train_verifier.ps1 -Data C:\verifier_data -Quick
   ```

   It should end with `Quick check passed`.
6. The real run (keep the laptop plugged in; it is kept awake automatically):

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\windows\train_verifier.ps1 -Data C:\verifier_data
   ```

   Progress lines show each of the 11 training runs (`external`, `external-probe`, `curve0/33/67`,
   `s1-cv0` ... `s1-cv4`, `final`) with the time left for that run.
7. When it prints `Done. Send back this one file`, send **`C:\snatchguard\verifier_ft_results.zip`**
   (about 250 MB: two trained models, the report and the log) back to the team, e.g. by Google Drive or
   WeTransfer.

## Linux

```bash
unzip snatchguard_code.zip -d ~/snatchguard && unzip verifier_data.zip -d ~
cd ~/snatchguard
bash scripts/train_verifier.sh ~/verifier_data --quick     # creates .venv on first use, then a 5-minute check
bash scripts/train_verifier.sh ~/verifier_data             # the real run
```

Send back `~/snatchguard/verifier_ft_results.zip`.

## If something goes wrong

| what you see | what to do |
|---|---|
| `nvidia-smi` is not recognised | install / update the NVIDIA driver, restart, try again |
| `No CUDA-enabled PyTorch could see the GPU` | update the NVIDIA driver, restart, run `setup.ps1` again |
| `GPU out of memory: starting again with batch 8` | nothing: it retries with smaller batches by itself |
| it is very slow (more than 4 hours) | close games / browsers using the GPU; or add `-Batch 8` |
| `has no index.json` | `-Data` must point at the folder that contains `index.json` and `strips` |
| the laptop slept or restarted | run the same command again (it starts that run from the beginning) |
| anything else | send a screenshot and `runtime\verifier_ft\run.log` |

## What it does (for the team)

`python -m snatchguard.verifier.finetune run` trains R(2+1)D-18 (Kinetics-400 start, stem and first block
frozen, batch-norm statistics fixed) on 12-frame clips around every contact candidate, with random time shifts,
crops, flips, colour changes and drawn-on news graphics. Training-pool events have times but no actor labels,
so their candidates in the event window are a bag of which at least one must score high (multiple-instance
learning); where a person has checked the bag (`trainpool bags`), the clips that show the grab are positives
outright. News uploads often circle the grab, so drawn-on graphics are added to 40 % of negative and 10 % of
positive training clips. Most positives are web clips and most normal footage is Snatch 1.0, so each source's
negatives (Snatch 1.0, web) are drawn as often as its share of the positives; otherwise the network can learn
"web footage = snatch". Each protocol fixes its alert threshold on held-out normal footage (20 % of each source's
normal hours, whole cameras or uploads) at 10 false alarms per hour on each source before scoring its test set:

* `external`: trained on Snatch 1.0 and the web training pool, tested on the frozen Snatch-Web test set
  (other cities and cameras; never trained or tuned on). Its model is kept (`verifier_ft_external.pt`) so the
  scores can be re-checked on the development laptop through OpenVINO.
* `external-probe`: the same with the network frozen (only the last layer learns): what fine-tuning adds over
  more data alone.
* `curve0/33/67`: `external` with 0, 1/3 and 2/3 of the pool's events: how recall grows with training events.
* `s1-cv0..4`: five folds holding out Snatch 1.0 cameras and event videos.
* `final`: everything except the validation split; `verifier_ft.pt` is the model to deploy
  (`configs/edge.yaml` -> `verifier.model`).

The package is made on the development laptop (perception caches must exist):

```bash
python -m snatchguard.webdata.trainpool build && python -m snatchguard.webdata.trainpool cache
python -m snatchguard.verifier.finetune pack --out ~/verifier_data
```
