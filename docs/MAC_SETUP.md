# SnatchGuard: running the web-data pipeline on a Mac

This guide continues SnatchGuard's web-data pipeline on a Mac. The pipeline downloads real CCTV snatch videos, cuts them into shots, removes duplicates, makes review kits for reviewers, and builds the training package for the GPU laptop.

Everything the Mac needs is on the **Expansion** hard disk, in the folder `SnatchGuard_Mac`.

**State at hand-over (11 Nov 2026, 19:00):**

| Item | State |
|---|---|
| Videos downloaded | 3,833 (about 900 more to come) |
| Videos cut into shots | 1,785 of 3,804 (2,019 left) |
| Duplicate check and pre-screen | First round partly done |
| Review kits | Not made yet |
| Code | Branch `snatchguard` |

**Every step resumes where it stopped.** Finished downloads, cut videos and checked shots are recorded in files that come with the data, so nothing is done twice.

---

## A. What's on the Expansion disk

```
Expansion/SnatchGuard_Mac/
├── snatchguard/          26 GB   the project: code, Snatch 1.0, models, detection caches
├── snatch_web_dataset/   61 GB   web videos, shot lists, review files, logs
├── verifier_data/        0.9 GB  the current training package (to be extended)
├── results_cmp/          1 MB    October GPU training results
└── MAC_SETUP.md                  this guide
```

---

## B. Check the Mac

1. **Disk:** Apple menu → About This Mac → More Info → Storage.
   - **About 100 GB free:** copy everything to the Mac (option C3-1). This is the recommended option.
   - **Less than that:** run from the Expansion disk (option C3-2). It's slower, but it works.
2. **RAM:** 8 GB works; 16 GB is better.
3. **Power:** keep the Mac plugged in the whole time.

---

## C. Set up the Mac (Terminal)

### C1. Tools

If Homebrew isn't installed, install it first from https://brew.sh.

```bash
xcode-select --install
brew install python@3.12 git
```

### C2. Plug in Expansion and copy the project

The project folder **must** end up at `~/snatchguard`, because paths and caches depend on it.

```bash
V=/Volumes/Expansion/SnatchGuard_Mac
rsync -rt --progress "$V/snatchguard" ~/
cd ~/snatchguard
git pull
```

`git pull` fetches any fixes pushed after the copy. If it asks for a login, skip it; the copy works as it is.

### C3. Data: choose ONE option

**C3-1. Copy to the Mac** (recommended, needs about 65 GB):
```bash
V=/Volumes/Expansion/SnatchGuard_Mac
rsync -rt --progress "$V/snatch_web_dataset" ~/
rsync -rt --progress "$V/verifier_data" ~/
```

**C3-2. Run from the Expansion disk** (Expansion must stay plugged in the whole time):
```bash
V=/Volumes/Expansion/SnatchGuard_Mac
ln -s "$V/snatch_web_dataset" ~/snatch_web_dataset
ln -s "$V/verifier_data" ~/verifier_data
```

**Both options, then:**
```bash
ln -sfn ~/snatch_web_dataset ~/snatchguard/data/snatch_web
```

### C4. Python environment

```bash
cd ~/snatchguard
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[edge,learned,web,dev]"
```

### C5. Check that everything works

```bash
cd ~/snatchguard && source .venv/bin/activate
python -m pytest -q tests/test_webdata.py
python -c "from ultralytics import YOLO; YOLO('models/yolo26n_512_openvino_model', task='detect'); print('detector ok')"
python -m yt_dlp --version
ls ~/snatch_web_dataset/raw/snatch/*.mp4 | wc -l
```

Expected output:
- The tests end with `passed` and no failures.
- The second command prints `detector ok`.
- The third prints a version number.
- The last prints about 3,800.

If the detector check fails, **stop and tell the project lead**. Don't switch to another detector, because the training data was built with this one.

---

## D. Resume the pipeline

Open four Terminal windows. In each one, run this first:

```bash
cd ~/snatchguard && source .venv/bin/activate
```

`caffeinate -i` keeps the Mac awake while the job runs.

**Window 1: downloads**
```bash
caffeinate -i python -m snatchguard.webdata.download --kind snatch --max 5000 \
  2>&1 | tee -a ~/snatch_web_dataset/meta/log_download_global.txt
```
The first line should say about 4,600 eligible candidates. Lines saying `has already been recorded in the archive` mean finished videos are being skipped, which is correct.

**Window 2: cutting videos into shots, worker 1**
```bash
caffeinate -i python -m snatchguard.webdata.segment --kind snatch \
  2>&1 | tee -a ~/snatch_web_dataset/meta/log_segment_backlog.txt
```

**Window 3: cutting, worker 2 (works from the other end of the list)**
```bash
caffeinate -i python -m snatchguard.webdata.segment --kind snatch --reverse \
  2>&1 | tee -a ~/snatch_web_dataset/meta/log_segment_reverse.txt
```

**Window 4: duplicate check and pre-screen, repeated every 10 minutes**
```bash
caffeinate -i scripts/webpool_process.sh \
  2>&1 | tee -a ~/snatch_web_dataset/meta/log_webpool_process.txt
```

**When window 3 prints `all videos segmented`:**
1. Press **Ctrl+C** in window 2.
2. Run:
   ```bash
   python -m snatchguard.webdata.segment --kind snatch --merge
   python -m snatchguard.webdata.segment --kind snatch
   ```
   The second command catches videos downloaded since the workers started.

**When window 1 finishes downloading:**
1. Run the window 2 command once more.
2. Let window 4 finish its last round. It stops by itself and prints `=== done`.

---

## E. Review kits for the reviewers

As soon as window 4 shows `pool to review:` with a number above zero:

```bash
python -m snatchguard.webdata.trainpool kit --batches 4 --per-batch 150 --tag kitA
```

- The kits appear as zips in `~/snatch_web_dataset/review/kits/`, about 150 MB each.
- Share them with reviewers through a **private** Drive link. Never make the link public.
- Each kit contains `HOW_TO_REVIEW.txt`, which explains how to fill in `review.csv`.
- For more kits later, change the tag: `--tag kitB`, `--tag kitC`, and so on. A clip never goes into two kits.

**When reviewers send their `review.csv` files back,** every file has the same name. Save each one in `~/Downloads/` under its kit's name, for example `review_kitA_01.csv` and `review_kitA_02.csv`, then run:

```bash
python -m snatchguard.webdata.trainpool import ~/Downloads/review_*.csv
```

If any row has a mistake, **nothing** is imported and the command names the row and the problem. Fix that row in the CSV and run the import again.

---

## F. Build the new training package (after the review cutoff)

```bash
python -m snatchguard.webdata.trainpool build
python -m snatchguard.webdata.trainpool leaks
python -m snatchguard.webdata.trainpool build
python -m snatchguard.webdata.trainpool cache
python -m snatchguard.verifier.finetune pack --out ~/verifier_data
cd ~ && zip -r verifier_data.zip verifier_data
```

- `build` runs twice on purpose: the second run drops any upload the leak check found to match the test footage.
- `cache` runs the detector on every new clip. It's the slowest step, taking about as long as the footage itself.
- Send `verifier_data.zip` to the GPU laptop and follow `docs/GPU_VERIFIER.md` there.

---

## G. Rules

- **Don't** rename or move `~/snatchguard` or `~/snatch_web_dataset`.
- **Don't** unplug Expansion while jobs run if you chose option C3-2.
- **Never** put Snatch-Web test clips into a review kit or into training.
- **Don't** edit files in `~/snatch_web_dataset/meta/` or `review/` by hand.
- **Don't** share or upload the video clips publicly. They're research data from public uploads.
- **Don't** run the pipeline on the Linux laptop as well. Only the Mac's copy is updated from now on.
- If something fails, send the project lead the last 20 lines of that Terminal window.

---

## Quick progress check (any time)

```bash
M=~/snatch_web_dataset/meta
echo "downloaded: $(wc -l < $M/downloaded_snatch.txt)"
tail -n 1 $M/log_segment_backlog.txt $M/log_segment_reverse.txt
tail -n 3 $M/log_webpool_process.txt
```
