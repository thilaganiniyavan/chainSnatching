# SnatchGuard – Stage 1: Dataset Inspection & Quality Assessment

Scope: read-only inspection of `chainSnatching/` and the new `dataset_1/`. Nothing in either was modified; no training was run.
Basis: files were staged to a scratch workspace and analysed with ffprobe, OpenCV (full decode of every clip), md5 and perceptual-hash (dHash) sequences. Numbers below are **measured by me** unless marked *(README)*, which means quoted from the project's own docs and **not re-measured**.

---

## 0. Executive summary

1. `dataset_1` is **not a new drop-in labelled dataset.** It is (a) a frozen 36-clip test set, (b) a 147-clip labelled training pool, (c) ~3.8 k unreviewed raw downloads, (d) a verifier package, and (e) a full copy of the project code/data. It contains **no boxes or keypoints**, so it cannot feed detector/pose training. It can feed the **learned verifier** and **evaluation**.
2. Labelled clips are technically healthy: **0 decode errors, 0 bad frames, 0 exact duplicates** across all 183 web clips.
3. The leakage risk between the pool and the frozen test set is **low and already controlled**: I found 0 near-duplicates between pool↔test and pool↔Snatch 1.0 snatch clips, no shared source uploads, and the 4 pipeline-flagged leaks are already excluded.
4. The main weaknesses are: **pool normals are unverified** (56/56 "pre-screened, not reviewed"), **small positive count** (141 in-scope events in total, ~89 usable verifier positives vs ~9.5 k negatives), **short temporal context in Snatch 1.0** (median 2 s before / 1.8 s after the grab), a **domain/"web = snatch" confound**, and a **stale verifier package** (covers 61 of 147 pool clips).
5. Verdict: **conditionally suitable** for verifier training/evaluation after the preprocessing in §9. Not suitable for detector/pose training. Real "confusing negatives" (two-wheeler passing, hugs, handovers) are scarce.

---

## 1. Structure

| Location | Content | Labelled? |
|---|---|---|
| `data/labels/snatch1.0.csv`, `actors.json` (main project) | Snatch 1.0 labels (no videos in main project) | yes |
| `dataset_1/snatch_web_dataset/clips/{normal,snatch}` | **Web test set**, 36 clips, frozen | yes |
| `dataset_1/snatch_web_dataset/train/` | **Training pool**: 147 clips, `labels.csv`, `manifest.csv`, `leak_check.json`, `bag_review.json` | yes |
| `dataset_1/snatch_web_dataset/raw/{snatch,normal}` | ~3,833 downloads *(count from `downloaded_snatch.txt`; directory listing truncated)* + 79 normal | **no** |
| `dataset_1/snatch_web_dataset/meta, review` | pipeline state (segments, dedup, prescreen, human decisions) | partly |
| `dataset_1/snatchguard/data/samples/snatch1.0/` | Snatch 1.0 videos: Normal (29 avi), Snatch Theft (35 mp4 + `s_9.avi`) | yes (via CSV) |
| `dataset_1/verifier_data/` | `index.json`, 9,877 JPEG strips, frozen embeddings | derived |
| `dataset_1/results_cmp/` | old/new GPU result zips | not analysed (results, not data) |
| `dataset_1/MAC_SETUP.md` | hand-over state, 11 Nov 2026 | – |

Label schema (same in all three label files): `video, category, lighting, event_start_s, event_end_s, fold, notes`; categories `snatch_bike | snatch_foot | normal`; in-scope = bike + foot.

## 2. Class distribution

| Set | Clips | Bike | Foot | Normal | Minutes (normal / snatch) | Timed events |
|---|---|---|---|---|---|---|
| Snatch 1.0 (labelled) | 64 | 13 | 12 | 29 (+9 "other", 1 unlocated) | – | 20 of 25 |
| Web test (frozen) | 36 | 15 | 10 | 11 | 21.0 / 9.0 | 25 of 25 |
| Web pool (train) | 147 | 64 | 27 | 56 | 58.5 / 19.6 | 91 of 91 |

- **In-scope events overall: 141** (bike 92 = 65 %, foot 49 = 35 %). Foot is the minority in every set.
- **Usable verifier positives are far fewer than events** (stale package, `index.json`): 89 positive candidate clips vs 9,533 negatives (**≈ 1 : 107**) and 255 ignored. Only the events with identified victim/attacker tracks count: README says 39; 7 Snatch 1.0 events were never tracked.
- **Lighting:** test 29 day / 7 night. Pool bike 53 day / 8 night / 2 IR, foot 18 day / 7 night; lighting is **blank for all 56 pool normals** and 3 snatch rows. Night/IR is rare everywhere.
- **Object stolen** (keyword scan of free-text notes; approximate): test – chain/necklace 11, phone 5, bag 5, unstated 4; pool – chain 32, phone 7, bag 7, unstated 45.
- Raw pool funnel (from `meta/`): 1,785 videos cut → 11,157 segments (8,912 static) → dedup 4,491 (ok 4,274 / dup_web 182 / check 22 / dup_snatch1 13) → prescreen 4,157 → human decisions 1,161 (other 715, not_cctv 161, unclear 118, no_event 80, **snatch_bike 59, snatch_foot 28** → ≈ 7.5 % yield). About half the videos are not yet segmented.

## 3. Quality issues

**Decode / integrity (measured).** All 183 web clips (36 + 147) and all 35 staged Snatch 1.0 snatch clips: 0 probe errors, 0 bad frames, 0 frame-count mismatches; no audio; no exact md5 duplicates.

**Format.** Web clips: h264 / yuv420p, re-encoded by the cutter (CRF 21). Snatch 1.0 snatch clips: 24 mpeg4 + 11 h264.

**Resolution / fps / duration / bitrate (medians).**

| | Median height | Bitrate | Sharpness* | Notes |
|---|---|---|---|---|
| Snatch 1.0 snatch clips (35) | 360 | 626 kbps | 2363 | 17 distinct resolutions; 18/35 have short side ≤ 360; only 12 are 1280×720; median duration 8 s |
| Snatch 1.0 normals | 1080 | ~4.2 Mbps | – | 1920×1080 @ 25 fps; **only `s_5.avi` probed** |
| Web pool (147) | 720 (120/147 are 1280×720) | 882 kbps | 4022 | |
| Web test (36) | 514 | 477 kbps | 2282 | one clip at 6 fps |

\*variance-of-Laplacian; relative only.

**Odd clips.** 3 portrait-orientation (`ZlaNuo2XGJ4_0`, `bUgLAbsR-bM_1`, Snatch 1.0 `7.mp4`); 5 grayscale/IR-like; 3 dark (mean luma < 70).

**Label-time errors in the pool (5 events fall outside the clip):**
- start < 0: `HmhkXz5aUN4_3` (−0.25 s), `kDW-NfzW0zw_2` (−0.07 s)
- end > duration: `8_Q3y8nQC4s_4` (8.8 vs 7.8 s), `DO-73PfXQ-k_2` (4.94 vs 4.42 s), `CWUjj9Mla2M_1` (3.09 vs 3.04 s)
Test set and Snatch 1.0 have none.

**Unverified negatives.** All 56 pool normals carry the note *"web street footage (pre-screened, not reviewed frame by frame)"*. A hidden snatch in a "normal" would become a false negative. The 11 test normals are human-reviewed, 6 are explicit hard negatives (rider waiting, U-turn beside pedestrian, crowded pavement…).

**On-screen overlays.** Pool event notes mention graphics/captions/logo bars in 35 of 91 events (bike 29/64, foot 6/27), news edits in 13, close-up/zoom in 32. These do not exist in Snatch 1.0 and can be learned as shortcuts.

**Vocabulary mismatch.** Snatch 1.0 uses `day/lowlight/ir`; the web sets use `day/night/ir`.

**Version/portability.** Verifier package paths are absolute `/home/thilagan/...`. The `dataset_1/snatchguard` copy is newer than the main project in `trainpool.py`, `segment.py`, `discover.py`.

## 4. Duplicate statistics

Method: dHash sequence (~4 fps) compared along alignment diagonals, Hamming ≤ 10; "strong" = ≥ 6 matching diagonal hits, ≥ 60 % diagonal fraction, lift ≥ 4 over the median diagonal. All pairs across test, pool and Snatch 1.0 snatch clips were compared.

- **16 strong near-duplicate pairs (26 clips). All are inside the training pool and share a source upload id.**
  - 15 pairs are normals: `-b` re-cuts and multiple segments of one camera (e.g. `Bek3iNbFYXY_33/36/42`, `RpbgLNS79l8_14/14-b`, `wiLRNffwvRk_0/0-b`, `lqI6DSpmhkM_0/5/5-b`, `sizjI3cO1G8_1/3/3-b`, `e3JZWDyG6Yk_2/5/5-b`, `Loz_MxDHnaw_0/0-b`, `PgqQPIwNAGQ_0/0-b`, `DCuzn4cakJo_0/0-b`, `LYs92vDZHKk_2/4`).
  - 1 pair is positive: `1zIVEDHkjJQ_2` / `_4` (same camera and incident).
- **Pool ↔ test: 0. Pool ↔ Snatch 1.0 snatch: 0. Test ↔ Snatch 1.0 snatch: 0.** Six visually inspected cross-set candidates were different scenes.
- Pool structure: 118 source uploads, 20 uploads have > 1 clip (49 clips, max 6); 11 `-b` re-cut variants. Test: 35 distinct uploads (`5EgvaLCfXWw` has 2 clips). Pool and test share no upload.
- Not covered by my own check: Snatch 1.0 **normals** (large `.avi` files exceed the staging limit; only `s_5.avi` was staged). That comparison relies on the pipeline's own fingerprints (`dedup.py` dropped 13 segments as `dup_snatch1`).

Files: `stage1_neardup_candidate_pairs.csv` (88 candidate rows with diag_match ≥ 3; the 16 strong pairs are the top rows meeting the full criteria), `stage1_web_clip_probe.csv` (per-clip probe).

## 5. Annotation compatibility

- Web test and pool `labels.csv` use the **same schema** as `snatch1.0.csv`; they work with `verifier/train.py`, `evaluation/external.py` and `verifier/finetune.py pack --web`.
- Pool rows have **no actor IDs** by design; they go through the multiple-instance "bag" path (`bag_review.json`: 35 reviewed videos, 31 with ≥ 1 clip showing the grab, 4 with none).
- Pool is a single fold `T`; test folds A/B; Snatch 1.0 folds A/B/pseudo/"-".
- **No bounding boxes, no keypoints, no tracks** in `dataset_1` → **incompatible with detector (YOLO26) and pose training**, which use ExDark / LLVIP / VisDrone / COCO plus Snatch 1.0 pseudo-labels.
- 5 Snatch 1.0 in-scope events are untimed (bike `13_0`, `30_0`; foot `1`, `4_0`, `18_0`); `6_0.mp4` is on disk but unlabelled.

## 6. Leakage risks

| Risk | Status |
|---|---|
| Pool ↔ frozen test, same upload | none found |
| Pool ↔ frozen test / Snatch 1.0, visually the same scene | 4 flagged by ORB (≥ 40 inliers): `tpqie1KTOjA_19`, `0NsSSGpB9CM_2`, `_Ltqj_uJdGU_49`, `gaMvw1dd1jU_13`. **All four are already excluded** from `train/labels.csv`. |
| Borderline ORB (25–32 inliers) | 5 clips remain: `IMpbWnoCoK0_5`, `PoRnkaQbaTs_13`, `lg26XcIHGQ0_4`, `aDExXODkbxY_52`, `_2WIVACU8NE_2`. My independent frame-hash check found 0 matched frames for each; likely unrelated, but a 1-minute visual check is cheap. |
| Pool normals vs Snatch 1.0 normals | the pool leak check does not cover this; relies on segment-level dedup (see §4 caveat) |
| Within-pool near-duplicates inflating CV | harmless only if CV groups by upload; `finetune.py` does (`"POOL:" + stem.rsplit("_",1)[0]`) |
| Raw pool ↔ test | rule in `MAC_SETUP.md`: test clips never enter review kits; not re-verified by me for the raw set |

## 7. Domain shift

- Snatch 1.0: Hyderabad CCTV, low-res (median 360 p), short clips cut 1–3 s around the grab; normals are 1080p 25 fps.
- Web: India, Pakistan, Malaysia, Indonesia, UK, USA, Brazil, Vietnam; upload years 2014–2024; 720 p, ~1.7× sharper than Snatch 1.0 snatch clips; news edits.
- Median brightness and saturation are similar across sets (luma ~116–122, saturation ~40–46).
- **"Web = snatch" confound** *(README)*: most positives are web, most negatives are Snatch 1.0 (partly mitigated by per-source negative sampling and per-source thresholds). Normal-candidate density per hour also differs by source (Snatch 1.0 ≈ 4,200, web test ≈ 1,500, pool ≈ 940), i.e. the sources look different to the candidate generator.
- Documented results *(README, not re-measured)*: rules 0/25 on both Snatch 1.0 and Snatch-Web; verifier ~3/25 on unseen cameras at ~11 false alarms/h; the 90 % recall/precision target is not met.

## 8. Temporal sufficiency

Lead = `event_start_s`, tail = duration − `event_end_s`.

| Set | Events | Lead ≥ 5 s | Tail ≥ 2 s | Both | Median lead / tail |
|---|---|---|---|---|---|
| Web test | 25 | 25 | 22 | 22 (bike 13/15, foot 9/10) | 9.6 s / 5.4 s |
| Web pool | 91 | 37 | 72 | 33 (bike 21/64, foot 12/27) | 4.0 s / 4.5 s |
| Snatch 1.0 | 20 | 6 | 9 | 4 (bike 1/11, foot 3/9) | 2.0 s / 1.8 s |

Interpretation: the test set has enough context for a getaway-based rule or a 10 s evidence clip. Snatch 1.0 does **not** (clips median 6.6 s), which is consistent with the rule engine scoring 0/25 there. For the verifier (12–16-frame clip around a contact pair) the context is sufficient; for the 10-second evidence video and any "motion before/after" logic it is not, for most pool and Snatch 1.0 clips.

Coverage of required scenario types:

| Scenario | Real data? |
|---|---|
| Bike snatch | yes: 92 events |
| Foot snatch | yes: 49 events |
| Normal pedestrians | yes: 29 (S1 avi) + 11 + 56 (unverified) |
| Two-wheelers passing without a snatch | only implicitly in S1 normals; 11 test normals (6 explicit hard negatives); **not labelled in the pool** |
| Other confusing negatives (hug, handover, tap, phone call) | **synthetic only** (`sim/synthetic.py`); Snatch 1.0's 9 "other" rows are not characterised in this report |

## 9. Strengths, weaknesses, suitability

**Strengths**
- Clean, fully decodable media; consistent label schema; clear frozen test set with documented leak controls.
- Adds 116 in-scope events (91 pool + 25 test) to Snatch 1.0's 25, plus a test set with explicit hard negatives.
- Existing tooling for dedup, leak checks, packing and upload-grouped CV.

**Weaknesses**
- Pool negatives unverified; no hard-negative labelling in the pool.
- Few verifier positives, extreme imbalance, foot and night under-represented.
- Overlay/news shortcuts; resolution and sharpness mismatch with CCTV.
- Verifier package stale (61 of 147 pool clips) and non-portable; code version split.
- Mac disk space is too low for on-device processing (`device_bash` refused; up to 17 GB needed).

**Suitability**
- Learned verifier: **yes, conditionally** (pool for training, 36-clip set test-only).
- Evaluation: **yes** for the frozen 36 clips; Snatch 1.0 only as short-clip secondary.
- Detector/pose training: **no**.
- Raw 3.8 k downloads: **not usable** until segmented, deduplicated and reviewed.

## 10. Recommended preprocessing (not yet done; awaiting approval)

1. Visually verify or relabel the 56 pool normals (sample first; a snatch hidden in a negative costs more than a missing one).
2. Clamp or correct the 5 out-of-range event times.
3. Normalise lighting vocabulary (`night` ↔ `lowlight`) and fill lighting for normals.
4. Eyeball the 5 borderline leak clips.
5. Group by source upload everywhere; collapse the `-b` re-cuts and the `1zIVEDHkjJQ_2/_4` pair for counting.
6. Treat overlays: mask the lower-third/logo regions or augment with synthetic overlays on Snatch 1.0 negatives.
7. Repack the verifier package from the full 147-clip pool with portable relative paths, using the newer `trainpool.py` / `segment.py` / `discover.py`.
8. Reconcile the `dataset_1/snatchguard` code with the main project before any shared-pipeline work.
9. Harvest real hard negatives (two-wheelers passing, crowds, hand-offs) from the raw pool review decisions (`other` = 715 segments is a likely source).
10. Free ≥ 20 GB on the Mac (or move processing to the GPU box) before Stage 2+.
11. Keep the 36-clip test set frozen and never in review kits or training.

## 11. Limitations of this assessment

- Snatch 1.0 normal `.avi` files were not decoded (only `s_5.avi`; `s_2.avi` > 400 MB staging limit).
- The ~3.8 k raw downloads were counted from logs, not probed.
- `results_cmp/` not analysed.
- Performance figures are quoted from the README, not re-measured; no evaluation or training was run.
- Stolen-object counts are keyword matches on free-text notes.
