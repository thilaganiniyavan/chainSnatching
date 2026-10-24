# SnatchGuard

**Live, rule-based chain-snatching detection at the edge, with alerts to a central dashboard.**

An edge node sits next to existing CCTV cameras, watches the streams continuously, and when a
chain snatching happens it sends an alert (annotated snapshot, video clip and the evidence
behind the decision) to a central dashboard within a few seconds. It covers snatchers on a
motorcycle and on foot, works at night (IR and colour low-light cameras) and is built to tell a
snatch apart from friends playing, chatting, hugging or handing something over.

**Target:** event-level **recall ≥ 90 % and precision ≥ 90 %**, measured with cross-validation on the
labelled Snatch 1.0 dataset. The whole path from raw data to that measurement is one command on an
NVIDIA GPU: see [docs/WINDOWS_GPU_GUIDE.md](docs/WINDOWS_GPU_GUIDE.md).

The decision is **rule-based and explainable**: a small state machine over physical
measurements (distances, speeds, hand-to-neck contact, victim reaction, getaway). Machine
learning is only used for *perception* (finding and tracking people and bikes, estimating
body keypoints), where it is unavoidable.

```
 ┌──────────────────────────── EDGE NODE (laptop simulation / Jetson) ───────────────────────────┐
 │ camera (RTSP / file / webcam)                                                                  │
 │   └─► latest-frame reader ─► lighting mode (day / low-light / IR) ─► enhancement               │
 │         └─► YOLO26 detect + OC-SORT track ─► riders vs pedestrians ─► (victim, attacker) pairs │
 │               └─► YOLO26-pose (GPU: whole frame, CPU: closest pairs) ─► RULE ENGINE            │
 │               └─► contact clips ─► learned video verifier (optional, own process) ─► SUSPECT   │
 │                     └─► CONFIRMED / SUSPECT ─► snapshot + clip ─► SQLite outbox ─┐             │
 │   tamper checks (blackout, blur, moved, frozen) ────────────────────────────────┤             │
 │   supervisor: one process per camera, auto-restart, heartbeat + thumbnails ─────┤             │
 └──────────────────────────────────────────────────────────────────────────────────┼─────────────┘
                                                              HTTPS, API key, retries │
 ┌─────────────────────────────────────── CENTRAL ──────────────────────────────────▼─────────────┐
 │ FastAPI ─► SQLite (→ PostgreSQL) + media store ─► live dashboard (Server-Sent Events)          │
 │ watchdog: node silent > 35 s ⇒ "NOT MONITORED" alert       operator review: confirm / false alarm│
 └─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Quick start (laptop, no GPU)

```bash
python3 -m venv .venv            # if ensurepip is missing: python3 -m venv --without-pip .venv && <other-venv>/pip --python .venv/bin/python install pip
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.venv/bin/pip install -e ".[edge,central,dev]"
.venv/bin/python scripts/fetch_models.py          # YOLO26n + YOLO26n-pose → OpenVINO (fast on CPU)

# terminal 1: central server + dashboard at http://127.0.0.1:8000
.venv/bin/python -m snatchguard.central

# terminal 2: an edge node. Pick a camera:
.venv/bin/python -m snatchguard.edge --source synthetic:snatch                 # scripted scene, no models needed
.venv/bin/python -m snatchguard.edge --source path/to/video.mp4 --loop         # a recording, played in real time
.venv/bin/python -m snatchguard.edge --source path/to/video.mp4 --night ir     # ...turned into IR night footage
.venv/bin/python -m snatchguard.edge --source rtsp://user:pass@camera/stream   # a real camera
.venv/bin/python -m snatchguard.edge                                           # all cameras in configs/edge.yaml
```

Scripted scenes: `snatch`, `snatch_chase`, `snatch_night_ir`, `snatch_head_on`, `passing_bike`,
`friends_chat`, `playful_tap`, `handover`, `hug`, `walking_alongside`, `phone_call`.

Get the Snatch 1.0 videos:

```bash
.venv/bin/python scripts/snatch_dataset.py download --to ~/snatch_zips      # all 7 parts, resumable
.venv/bin/python scripts/snatch_dataset.py import --from /media/usb/snatch  # or: copied from another machine
.venv/bin/python scripts/snatch_dataset.py verify                           # 64/64 labelled videos present?
.venv/bin/python scripts/snatch_dataset.py get-all --category "Snatch Theft" --max-mb 20  # just the clips
```

## How a snatch is recognised

Everything is measured in **victim body-heights (H)** and **seconds**, so the same thresholds
work on any camera, resolution and frame rate (`configs/rules.yaml`).

**Who is who.** A person is a *rider* when their box sits on a bike and moves with it. A person
becomes a possible *victim* only after being seen **on foot**: clear of every bike for 1 s,
with an upright standing box (seated riders are squat), at walking or running speed. Riders
never become victims, and a victim is never re-labelled as a rider while the bike is
beside them.

**Per (victim, attacker) pair state machine:** `WATCH → CONTACT → decide → COOLDOWN`. The attacker
is a motorcycle/bicycle with its riders, or a person on foot.

| Evidence | Rule (all from geometry over time) | Weight |
|---|---|---|
| **Contact** | a rider's **wrist keypoint reaches the victim's neck** (pose), or the rider's box overlaps the victim's upper body at the same ground depth (box-only fallback, 60–70%) | 0.30 |
| **Escape** | the bike **speeds up** after the contact, or had slowed for the grab and then leaves fast, or leaves the frame | 0.25 |
| **Reaction** | victim's **hands go to the neck** (only if they weren't already there, e.g. a phone call), a fall (top of the box drops, feet stay down), a jolt, or running after the bike | 0.25 |
| **Approach** | gap closing, bike **slowing down** beside the victim, coming from behind | 0.20 |

**Playfulness and familiarity checks** subtract from the score:

| Indicator | Why it means "not a snatch" |
|---|---|
| stood together ≥ 5 s before the contact | people who know each other talk first |
| moved together ≥ 5 s | walking alongside a friend's bike |
| sustained contact ≥ 2 s | hugs and handovers are long; a grab is a fraction of a second |
| ≥ 3 separate contact encounters | playing around |
| bike stays near the victim afterwards | a snatcher never stays |
| victim waves / carries on calmly | no distress |

| 'victim' was among vehicles most of the 4 s before | a rider waiting in traffic, not a pedestrian |

**Decision.** **CONFIRMED** needs contact **and** escape **and** a corroborating cue (pose contact,
a victim reaction or the bike slowing for the grab), score ≥ 0.70, and no lingering.
**SUSPECT** (score ≥ 0.45) still needs an escape or a pose-level hand-at-neck contact, because
a snatcher always leaves. Everything else is dismissed and only logged locally. Every alert
carries the full evidence list, so the operator can see why it fired.

**On foot.** People brush past each other all the time in crowds, so an on-foot alert is stricter:
CONFIRMED needs the attacker's hand at the victim's neck (pose), a real run away (≥ 1.8 H/s) and a
victim reaction or approach; SUSPECT needs the run plus a hand-at-neck or a reaction. Box overlap
between two pedestrians never alerts on its own.

All thresholds in `configs/rules.yaml` are starting values: `evaluation.tune` fits them to labelled
footage with cross-validation and writes `configs/rules.tuned.yaml`.

**Night.** The lighting analyser switches between `day`, `lowlight` and `ir` (with hysteresis).
Each mode has its own enhancement (CLAHE / gamma) and rule profile (looser keypoint confidence,
larger neck radius, more weight on box contact when pose is unreliable). Alerts are tagged with
the mode and a visibility level (`high` = hand-at-neck seen, `medium`/`low` = boxes and motion only).

## Keeping the system itself reliable

| Failure | Handling |
|---|---|
| network or central down | alerts wait in an on-disk outbox and are retried with backoff; duplicates are impossible (edge-generated ids, idempotent insert) |
| camera stream drops | reconnect with backoff; status is visible on the dashboard |
| lens covered / sprayed / camera turned / video frozen | tamper alerts (reference learned per lighting mode, so night doesn't trip it) |
| camera process crashes | the supervisor restarts it with backoff; the restart count is shown |
| whole edge node dies | central raises **"NOT MONITORED"** after 35 s without a heartbeat |
| too slow to analyse reliably | the camera is flagged `low fps` below 5 fps (the tested operating envelope) |
| the same snatch seen by two pairs | one alert per attacker and per victim per cooldown |
| tracker restarts an object under a new id (occlusion, compression) | track stitching: a new id where a just-lost track was heading continues that track |

## Results so far

### Rules on scripted scenes (`python -m snatchguard.sim.run_scenarios`, `pytest`)
23 scenes (motorcycle and on-foot snatches, and their look-alikes) × 5 noise conditions (tracker jitter
up to 4 px, 8–30 % dropped frames):

| processing rate | wrong verdicts |
|---|---|
| 12 / 8 / 6 fps | **0** (no missed snatch, no false alarm) |
| 5 fps | 1 / 92 |
| ~2.8 fps effective (4 fps, 30 % dropped) | 4 / 92, **all towards weaker alerts**, never a false CONFIRMED |

Night IR without any pose (boxes only) is CONFIRMED for motorcycle snatches through the chase and approach
evidence (SUSPECT on foot, where pose is required). The playful tap, friends chatting or hugging, handover,
walking alongside, phone call, jogger, crowd brush and passing bikes are dismissed.

### Real CCTV (Snatch 1.0, pretrained YOLO26n, laptop CPU, before training and tuning)
All numbers are with the **pretrained** YOLO26n (before fine-tuning), 512 px, 8 fps analysis,
on this laptop's CPU (about 2–5 fps achieved on crowded scenes).

**False alarms on real footage** (`python -m snatchguard.evaluation.benchmark --only normal --max-seconds 600`, plus `analyse s_9.avi`):

| footage | length | close encounters judged | false CONFIRMED | false SUSPECT |
|---|---|---|---|---|
| busy Hyderabad junction, day (`s_2`, `s_3`) | 20 min | ~65 | **0** | 1 (≈3 / h) |
| crowded night market, shop lighting (`s_9`) | 8.3 min | 77 | **0** | 2 (≈14 / h) |

Every false alarm found during development was traced to a cause and fixed with a general rule (riders whose
bike was missed, riders queued at a signal, a rider's own scooter detected as a second object, tracks lost at
the frame edge in slow crowds), each with a regression test.

**False alarms on all 14 held-out normal videos** (first 10 min of each = 2.0 h: day junctions, colour-night
streets and real IR night markets; laptop perception, default rules, `evaluation.replay --only normal`):
**0 CONFIRMED, 4 SUSPECT (2.0 / h)**, down from 20 in the first 1.2 h before the crowd-aware rules. With about 25
real events, 90 % precision needs under ~1 false alarm per hour, so the GPU run's tuning has to trim the
remaining SUSPECT margin (it optimises exactly that).

**Cross-validated acceptance run on this laptop** (pretrained YOLO26n, 512 px, CPU; 25 snatch events, 20 min of
held-out normal footage, `evaluation.tune --trials 60`): recall **0 / 25**, no false alarms. On-foot clips reach
contact in 5 of 12 but never show the getaway; motorcycle clips mostly fail at detection. This is the "before"
column; the GPU pipeline produces the "after".

**Recall on the 13 hand-labelled motorcycle-snatch clips** (`data/labels/snatch1.0.csv`): **0 / 13** so far.
`python -m snatchguard.evaluation.funnel` shows why: the pretrained detector never finds the motorcycle in a
large share of these small, blurry, re-encoded clips (even YOLO26x at 1280 px missed a clearly visible
two-rider bike), and most clips are cut 1–3 s around the grab, so there's no context before it and no getaway
after it. This is the perception gap the GPU training targets (VisDrone elevated views, CCTV
degradation, night variants, pseudo-labels from real Hyderabad CCTV). The acceptance report produced by
`training.run_all` (with `--baseline`) gives the before/after comparison on the full dataset.

Of the 35 Snatch 1.0 clips, 13 are motorcycle snatches and 12 on-foot snatches (both in scope), and 9 are
staged videos, a movie scene or screen recordings (out of scope).

## Learned verifier (experimental, off by default)

The rules miss nearly every real snatch: 0 / 25 on Snatch 1.0 and 0 / 25 on the Snatch-Web set (25 snatch
and 11 normal clips from other cities and cameras, `scripts/snatch_dataset.py`; not redistributed) at the
deployed setting. Where the rules do pick the right pair, the snatcher's track usually ends within a second
of the grab (5 of 7), so the getaway the rules wait for is never seen. The verifier is a second, learned path
beside the rules (`snatchguard/verifier/`):

1. **Candidates** (`candidates.py`): each new close contact between two tracked people (or a person and a
   rider) opens a clip. Contacts in the same place at the same moment share one clip. No roles are guessed.
2. **Verification** (`model.py`): 12 frames from 1.5 s before to 2.5 s after the contact, cropped around the
   pair at 112 px, go through a frozen Kinetics-400 R(2+1)D-18 (OpenVINO on CPU). A small head (standardise →
   PCA 32 → logistic regression) turns the 512-d embedding into a probability.
3. **Live** (`live.py`, `serve.py`): clips are scored in a separate process at the lowest CPU priority, newest
   first. A score over the threshold becomes a SUSPECT alert with snapshot and clip, marked "learned verifier"
   on the dashboard.

**Labels name the actors** (`data/labels/actors.json`): for each timed event, the track ids of the victim and
the snatcher. A clip is positive only if it contains that pair inside the event window; bystanders at the
same moment are negatives. Events where the pair cannot be told apart are left out (`unclear`, 6). Events
where the detector never tracks one of the two count as misses (`untracked`, 7).

```bash
python scripts/fetch_models.py --verifier        # Kinetics-400 R(2+1)D-18 weights
python -m snatchguard.verifier.train features    # candidate clips -> embeddings (~50 min on this laptop's CPU)
python -m snatchguard.verifier.train fit         # held-out evaluation + final model + threshold
```
Then set `verifier.enabled: true` in `configs/edge.yaml`.

**Held-out results** (`runtime/verifier/report.json`). Counted over the events whose actors can be
identified: 16 of the 25 Snatch 1.0 events and 23 of the 25 Snatch-Web events. False alarms are per hour of
normal footage.

| protocol | clip AUC | caught at ≤ 10 false alarms / h | caught at ≤ 100 / h |
|---|---|---|---|
| train Snatch 1.0 → test Snatch-Web | 0.77 | 3 / 23 (5.7 / h) | 5 / 23 |
| train Snatch-Web → test Snatch 1.0 | 0.72 | 0 / 16 | 2 / 16 |
| pooled 5-fold, whole cameras held out: Snatch 1.0 | 0.90 | 3 / 16 (none in a 14 % sample, 95 % bound ≈ 10 / h) | 4 / 16 |
| pooled 5-fold, whole cameras held out: Snatch-Web | 0.80 | 4 / 23 (8.6 / h) | 10 / 23 |

The verifier trains on Snatch-Web too: pooled cross-validation holds out whole uploads, and the final model
uses all of it. So for the verifier, Snatch-Web is no longer the untouched external test it is for the rules,
and the first row is its only fully external number.

This beats the rules, which catch none, but it is far from the 90 % target. On held-out cameras it catches
about one snatch in five at a tolerable false-alarm rate, and it does not yet transfer from one collection to
the other. The limit is labelled events (39 usable), not training time. The stored threshold (0.983) is
the pooled cross-validation threshold for ≤ 10 false alarms per hour.

**Live on this laptop** (Ryzen 3 3200U, 2 cores / 4 threads). The test scene is a crowded Snatch 1.0 normal
video (`s_23`, 60 s, real time); frame rates are medians of readings every 2 s:

| verifier | rules' analysed frames / s | candidates scored | dropped (waited > 20 s) | false alarms |
|---|---|---|---|---|
| off | 2.7 | – | – | 0 |
| own process, lowest priority (default) | 2.3 | 18 of 30 | 12 | 0 |
| inside the camera process | 1.3 | 27 of 29 | 2 | 1 (a shopkeeper and a customer at a stall) |

- The network takes 0.86 s per clip (OpenVINO; 1.25 s in PyTorch), so on this CPU it cannot keep up with a busy
  street.
- Below about 3 analysed frames per second, frames repeat within a clip. The clips it was trained on are cut
  from the full-rate video, so live scores here are not the offline scores.
- In a playlist check (normal footage, then four snatch clips), SUSPECT alerts reached the central server with
  snapshot and clip. Those clips were in the training set, so this shows the live path works, not how accurate it is.
- On a GPU the network should take well under 0.1 s per clip. That is not measured on this project's hardware
  yet.

### More training events and GPU fine-tuning

The binding limit above is labelled events, so a **training pool** of web footage
(`snatchguard/webdata/trainpool.py`) adds events from the downloads the clean test set did not take: unreviewed
CCTV-like shots, snatches rejected from the test set for news graphics or a short lead-in, and new searches
(Vietnam, Thailand, Malaysia, Indonesia, Latin America, more Indian languages, CCTV compilations). It is kept
apart from both test sets: every upload that gave a Snatch-Web test clip is excluded whole, footage repeated
from Snatch 1.0 or another upload is dropped by `dedup.py`, and pool events are never scored as test events.
Each event is reviewed by eye (`review/train_decisions_snatch.csv`: decision, lighting, grab time, note);
slow-motion replays, zooms during the grab, picture-in-picture, phones filming monitors, hold-ups and vehicle
thefts are left out. Pool events have grab times but no actor ids, so their candidates inside the event window
form a bag of which at least one must score high (multiple-instance learning).

With a GPU the whole network can be fine-tuned instead of only the head (`snatchguard/verifier/finetune.py`).
Detection, tracking, candidates and labels run once on the development laptop and are written as a package of
JPEG clip strips, so the GPU machine needs only PyTorch: **[docs/GPU_VERIFIER.md](docs/GPU_VERIFIER.md)**.

```bash
python -m snatchguard.webdata.trainpool sheets             # review images for unreviewed shots
python -m snatchguard.webdata.trainpool build && python -m snatchguard.webdata.trainpool cache
python -m snatchguard.verifier.finetune pack --out ~/verifier_data        # development laptop
python -m snatchguard.verifier.finetune run --data ~/verifier_data        # GPU machine
```

Snatch 1.0 and web footage look different, and most positives are web clips while most normal footage is
Snatch 1.0. Trained on that mix as it is, the head-only verifier learned "web footage = snatch": held-out web
normal footage scored 0.94 (median), and on the Snatch-Web test set a threshold fixed on mostly Snatch 1.0
validation footage gave 77 false alarms per hour instead of 10 (9 of 21 events caught). So each source's
negatives are now drawn as often as its share of the positives, validation footage is held out from each
source, and the threshold must keep each source at or under the budget.

Every protocol fixes its threshold on held-out validation footage (10 false alarms per hour) before its test set
is scored: `external` trains on Snatch 1.0 + pool and tests the frozen Snatch-Web test set, once fine-tuned and
once with the network frozen (a linear probe on the same data), so the gain from fine-tuning is measured
separately from the gain from more data; `s1-cv` holds out Snatch 1.0 cameras in 5 folds; `final` is the model
to deploy. A fine-tuned model loads like the head-only one (`verifier.model` in `configs/edge.yaml`) and runs
through OpenVINO on a CPU at the same cost per clip.

**Where events were lost, and what changed (2026-10-24).** Of the 50 test events (25 Snatch 1.0, 25 Snatch-Web),
only 24 reached the verifier as a candidate between the labelled victim and attacker; the rest were lost before
any classifier: 7 actors never tracked, 6 unresolvable at 512 px, 5 untimed, and 8 tracked events with no
victim-attacker candidate. In 3 of those 8 a pair stayed "in contact" while the detector lost one of them, so the
real grab, seconds later, continued the old contact instead of starting a new one; a pair not seen together for
0.5 s (`ABSENT_S`) is now out of contact. On the cached footage this gives 28 of 50 events a correct candidate
(Snatch 1.0 6 -> 9 of 11 tracked events, Snatch-Web 18 -> 19 of 21) for 13 % more candidates on normal footage
(3,240 -> 3,653 per hour). The package was rebuilt with it. The other losses need a better detector.

The GPU run now trains two pretrained networks under the same protocols, R(2+1)D-18 (Kinetics-400 top-1 67.5 %)
and Swin3D-T (77.7 %; its window attention takes the same 12 x 112 x 112 clip), and scores every clip both alone
and as the mean of six views (shifted 0.73 s either way, mirrored), each scoring with its own thresholds fixed on
validation footage: SUSPECT at 10 and CONFIRMED at 1 false alarm per hour. `finetune select` picks the network
and scoring to deploy from the Snatch 1.0 cross-validation only, so the Snatch-Web result stays held out for every
candidate. Live, alerts go to the dashboard and, with a Telegram bot configured, to phones with the snapshot and
then the evidence clip (`snatchguard/central/notify.py`), logging the delay from contact to phone.

## Training and the 90 % acceptance test (NVIDIA GPU)

Step-by-step for Windows 11 + RTX 4060: **[docs/WINDOWS_GPU_GUIDE.md](docs/WINDOWS_GPU_GUIDE.md)**. In short:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1
.\.venv\Scripts\python.exe scripts\snatch_dataset.py download --to D:\snatch_zips
powershell -ExecutionPolicy Bypass -File scripts\windows\train_all.ps1 -Zips D:\snatch_zips -Baseline
```
(Linux: `python -m training.run_all --zips /path/to/zips --baseline`.)

`training.run_all` builds the night/CCTV training sets (ExDark, LLVIP, VisDrone, COCO + synthetic IR,
low-light and CCTV degradation), pseudo-labels the `pseudo` fold of real Hyderabad CCTV with a YOLO26x
teacher, fine-tunes YOLO26s detect + pose, caches perception over every evaluation video, tunes the rule
thresholds with 2-fold cross-validation, checks the same held-out footage as simulated IR and low-light
night video, and writes `runtime/tuning/acceptance_report.md`: recall, precision, 95 % confidence
intervals, false alarms per hour, night results and every clip's outcome, with PASS/NOT MET against
90 %/90 %. Every step is resumable. `--profile smoke` runs the whole chain on any CPU in
minutes to check the plumbing.

**Splits** (`data/labels/snatch1.0.csv`, column `fold`): 15 normal videos (`pseudo`) are only ever used to
teach the detector; 14 normal videos (3.1 h) and the 25 in-scope snatch clips are split into folds A and B
for rule tuning and scoring. Nothing is scored on data it was fitted to.

## Evaluation tools

```bash
python -m snatchguard.evaluation.cache --config configs/edge.gpu.yaml  # perception once per video
python -m snatchguard.evaluation.tune --config configs/edge.gpu.yaml   # cross-validated tuning
python -m snatchguard.evaluation.acceptance                            # the 90 %/90 % report
python -m snatchguard.evaluation.funnel                         # where each clip is lost (detection → alert)
python -m snatchguard.sim.run_scenarios                        # rule verdicts on scripted scenes
python -m snatchguard.evaluation.analyse VIDEO --debug out.mp4 # what the pipeline sees, frame by frame
python -m snatchguard.evaluation.analyse VIDEO --zone 0 0 .7 .7  # detection zone = zoom for distant cameras
python -m snatchguard.evaluation.benchmark --only snatch_bike  # recall on labelled snatch clips
python -m snatchguard.evaluation.benchmark --only normal       # false alarms per hour on normal footage
python -m snatchguard.evaluation.benchmark --night ir          # the same, with synthetic IR night
```
Labels live in `data/labels/snatch1.0.csv` (category, event start/end times, fold); every snatch clip
was reviewed frame by frame. `scripts/label_events.py` adds or corrects times for new footage.

## The two-machine demo (GPU edge box + laptop dashboard)

```bash
# laptop
SNATCHGUARD_API_KEYS=my-edge-key SNATCHGUARD_DASHBOARD_PASSWORD=choose-one scripts/laptop/run_central.sh
```
```powershell
# GPU box (Windows)
powershell -ExecutionPolicy Bypass -File scripts\windows\run_edge.ps1 -Central http://<laptop-ip>:8000 -ApiKey my-edge-key
```
The GPU box's camera defaults to `configs/playlists/demo.txt`: held-out normal Hyderabad CCTV with real
snatch clips in between, played in real time as one live camera (`playlist:` sources reset tracking
at every cut). Use `-Source rtsp://...` for a real camera.

## Moving to real edge hardware

The code is hardware-agnostic: only the model files change. On an NVIDIA Jetson (Orin Nano Super or
better) export on the device with `yolo export model=best.pt format=engine half=True` and point
`configs/edge.yaml` at the `.engine` files; expect 25–60 fps per model instead of about 7 on this laptop,
which allows the `s` models, 640–960 px input and 2–4 cameras per box. Point `central.url` at the
central server (HTTPS in production) and give every node its own API key
(`SNATCHGUARD_API_KEYS` on central).

## Layout

```
snatchguard/
  core/        types, geometry, config loading
  rules/       scene model (roles), pair measurements, evidence evaluation, rule engine
  edge/        source, lighting, perception, pipeline, evidence, tamper, outbox, publisher, camera worker, node
  central/     FastAPI app, storage, dashboard (static/)
  sim/         scripted scenes, night simulation, scenario runner
  evaluation/  analyse, benchmark, funnel, perception cache, metrics, cross-validated tuner, acceptance report
  verifier/    learned verifier: contact candidates, video model, live stage + its process, training, GPU fine-tuning
  webdata/     web footage: search, download, shots, de-duplication, review, clean test set, training pool
configs/       edge.yaml (laptop), rules.yaml, tracker.yaml, playlists/ (generated: edge.gpu.yaml, rules.tuned.yaml)
data/labels/   snatch1.0.csv (events, folds), actors.json (victim / snatcher track ids per event)
training/      run_all (one-command pipeline), dataset builder, pseudo-labelling, train/export, Colab notebook
scripts/       windows/ (setup, train_all, train_verifier, run_edge), train_verifier.sh, laptop/run_central.sh,
               Snatch 1.0 tool, model fetch
docs/          WINDOWS_GPU_GUIDE.md, GPU_VERIFIER.md
tests/         rules (scripted scenes), scene, edge parts, central API, tuning pipeline, verifier, fine-tuning
```

## Data and licences

Snatch 1.0 (Roy et al., Pattern Recognition Letters 2018): CC-BY-SA 4.0 · ExDark: BSD-3, authors ask
for contact before commercial use · LLVIP: non-commercial · VisDrone: academic use · COCO: CC-BY 4.0 ·
torchvision R(2+1)D-18 (verifier trunk): BSD-3 code, Kinetics-400 weights for research use ·
Ultralytics YOLO: AGPL-3.0 (fine for an open or academic project; commercial deployment needs an
Ultralytics licence or a swap to an Apache-2.0 detector such as RF-DETR).
