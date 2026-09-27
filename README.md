# SnatchGuard

**Live, rule-based chain-snatching detection at the edge, with alerts to a central dashboard.**

An edge node sits next to existing CCTV cameras, watches the streams continuously, and when a
chain snatching happens it sends an alert (annotated snapshot, video clip and the evidence
behind the decision) to a central dashboard within a few seconds. It works at night (IR and
colour low-light cameras) and is built to tell a snatch apart from friends playing,
chatting, hugging or handing something over.

The decision is **rule-based and explainable**: a small state machine over physical
measurements (distances, speeds, hand-to-neck contact, victim reaction, getaway). Machine
learning is only used for *perception* (finding and tracking people and bikes, estimating
body keypoints), where it is unavoidable.

```
 ┌──────────────────────────── EDGE NODE (laptop simulation / Jetson) ───────────────────────────┐
 │ camera (RTSP / file / webcam)                                                                  │
 │   └─► latest-frame reader ─► lighting mode (day / low-light / IR) ─► enhancement               │
 │         └─► YOLO26 detect + OC-SORT track ─► scene: riders vs pedestrians ─► (victim, bike) pairs│
 │               └─► YOLO26-pose on the closest pairs only ─► RULE ENGINE (per-pair state machine) │
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

Get the Snatch 1.0 test videos (only the files you need, via HTTP range requests):

```bash
.venv/bin/python scripts/snatch_dataset.py list
.venv/bin/python scripts/snatch_dataset.py get-all --category "Snatch Theft" --max 60 --max-mb 20
```

## How a snatch is recognised

Everything is measured in **victim body-heights (H)** and **seconds**, so the same thresholds
work on any camera, resolution and frame rate (`configs/rules.yaml`).

**Who is who.** A person is a *rider* when their box sits on a bike and moves with it. A person
becomes a possible *victim* only after being seen **on foot**: clear of every bike for 1 s,
with an upright standing box (seated riders are squat), at walking or running speed. Riders
never become victims, and a victim is never re-labelled as a rider while the bike is
beside them.

**Per (victim, bike) pair state machine:** `WATCH → CONTACT → decide → COOLDOWN`

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

**Decision.** **CONFIRMED** needs contact **and** escape **and** a corroborating cue (pose contact,
a victim reaction or the bike slowing for the grab), score ≥ 0.70, and no lingering.
**SUSPECT** (score ≥ 0.45) still needs an escape or a pose-level hand-at-neck contact, because
a snatcher always leaves. Everything else is dismissed and only logged locally. Every alert
carries the full evidence list, so the operator can see why it fired.

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
| the same snatch seen by two pairs | one alert per bike per cooldown |

## Results so far

### Rules on scripted scenes (`python -m snatchguard.sim.run_scenarios`, `pytest`)
15 scenes × 5 noise conditions (tracker jitter up to 4 px, 8–30% dropped frames):

| processing rate | wrong verdicts |
|---|---|
| 12 / 8 / 6 / 5 fps | **0** (no missed snatch, no false alarm) |
| ~2.8 fps effective (4 fps, 30% dropped) | ~10%, **all towards weaker alerts** (CONFIRMED → SUSPECT), never a false CONFIRMED |

Night IR without any pose (boxes only) is CONFIRMED through the chase and approach evidence.
The playful tap, friends chatting, handover, hug, walking alongside, phone call and passing bikes are dismissed.

### Real CCTV (Snatch 1.0, pretrained YOLO26n, laptop CPU)
All numbers are with the **pretrained** YOLO26n (before the Colab fine-tuning), 512 px, 8 fps analysis,
on this laptop's CPU (about 2–5 fps achieved on crowded scenes).

**False alarms on real footage** (`python -m snatchguard.evaluation.benchmark --only normal --max-seconds 600`, plus `analyse s_9.avi`):

| footage | length | close encounters judged | false CONFIRMED | false SUSPECT |
|---|---|---|---|---|
| busy Hyderabad junction, day (`s_2`, `s_3`) | 20 min | ~65 | **0** | 1 (≈3 / h) |
| crowded night market, shop lighting (`s_9`) | 8.3 min | 77 | **0** | 2 (≈14 / h) |

Every false alarm found during development was traced to a cause and fixed with a general rule (riders whose
bike was missed, riders queued at a signal, a rider's own scooter detected as a second object, tracks lost at
the frame edge in slow crowds), each with a regression test.

**Recall on the 13 hand-labelled motorcycle-snatch clips** (`data/labels/snatch1.0.csv`): **0 / 13** so far.
`python -m snatchguard.evaluation.funnel` shows why: the pretrained detector never finds the motorcycle in a
large share of these small, blurry, re-encoded clips (even YOLO26x at 1280 px missed a clearly visible
two-rider bike), and most clips are cut 1–3 s around the grab, so there's no context before it and no getaway
after it. This is the perception gap the Colab training targets (VisDrone elevated views, CCTV
degradation, night variants, pseudo-labels from real Hyderabad CCTV). **Re-run the benchmark after
installing the fine-tuned models: that before/after comparison is the key result.**

Of the 35 Snatch 1.0 clips, 13 are motorcycle snatches (the system's target), 12 are on-foot snatches (a
possible future "loiter then lunge" rule, e.g. `10_0`), and 9 are staged videos, a movie scene or screen
recordings (out of scope).

## Training (Google Colab)

Open `training/colab/SnatchGuard_Train.ipynb` in Colab (T4 GPU) and run all cells. It:
1. builds a night-robust set: **ExDark** (real low-light), **LLVIP** (real infrared), **VisDrone**
   (elevated street views with riders and motorcycles) and **COCO**, with synthetic IR / low-light
   variants and CCTV degradation (blur, heavy JPEG, low resolution);
2. optionally **pseudo-labels your own CCTV** with a large teacher (YOLO26x, 1280 px) so the small
   edge model learns your cameras (the biggest single gain);
3. fine-tunes YOLO26n detect + pose, **auto-resuming** from Google Drive after disconnects;
4. reports day vs night mAP, pretrained vs fine-tuned;
5. exports OpenVINO / ONNX. Then locally: `python scripts/fetch_models.py --from-colab det.pt pose.pt`.

## Evaluation tools

```bash
python -m snatchguard.sim.run_scenarios                        # rule verdicts on scripted scenes
python -m snatchguard.evaluation.analyse VIDEO --debug out.mp4 # what the pipeline sees, frame by frame
python -m snatchguard.evaluation.analyse VIDEO --zone 0 0 .7 .7  # detection zone = zoom for distant cameras
python -m snatchguard.evaluation.benchmark --only snatch_bike  # recall on labelled snatch clips
python -m snatchguard.evaluation.benchmark --only normal       # false alarms per hour on normal footage
python -m snatchguard.evaluation.benchmark --night ir          # the same, with synthetic IR night
```
Labels live in `data/labels/snatch1.0.csv` (category per clip; add `event_start_s` / `event_end_s`
for time-accurate scoring).

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
  evaluation/  offline analyse + benchmark
configs/       edge.yaml, rules.yaml, tracker.yaml
training/      dataset builder, pseudo-labelling, training/export, Colab notebook
scripts/       model fetch/export, Snatch 1.0 downloader, contact sheets
tests/         rules, scene, edge parts, central API
```

## Data and licences

Snatch 1.0 (Roy et al., Pattern Recognition Letters 2018): CC-BY-SA 4.0 · ExDark: BSD-3, authors ask
for contact before commercial use · LLVIP: non-commercial · VisDrone: academic use · COCO: CC-BY 4.0 ·
Ultralytics YOLO: AGPL-3.0 (fine for an open or academic project; commercial deployment needs an
Ultralytics licence or a swap to an Apache-2.0 detector such as RF-DETR).
