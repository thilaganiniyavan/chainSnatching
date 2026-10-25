# What SnatchGuard does, and the evidence for it

The project website, the README and any presentation state only what is in sections A and B, with the wording
and numbers given here. Items in C are said to be "built, not yet demonstrated" or are left out; items in D appear
only under a roadmap, marked as not done. When something moves (a test is run, a result comes back), this file
changes first.

Last checked: 2026-10-25.

## A. Measured on real footage

Every number is from real CCTV, at event level, with 95 % intervals (Wilson for proportions, exact Poisson for
rates). Snatch 1.0: 25 in-scope events, 2.0 h of held-out normal footage. Snatch-Web: 25 events from other cities,
0.35 h of normal footage, never used to tune the rules.

| Claim | Number | Evidence |
|---|---|---|
| Rules as deployed raise few false alarms on real streets | **0 CONFIRMED, 4 SUSPECT in 2.0 h** (2.0 per hour, 0.5-5.1) on 14 held-out videos: day junctions, colour-night streets, IR night markets | README "False alarms on all 14 held-out normal videos"; `python -m snatchguard.evaluation.replay --only normal` |
| Crowd-aware rules cut false alarms at night | **20 -> 1** in 1.4 h of real night CCTV | commit e43c7fa; website Table 6 |
| Rules detection rate | **0 / 25** Snatch 1.0, **0 / 25** Snatch-Web (0-13 %) | website Table 4; `evaluation.funnel`, actor audit |
| Why the rules miss | in 5 of the 7 decisions about the real victim and attacker, the attacker's track ends within 0.7 s after contact, so the getaway is never measured | website section 5.3 |
| Learned verifier (frozen network, CPU), trained on Snatch 1.0 only, threshold fixed in advance | **3 / 25** Snatch-Web events (4-30 %) at **11.4** false alarms per hour (3.1-29.3) | `runtime/verifier/report.json`; website Table 7 |
| Learned verifier ranks clips | clip AUC **0.90** (Snatch 1.0) and **0.80** (Snatch-Web) with whole cameras held out (5-fold); **0.77** fully external | `runtime/verifier/report.json` |
| Learned window scorer | 6 / 25 at 10.5 per hour (Snatch 1.0, out of fold); 6 / 25 at 45.7 per hour (Snatch-Web) | website Table 4 |
| Candidate-stage fix (2026-10-24) | events with a candidate between the labelled victim and attacker: **24 -> 28 of 50**; candidates per normal hour 3,240 -> 3,653 | README "Where events were lost"; `snatchguard/verifier/candidates.py` (`ABSENT_S`) |
| Laptop CPU throughput | **2-5** analysed frames per second on crowded scenes (2.7 on s_23), Ryzen 3 3200U; verifier **0.86 s** per clip | website section 5.5 |
| GPU fine-tuned verifier (R(2+1)D-18, Swin3D-T, six-view scoring, CONFIRMED / SUSPECT thresholds) | **pending** (`verifier_ft_results.zip`) | `docs/GPU_VERIFIER.md` |

## B. Built and tested

Automated tests (`pytest`) or an end-to-end run. These show the feature works as designed; they are not
measurements on real incidents and are never quoted as accuracy.

| Feature | Evidence |
|---|---|
| Live pipeline: camera (RTSP, video file, webcam, playlist) -> YOLO26n detection + OC-SORT tracking -> rule engine and learned verifier -> alert with annotated snapshot and video clip -> central dashboard | end-to-end run 2026-10-25: a real Snatch-Web clip, edge node, verifier and central server; alerts arrived with snapshot and clip (test model, so its scores mean nothing) |
| Explainable decisions: every rule alert lists its cues (contact, escape, reaction, approach) with their values | `tests/test_rules.py`; alert payloads |
| Two alert levels: CONFIRMED and SUSPECT | rules: `tests/test_rules.py`; verifier: `tests/test_verifier.py`, `tests/test_finetune.py` |
| Live dashboard: alerts as they happen, sound, snapshot and clip, operator review (confirm / false alarm), optional login | `tests/test_central.py` |
| Alerts are not lost when the network or the server is down: on-disk outbox, retries, no duplicates | `test_outbox_retry_and_persistence`, `test_event_is_idempotent_and_listed` |
| Camera offline: "NOT MONITORED" alert after 35 s without a heartbeat | `test_silent_node_raises_offline_alert` |
| Tamper alerts: covered lens, camera moved, frozen video; not triggered by the switch from day to night | `tests/test_edge_parts.py` (synthetic frames) |
| Lighting modes day / low light / IR, switched with hysteresis | `tests/test_edge_parts.py` |
| Tracks continued across tracker identity switches | `tests/test_scene.py` |

## C. Built, not yet demonstrated

| Feature | What is missing |
|---|---|
| Phone alerts on Telegram: snapshot, then the clip as a reply; retries; delay log | tested only against a simulated Telegram server (`tests/test_notify.py`); needs one test with a real bot and phone |
| Live system on the GPU laptop (`scripts/windows/run_live.ps1`) | not yet run; its frame rate and the delay from contact to alert are to be measured on the Snatch-Web playlist |
| Alert delay | measured once (CPU, 2026-10-25): 4.1-4.2 s after the contact; by design the clip needs 2.5 s after the contact (3.2 s with six views). Not a statistic yet |
| Several cameras on one edge node (one process each) and automatic restart of a crashed camera process | implemented (`snatchguard/edge/node.py`), not tested; no "N cameras per node" figure may be given |
| Blurred / sprayed lens tamper check | implemented (`snatchguard/edge/tamper.py`), no test |

## D. Not done: never claim

* The 90 % recall / 90 % precision target. It was not reached.
* Detection of real snatches at night. Night is measured only for false alarms; no real night snatch has been caught.
* Deployment on a Jetson or other embedded board, and any frame rate there (the README's 25-60 fps is an estimate).
* PostgreSQL (the server uses SQLite).
* A field trial, real incidents, police or control-room integration.
* Identifying people (faces) or vehicles (number plates). The system does neither.
