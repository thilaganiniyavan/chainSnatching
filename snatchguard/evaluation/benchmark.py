"""Benchmark the live pipeline on a labelled video set.

Labels CSV columns: video, category, lighting, event_start_s, event_end_s, notes
  category: snatch_bike (in scope) | snatch_foot | other (out of scope) | normal (no snatch)
  event_start_s / event_end_s: optional; if given, an alert only counts when it falls in
  [start - 1, end + 6] s; otherwise any alert in the clip counts (clip-level).

Reports, per operating point:
  * recall on in-scope snatch clips (CONFIRMED only, and CONFIRMED or SUSPECT)
  * alert latency after the contact
  * false alarms per hour on normal footage (the number that decides if a live system is usable)
  * the same with synthetic IR / low-light applied (--night ir|lowlight) to test night robustness

    python -m snatchguard.evaluation.benchmark --only snatch_bike
    python -m snatchguard.evaluation.benchmark --only normal --max-seconds 600
    python -m snatchguard.evaluation.benchmark --night ir
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

from snatchguard.core.config import REPO_ROOT, load_yaml
from snatchguard.evaluation.analyse import analyse_video

DATA = REPO_ROOT / "data" / "samples" / "snatch1.0" / "Chain Snatching Videos"


def load_labels(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def in_window(alert: dict, row: dict) -> bool:
    if not row.get("event_start_s"):
        return True
    t = alert["video_time_s"]
    return float(row["event_start_s"]) - 1.0 <= t <= float(row["event_end_s"] or row["event_start_s"]) + 6.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(REPO_ROOT / "data" / "labels" / "snatch1.0.csv"))
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "edge.yaml"))
    ap.add_argument("--only", help="category to run (snatch_bike, normal, ...)")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--night", choices=["none", "ir", "lowlight"], default="none")
    ap.add_argument("--max-seconds", type=float, help="cap per video (long normal videos)")
    ap.add_argument("--out", default=str(REPO_ROOT / "runtime" / "benchmark"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cfg = load_yaml(args.config)
    rows = [r for r in load_labels(Path(args.labels)) if not args.only or r["category"] == args.only]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.only or 'all'}_{args.night}_{int(time.time())}"
    results = []
    for row in rows:
        path = Path(args.data) / row["video"]
        if not path.exists():
            print(f"missing {path}, skipped")
            continue
        res = analyse_video(str(path), cfg, args.fps, args.night, "auto", max_seconds=args.max_seconds,
                            progress=row["category"] == "normal")
        confirmed = [a for a in res.alerts(("CONFIRMED",)) if in_window(a, row)]
        any_alert = [a for a in res.alerts() if in_window(a, row)]
        rec = {
            "video": row["video"], "category": row["category"], "duration_s": round(res.duration_s, 1),
            "proc_fps": round(res.proc_fps, 1), "encounters": len(res.decisions),
            "confirmed": len(res.alerts(("CONFIRMED",))), "suspect": len(res.alerts(("SUSPECT",))),
            "hit_confirmed": bool(confirmed), "hit_any": bool(any_alert),
            "latency_s": min((a["latency_s"] for a in any_alert), default=None),
            "alerts": [{k: a[k] for k in ("video_time_s", "level", "score", "confidence", "penalties")}
                       | {"evidence": [e["key"] for e in a["evidence"] if e["passed"]]} for a in res.alerts()],
        }
        results.append(rec)
        mark = "HIT " if rec["hit_confirmed"] else ("susp" if rec["hit_any"] else "----")
        print(f"{mark} {row['category']:11s} {row['video']:28s} {rec['duration_s']:7.1f}s  "
              f"enc={rec['encounters']:2d} C={rec['confirmed']} S={rec['suspect']}  {rec['proc_fps']} fps", flush=True)
        (out_dir / f"{tag}.json").write_text(json.dumps(results, indent=1))

    print("\n=== summary", tag)
    for cat in sorted({r["category"] for r in results}):
        rs = [r for r in results if r["category"] == cat]
        hours = sum(r["duration_s"] for r in rs) / 3600
        if cat == "normal":
            fa_c = sum(r["confirmed"] for r in rs)
            fa_s = sum(r["suspect"] for r in rs)
            print(f"{cat:11s} {hours * 60:6.1f} min: false CONFIRMED {fa_c} ({fa_c / hours:.2f}/h), "
                  f"SUSPECT {fa_s} ({fa_s / hours:.2f}/h)")
        else:
            hc = sum(r["hit_confirmed"] for r in rs)
            ha = sum(r["hit_any"] for r in rs)
            lats = [r["latency_s"] for r in rs if r["latency_s"] is not None]
            print(f"{cat:11s} {len(rs):3d} clips: CONFIRMED {hc}/{len(rs)}, CONFIRMED+SUSPECT {ha}/{len(rs)}"
                  + (f", median latency {sorted(lats)[len(lats) // 2]:.1f}s" if lats else ""))
    print(f"details: {out_dir / (tag + '.json')}")


if __name__ == "__main__":
    main()
