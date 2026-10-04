"""List every alert the rules raise on cached videos (seconds, no models needed).

Use it to inspect false alarms on normal footage or misses on snatch clips after changing
`configs/rules.yaml`, before re-running the full tuning.

    python -m snatchguard.evaluation.replay --only normal
    python -m snatchguard.evaluation.replay --only snatch_bike --all      # include dismissed encounters
    python -m snatchguard.evaluation.replay --config configs/edge.gpu.yaml --rules configs/rules.tuned.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from snatchguard.core.config import DEFAULT_RULES, REPO_ROOT, RuleConfig, load_yaml
from snatchguard.evaluation import cache as cache_mod
from snatchguard.evaluation.benchmark import DATA, load_labels
from snatchguard.evaluation.metrics import event_window, overlaps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "edge.yaml"))
    ap.add_argument("--rules", default=str(DEFAULT_RULES))
    ap.add_argument("--labels", default=str(REPO_ROOT / "data" / "labels" / "snatch1.0.csv"))
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--only", help="category: snatch_bike | snatch_foot | normal")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--night", default="none")
    ap.add_argument("--all", action="store_true", help="also show dismissed encounters")
    args = ap.parse_args()

    rules = RuleConfig.load(args.rules)
    tag = cache_mod.perception_tag(load_yaml(args.config), args.fps, args.night)
    n_alerts = n_videos = 0
    hours = 0.0
    for row in load_labels(Path(args.labels)):
        if args.only and row["category"] != args.only:
            continue
        path = cache_mod.cache_path(str(Path(args.data) / row["video"]), tag)
        if not path.exists():
            continue
        cv = cache_mod.load(path)
        n_videos += 1
        hours += cv.duration_s / 3600
        window = event_window(row, cv.duration_s) if row["category"] != "normal" else None
        for d in cache_mod.replay(cv, rules):
            if d["level"] == "DISMISSED" and not args.all:
                continue
            n_alerts += d["level"] != "DISMISSED"
            verdict = "" if window is None else ("TRUE " if overlaps(d, window) else "WRONG-TIME ")
            ev = [e["key"] for e in d["evidence"] if e["passed"] and e["group"] != "benign"]
            pen = list(d["penalties"])
            print(f"{row['video']:26s} t={d['contact_start']:7.1f}s {verdict}{d['level']:9s} {d.get('mode', ''):10s} "
                  f"score={d['score']:.2f} vis={d['confidence']:6s} victim={d['victim_id']} attacker={d['vehicle_id']} "
                  f"{ev} {'penalties=' + str(pen) if pen else ''}")
    print(f"\n{n_alerts} alerts on {n_videos} cached videos ({hours:.2f} h)")


if __name__ == "__main__":
    main()
