"""External test: score a separate labelled dataset (e.g. ~/snatch_web_dataset) with fixed rules.

Nothing is tuned on it: thresholds come from Snatch 1.0 (configs/rules.tuned.yaml). This measures how
well the system generalises to footage from other sources, cameras and cities.

    python -m snatchguard.evaluation.external --config configs/edge.gpu.yaml \\
        --labels ~/snatch_web_dataset/labels.csv --data ~/snatch_web_dataset/clips --rules configs/rules.tuned.yaml
Writes runtime/tuning/external.json and appends nothing else; acceptance.py shows it when present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from snatchguard.core.config import REPO_ROOT, RuleConfig, load_yaml
from snatchguard.evaluation import cache as cache_mod
from snatchguard.evaluation.benchmark import load_labels
from snatchguard.evaluation.metrics import IN_SCOPE, score

WEB = Path.home() / "snatch_web_dataset"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "edge.yaml"))
    ap.add_argument("--rules", default=str(REPO_ROOT / "configs" / "rules.tuned.yaml"))
    ap.add_argument("--labels", default=str(WEB / "labels.csv"))
    ap.add_argument("--data", default=str(WEB / "clips"))
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--level", choices=["any", "confirmed"], default="any")
    ap.add_argument("--name", default="Snatch-Web")
    ap.add_argument("--out", default=str(REPO_ROOT / "runtime" / "tuning" / "external.json"))
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    rules_path = Path(args.rules) if Path(args.rules).exists() else REPO_ROOT / "configs" / "rules.yaml"
    rules = RuleConfig.load(rules_path)
    results = {}
    for row in load_labels(Path(args.labels)):
        if row["category"] not in IN_SCOPE and row["category"] != "normal":
            continue
        path = Path(args.data) / row["video"]
        if not path.exists():
            print(f"missing {path}")
            continue
        cv = cache_mod.get(str(path), cfg, args.fps, progress=False)
        results[row["video"]] = (row, cv.duration_s, cache_mod.replay(cv, rules))
        print(f"  {row['video']}", flush=True)
    m = score(results, args.level)
    out = {"name": args.name, "rules": str(rules_path), "level": args.level, **m.summary(),
           "by_category": {}, "by_lighting": {}}
    for o in m.outcomes:
        if o.category == "normal":
            continue
        for key, val in (("by_category", o.category), ("by_lighting", o.lighting or "day")):
            b = out[key].setdefault(val, {"events": 0, "caught": 0})
            b["events"] += 1
            b["caught"] += int(o.detected)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{args.name}: recall {m.recall:.1%} {out['recall_ci95']}, precision {m.precision:.1%} "
          f"{out['precision_ci95']}, {m.false_alarms_per_hour:.2f} false alarms/h over {m.normal_hours:.2f} h")


if __name__ == "__main__":
    main()
