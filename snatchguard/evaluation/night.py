"""Night robustness: the held-out test footage turned into simulated IR and low-light night video,
scored with the deployed (tuned) thresholds. Nothing is tuned on it.

    python -m snatchguard.evaluation.cache --config configs/edge.gpu.yaml --night ir --max-seconds 600
    python -m snatchguard.evaluation.cache --config configs/edge.gpu.yaml --night lowlight --max-seconds 600
    python -m snatchguard.evaluation.night --config configs/edge.gpu.yaml --rules configs/rules.tuned.yaml

Simulated night is not real night: the effects model IR fall-off, low light, colour cast, noise and
blur, but the detector was also trained with this family of effects (on other images), so treat
these numbers as a robustness check next to the real-night clips, not as a replacement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from snatchguard.core.config import REPO_ROOT, RuleConfig, load_yaml
from snatchguard.evaluation import cache as cache_mod
from snatchguard.evaluation.benchmark import DATA, load_labels
from snatchguard.evaluation.metrics import IN_SCOPE, score

OUT = REPO_ROOT / "runtime" / "tuning" / "night.json"


def evaluate_mode(mode: str, edge_cfg: dict, rules: RuleConfig, labels: Path, data: Path, fps: float,
                  level: str) -> dict | None:
    tag = cache_mod.perception_tag(edge_cfg, fps, mode)
    results = {}
    for row in load_labels(labels):
        if row.get("fold") not in ("A", "B") or (row["category"] not in IN_SCOPE and row["category"] != "normal"):
            continue
        path = cache_mod.cache_path(str(data / row["video"]), tag)
        if not path.exists():
            continue
        cv = cache_mod.load(path)
        results[row["video"]] = (row, cv.duration_s, cache_mod.replay(cv, rules))
    if not results:
        return None
    m = score(results, level)
    out = m.summary()
    out["by_category"] = {}
    for cat in IN_SCOPE:
        os_ = [o for o in m.outcomes if o.category == cat]
        if os_:
            out["by_category"][cat] = {"events": len(os_), "caught": sum(o.detected for o in os_)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "edge.yaml"))
    ap.add_argument("--rules", default=str(REPO_ROOT / "configs" / "rules.tuned.yaml"))
    ap.add_argument("--labels", default=str(REPO_ROOT / "data" / "labels" / "snatch1.0.csv"))
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--level", choices=["any", "confirmed"], default="any")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rules_path = Path(args.rules) if Path(args.rules).exists() else REPO_ROOT / "configs" / "rules.yaml"
    rules = RuleConfig.load(rules_path)
    cfg = load_yaml(args.config)
    res = {"rules": str(rules_path), "level": args.level, "modes": {}}
    for mode in ("ir", "lowlight"):
        r = evaluate_mode(mode, cfg, rules, Path(args.labels), Path(args.data), args.fps, args.level)
        if r is None:
            print(f"{mode}: no night caches (run evaluation.cache --night {mode})")
            continue
        res["modes"][mode] = r
        print(f"{mode:8s}: recall {r['recall']:.1%} {r['recall_ci95']}, precision {r['precision']:.1%}, "
              f"{r['false_alarms_per_hour']:.2f} false alarms/h over {r['normal_hours']} h")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
