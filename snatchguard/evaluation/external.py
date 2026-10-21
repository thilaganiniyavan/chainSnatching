"""External test: score a separate labelled dataset (e.g. ~/snatch_web_dataset) with fixed settings.

Nothing is tuned on it: rule thresholds come from Snatch 1.0 (configs/rules.tuned.yaml) and the fusion
thresholds from configs/fusion.tuned.yaml. It measures how well the system generalises to footage
from other sources, cameras and cities. Every variant is scored: rules only, the learned scorer only
(runtime/learned/oof/web_external/, a model trained on Snatch 1.0 only), and the fused system.

    python -m snatchguard.evaluation.external --config configs/edge.gpu.yaml \\
        --labels ~/snatch_web_dataset/labels.csv --data ~/snatch_web_dataset/clips --rules configs/rules.tuned.yaml
Writes runtime/tuning/external.json; acceptance.py shows it when present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from snatchguard.core.config import REPO_ROOT, RuleConfig, load_yaml
from snatchguard.evaluation import cache as cache_mod
from snatchguard.evaluation.benchmark import load_labels
from snatchguard.evaluation.metrics import IN_SCOPE, Metrics, score

WEB = Path.home() / "snatch_web_dataset"
PROTOCOL = "web_external"


def _breakdown(m: Metrics) -> dict:
    out = {"by_category": {}, "by_lighting": {}}
    for o in m.outcomes:
        if o.category == "normal":
            continue
        for key, val in (("by_category", o.category), ("by_lighting", o.lighting or "day")):
            b = out[key].setdefault(val, {"events": 0, "caught": 0})
            b["events"] += 1
            b["caught"] += int(o.detected)
    return out


def score_external(edge_cfg: dict, rules: RuleConfig, labels: Path, data: Path, fps: float = 8.0,
                   level: str = "any", fusion: dict | None = None, oof_dir: Path | None = None,
                   build_missing: bool = True, name: str = "Snatch-Web") -> dict:
    """fusion: variant -> FusionParams (from fusion tuning). Without it only the rules are scored."""
    from snatchguard.evaluation.fusion import OOF_DIR, Item, evaluate, load_windows, prepare
    oof_dir = oof_dir or OOF_DIR
    tag = cache_mod.perception_tag(edge_cfg, fps)
    items, missing = [], []
    for row in load_labels(labels):
        if row["category"] not in IN_SCOPE and row["category"] != "normal":
            continue
        path = data / row["video"]
        if not path.exists():
            missing.append(row["video"])
            continue
        cf = cache_mod.cache_path(str(path), tag)
        if not cf.exists():
            if not build_missing:
                missing.append(row["video"])
                continue
            cache_mod.save(cache_mod.build(str(path), edge_cfg, fps), cf)
            print(f"  cached {row['video']}", flush=True)
        items.append(Item(row, str(path), cf))
    if not items:
        return {}
    from snatchguard.evaluation.fusion import FusionParams
    fusion = fusion or {"rules": FusionParams(use_model=False)}
    has_model = any(load_windows(PROTOCOL, it.video_path, oof_dir) is not None for it in items)
    out = {"name": name, "level": level, "videos": len(items), "missing": missing, "variants": {}}
    for variant, params in fusion.items():
        if variant != "rules" and not has_model:
            continue
        prepared = prepare(items, rules, PROTOCOL if variant != "rules" else None, oof_dir)
        m = evaluate(prepared, params, variant, level)
        out["variants"][variant] = {**m.summary(), **_breakdown(m),
                                    "clips": [{"video": o.video, "category": o.category, "detected": o.detected,
                                               "true_alerts": o.true_alerts, "false_alarms": o.false_alarms}
                                              for o in m.outcomes]}
    # the rules-only numbers stay at the top level (the format acceptance.py has always read)
    out.update({k: v for k, v in out["variants"]["rules"].items() if k != "clips"})
    return out


def main() -> None:
    from snatchguard.evaluation.fusion import FUSION_TUNED, FusionParams, load_rules
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "edge.yaml"))
    ap.add_argument("--rules", default=str(REPO_ROOT / "configs" / "rules.tuned.yaml"))
    ap.add_argument("--fusion", default=str(REPO_ROOT / "runtime" / "tuning" / "fusion.json"),
                    help="fusion tuning result (per-variant thresholds); rules only when missing")
    ap.add_argument("--labels", default=str(WEB / "labels.csv"))
    ap.add_argument("--data", default=str(WEB / "clips"))
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--level", choices=["any", "confirmed"], default="any")
    ap.add_argument("--name", default="Snatch-Web")
    ap.add_argument("--out", default=str(REPO_ROOT / "runtime" / "tuning" / "external.json"))
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    rules_path = Path(args.rules) if Path(args.rules).exists() else REPO_ROOT / "configs" / "rules.yaml"
    rules = load_rules(rules_path)
    fusion = {"rules": FusionParams(use_model=False)}
    if Path(args.fusion).exists():
        res = json.loads(Path(args.fusion).read_text(encoding="utf-8"))
        fusion.update({v: FusionParams.from_dict(r["params_final"]) for v, r in res["variants"].items() if v != "rules"})
    elif FUSION_TUNED.exists():
        fusion["fused"] = FusionParams.load(FUSION_TUNED)
    out = score_external(cfg, rules, Path(args.labels), Path(args.data), args.fps, args.level, fusion, name=args.name)
    if not out:
        raise SystemExit("no external videos found")
    out["rules"] = str(rules_path)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for v, r in out["variants"].items():
        print(f"{args.name} [{v}]: recall {r['recall']:.1%} {r['recall_ci95']}, precision {r['precision']:.1%} "
              f"{r['precision_ci95']}, {r['false_alarms_per_hour']:.2f} false alarms/h over {r['normal_hours']:.2f} h")


if __name__ == "__main__":
    main()
