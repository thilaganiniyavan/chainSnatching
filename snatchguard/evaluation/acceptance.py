"""Acceptance report: does the system meet the >= 90 % recall and precision targets?

Reads runtime/tuning/result.json (rule tuning, evaluation.tune) and, when present, fusion.json (rules +
learned scorer, evaluation.fusion), external.json and night.json next to it, and writes a Markdown
report: cross-validated numbers with 95 % confidence intervals for rules only, the learned scorer only
and the fused system side by side, per category and lighting, every clip's outcome, the external test,
and the tuned thresholds.

    python -m snatchguard.evaluation.acceptance
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from snatchguard.evaluation.metrics import wilson
from snatchguard.evaluation.tune import OUT_DIR

VARIANT_NAME = {"rules": "Rules only", "model": "Learned scorer only", "fused": "Rules + learned scorer (fused)"}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ci(c: list[float]) -> str:
    return f"{100 * c[0]:.0f}–{100 * c[1]:.0f}%"


def prec(m: dict) -> str:
    n = m["true_alerts"] + m["false_alarms"]
    return f"**{pct(m['precision'])}** ({m['true_alerts']}/{n}, 95% CI {ci(m['precision_ci95'])})" if n else \
        "n/a (no alerts raised)"


def rec(m: dict) -> str:
    return f"**{pct(m['recall'])}** ({m['events_detected']}/{m['events']}, 95% CI {ci(m['recall_ci95'])})"


def verdict_lines(cv: dict, target: float, passes: bool, system: str) -> list[str]:
    """PASS is never printed without the cross-validated numbers and their lower 95 % bounds beside it."""
    n_alerts = cv["true_alerts"] + cv["false_alarms"]
    lo_r = cv["recall_ci95"][0]
    lo_p = cv["precision_ci95"][0] if n_alerts else 0.0
    lines = [f"## Result: {'**PASS**' if passes else '**NOT MET**'} against the {pct(target)} recall and precision "
             f"targets ({system})", "",
             f"Cross-validated recall {rec(cv)}, precision {prec(cv)}. "
             f"Lower 95% bounds: recall {pct(lo_r)}, precision {pct(lo_p) if n_alerts else 'n/a'}.", ""]
    if passes and (lo_r < target or lo_p < target):
        lines += ["> The point estimates meet the target, but the lower confidence bounds do not: with this many "
                  "events the data cannot rule out a true rate below the target.", ""]
    return lines


def _caught(o: dict) -> str:
    if o["category"] == "normal":
        return f"{o['false_alarms']} FA" if o["false_alarms"] else "–"
    s = "yes" if o["detected"] else "**missed**"
    return s + (f" (+{o['false_alarms']} FA)" if o["false_alarms"] else "")


def render(res: dict | None, fusion: dict | None = None, external: dict | None = None,
           night: dict | None = None) -> str:
    base = res or {}
    target = (fusion or base)["target"]
    level = (fusion or base)["level"]
    lines = [
        "# SnatchGuard acceptance report",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. Alert level counted: `{level}` "
        f"({'CONFIRMED and SUSPECT' if level == 'any' else 'CONFIRMED only'}).",
        "",
    ]
    variants = (fusion or {}).get("variants") or {}
    if fusion and variants:
        deployed = fusion["deployed_variant"]
        d = variants[deployed]
        lines += verdict_lines(d["cross_validated"], target, d["passes"], f"deployed: {VARIANT_NAME[deployed]}")
    else:
        lines += verdict_lines(base["cross_validated"], target, base["passes"], "rules only")
    lines += ["Numbers are **cross-validated**: thresholds tuned on one half of the data were scored on the "
              "other half, and the two halves pooled. No number below was measured on data it was tuned on.", ""]
    if fusion:
        lines += [f"Rule thresholds: {fusion.get('rules', 'see below')}.", ""]

    if variants:
        lines += ["## Rules, learned scorer and fused system side by side", "",
                  "| system | recall (events caught) | precision (alerts that were real) | false alarms / hour | "
                  "fit on all data (optimistic, not reported) |", "|---|---|---|---|---|"]
        for v, r in variants.items():
            cv, fit = r["cross_validated"], r["final_fit_on_all_data"]
            mark = " ← deployed" if v == fusion["deployed_variant"] else ""
            lines.append(f"| {VARIANT_NAME[v]}{mark} | {rec(cv)} | {prec(cv)} | {cv['false_alarms_per_hour']:.2f} | "
                         f"recall {pct(fit['recall'])}, precision {pct(fit['precision'])} |")
        if not fusion.get("model_available"):
            lines += ["", "> No learned-scorer scores were found (runtime/learned/oof/snatch_ab/), so only the rules "
                      "were evaluated."]
        lines.append("")
    if res:
        cv, dflt, fit = res["cross_validated"], res["default_rules"], res["final_fit_on_all_data"]
        lines += ["## Rule thresholds: before and after tuning", "",
                  "| | recall | precision | false alarms / hour |", "|---|---|---|---|",
                  f"| Cross-validated (tuned per fold) | {rec(cv)} | {prec(cv)} | {cv['false_alarms_per_hour']:.2f} |",
                  f"| Default thresholds (before tuning) | {pct(dflt['recall'])} | {prec(dflt).replace('**', '')} | "
                  f"{dflt['false_alarms_per_hour']:.2f} |",
                  f"| Final thresholds on all data (optimistic) | {pct(fit['recall'])} | {prec(fit).replace('**', '')} | "
                  f"{fit['false_alarms_per_hour']:.2f} |", ""]
    data = (fusion or base)["data"]
    lines += [f"Data: {data['events']} labelled snatch events, {data['normal_hours']} h of held-out normal CCTV, "
              f"{data['videos']} videos.", ""]
    n_events = data["events"]
    if n_events < 30:
        lines += [f"> With {n_events} events, one missed snatch moves recall by {100 / max(n_events, 1):.1f} points; "
                  "the confidence intervals are the honest range.", ""]

    # per-variant outcomes (rules-only result.json when there is no fusion run)
    outcomes = {v: r["outcomes"] for v, r in variants.items()} or {"rules": base["outcomes"]}
    names = list(outcomes)
    for title, key in (("category", "category"), ("lighting", "lighting")):
        groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for v, os_ in outcomes.items():
            for o in os_:
                if o["category"] != "normal":
                    groups[o[key] or "day"][v].append(o)
        lines += [f"## By {title}", "", f"| {title} | events | " + " | ".join(VARIANT_NAME[v] for v in names) + " |",
                  "|---|---|" + "---|" * len(names)]
        for g, per in sorted(groups.items()):
            n = len(next(iter(per.values())))
            cells = []
            for v in names:
                k = sum(o["detected"] for o in per.get(v, []))
                lo, hi = wilson(k, n)
                cells.append(f"{k}/{n} ({100 * lo:.0f}–{100 * hi:.0f}%)")
            lines.append(f"| {g} | {n} | " + " | ".join(cells) + " |")
        if key == "category":
            cells = []
            for v in names:
                normals = [o for o in outcomes[v] if o["category"] == "normal"]
                hours = sum(o["duration_s"] for o in normals) / 3600
                fa = sum(o["false_alarms"] for o in normals)
                cells.append(f"{fa} ({fa / hours if hours else 0:.2f}/h)")
            lines.append("| false alarms on normal footage | – | " + " | ".join(cells) + " |")
        lines.append("")

    if night and night.get("modes"):
        lines += ["## Night (simulated IR and low-light versions of the same held-out footage, rules only)", "",
                  "Scored with the deployed thresholds, nothing tuned on it. Simulated night models IR fall-off, "
                  "low light, colour cast, noise and blur; it is a robustness check, not a substitute for real "
                  "night recordings.", "",
                  "| condition | recall | precision | false alarms / hour | motorcycle caught | on foot caught |",
                  "|---|---|---|---|---|---|"]
        for mode, r in night["modes"].items():
            bc = r.get("by_category", {})
            cell = lambda c: f"{bc[c]['caught']}/{bc[c]['events']}" if c in bc else "–"  # noqa: E731
            lines.append(f"| {'IR (infrared camera)' if mode == 'ir' else 'colour low-light'} | {pct(r['recall'])} "
                         f"({ci(r['recall_ci95'])}) | {prec(r)} | {r['false_alarms_per_hour']:.2f} | "
                         f"{cell('snatch_bike')} | {cell('snatch_foot')} |")
        lines.append("")

    if external:
        ext_vars = external.get("variants") or {"rules": external}
        lines += [f"## External test: {external['name']} (independent footage; every threshold fixed from Snatch 1.0, "
                  "the learned scorer trained on Snatch 1.0 only)", "",
                  "| system | recall | precision | false alarms / hour | by category | by lighting |",
                  "|---|---|---|---|---|---|"]
        for v, e in ext_vars.items():
            lines.append(f"| {VARIANT_NAME.get(v, v)} | {rec(e)} | {prec(e)} | {e['false_alarms_per_hour']:.2f} over "
                         f"{e['normal_hours']} h | "
                         + ", ".join(f"{k}: {x['caught']}/{x['events']}" for k, x in e["by_category"].items()) + " | "
                         + ", ".join(f"{k}: {x['caught']}/{x['events']}" for k, x in e["by_lighting"].items()) + " |")
        if external.get("missing"):
            lines.append(f"\n{len(external['missing'])} external videos had no perception cache and were skipped.")
        lines.append("")

    lines += ["## Every clip (scored on the fold it was not tuned on)", "",
              "| video | category | lighting | fold | " + " | ".join(VARIANT_NAME[v] for v in names) + " |",
              "|---|---|---|---|" + "---|" * len(names)]
    by_video: dict[str, dict[str, dict]] = defaultdict(dict)
    for v, os_ in outcomes.items():
        for o in os_:
            by_video[o["video"]][v] = o
    for video, per in sorted(by_video.items(), key=lambda kv: (next(iter(kv[1].values()))["category"], kv[0])):
        o0 = next(iter(per.values()))
        lines.append(f"| {Path(video).name} | {o0['category']} | {o0.get('lighting') or '–'} | {o0['fold']} | "
                     + " | ".join(_caught(per[v]) if v in per else "–" for v in names) + " |")
    lines.append("")

    if res:
        lines += ["## Tuned rule thresholds (deployed: `" + res["tuned_rules"] + "`)", "",
                  "| parameter | default | tuned |", "|---|---|---|"]
        for k, v in res["params_final"].items():
            d = res["params_default"][k]
            mark = " ←" if abs(v - d) > 1e-6 else ""
            lines.append(f"| `{k}` | {d:.3g} | {v:.3g}{mark} |")
        lines.append("")
    if variants:
        dep = variants[fusion["deployed_variant"]]["params_final"]
        lines += ["## Fusion thresholds (deployed: `configs/fusion.tuned.yaml`; values >= 1 switch a path off)", "",
                  "| parameter | value |", "|---|---|"] + [f"| `{k}` | {v} |" for k, v in dep.items()] + [""]
    lines += ["## Definitions", "",
              "* **Event**: one labelled snatch. An alert counts for it when its contact interval overlaps "
              "[start − 1.5 s, end + 1.5 s] (whole clip if no times were labelled).",
              "* **Recall** = events with at least one alert / events. **Precision** = alerts matching an event "
              "/ all alerts (normal footage and wrong moments inside snatch clips count as false).",
              "* **Learned scorer only**: alerts raised when the scorer's window score stays above a threshold; its "
              "scores on each fold come from a model trained on the other fold.",
              "* **Fused**: the rule decisions gated, upgraded or rescued by the learned scorer, plus model-only "
              "alerts (snatchguard/evaluation/fusion.py; the live edge runs the same code).",
              "* 95% confidence intervals: Wilson score interval.", ""]
    return "\n".join(lines)


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=str(OUT_DIR / "result.json"))
    ap.add_argument("--out", default=str(OUT_DIR / "acceptance_report.md"))
    args = ap.parse_args()
    folder = Path(args.result).parent
    res = _read(Path(args.result))
    fusion = _read(folder / "fusion.json")
    if res is None and fusion is None:
        raise SystemExit(f"neither {args.result} nor {folder / 'fusion.json'} exists: run evaluation.tune first")
    external = _read(folder / "external.json") or (fusion or {}).get("external") or None
    text = render(res, fusion, external, _read(folder / "night.json"))
    Path(args.out).write_text(text, encoding="utf-8")
    if fusion and fusion.get("variants"):
        passes = fusion["variants"][fusion["deployed_variant"]]["passes"]
    else:
        passes = res["passes"]
    print(f"wrote {args.out}  ->  {'PASS' if passes else 'NOT MET'}")


if __name__ == "__main__":
    main()
