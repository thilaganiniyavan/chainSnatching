"""Acceptance report: does the system meet the >= 90 % recall and precision targets?

Reads runtime/tuning/result.json (written by evaluation.tune) and writes a Markdown report with
the cross-validated numbers, 95 % confidence intervals, per-category breakdown, every clip's outcome,
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


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ci(c: list[float]) -> str:
    return f"{100 * c[0]:.0f}–{100 * c[1]:.0f}%"


def prec(m: dict) -> str:
    n = m["true_alerts"] + m["false_alarms"]
    return f"**{pct(m['precision'])}** ({m['true_alerts']}/{n}, 95% CI {ci(m['precision_ci95'])})" if n else \
        "n/a (no alerts raised)"


def render(res: dict) -> str:
    cv, base, fit = res["cross_validated"], res["default_rules"], res["final_fit_on_all_data"]
    target = res["target"]
    verdict = "**PASS**" if res["passes"] else "**NOT MET**"
    lines = [
        "# SnatchGuard acceptance report",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. Alert level counted: `{res['level']}` "
        f"({'CONFIRMED and SUSPECT' if res['level'] == 'any' else 'CONFIRMED only'}).",
        "",
        f"## Result: {verdict} against the {pct(target)} recall and precision targets",
        "",
        "Numbers are **cross-validated**: thresholds tuned on one half of the data were scored on the "
        "other half, and the two halves pooled. No number below was measured on data it was tuned on.",
        "",
        "| | recall (events caught) | precision (alerts that were real) | false alarms / hour of normal footage |",
        "|---|---|---|---|",
        f"| **Cross-validated (reported)** | **{pct(cv['recall'])}** ({cv['events_detected']}/{cv['events']}, "
        f"95% CI {ci(cv['recall_ci95'])}) | {prec(cv)} | {cv['false_alarms_per_hour']:.2f} |",
        f"| Default thresholds (before tuning) | {pct(base['recall'])} | {prec(base).replace('**', '')} | "
        f"{base['false_alarms_per_hour']:.2f} |",
        f"| Final thresholds on all data (optimistic, not the reported number) | {pct(fit['recall'])} | "
        f"{prec(fit).replace('**', '')} | {fit['false_alarms_per_hour']:.2f} |",
        "",
        f"Data: {res['data']['events']} labelled snatch events, {res['data']['normal_hours']} h of held-out "
        f"normal CCTV, {res['data']['videos']} videos.",
        "",
    ]
    if cv["events"] < 30:
        lines += [f"> With {cv['events']} events, one missed snatch moves recall by "
                  f"{100 / max(cv['events'], 1):.1f} points; the confidence interval above is the honest range.", ""]

    groups: dict[str, list[dict]] = defaultdict(list)
    for o in res["outcomes"]:
        groups[o["category"]].append(o)
    lines += ["## By category", "", "| category | events | caught | recall | false alarms |", "|---|---|---|---|---|"]
    for cat, os_ in sorted(groups.items()):
        if cat == "normal":
            hours = sum(o["duration_s"] for o in os_) / 3600
            fa = sum(o["false_alarms"] for o in os_)
            lines.append(f"| normal footage ({hours:.1f} h) | – | – | – | {fa} ({fa / hours if hours else 0:.2f}/h) |")
        else:
            caught = sum(o["detected"] for o in os_)
            lo, hi = wilson(caught, len(os_))
            lines.append(f"| {cat} | {len(os_)} | {caught} | {pct(caught / len(os_))} ({100 * lo:.0f}–{100 * hi:.0f}%) | "
                         f"{sum(o['false_alarms'] for o in os_)} |")
    lines += ["", "## Every clip (scored on the fold it was not tuned on)", "",
              "| video | category | fold | caught | true alerts | false alarms | latency after grab |",
              "|---|---|---|---|---|---|---|"]
    for o in sorted(res["outcomes"], key=lambda o: (o["category"], o["video"])):
        caught = "–" if o["category"] == "normal" else ("yes" if o["detected"] else "**missed**")
        lat = f"{o['latency_s']:.1f} s" if o.get("latency_s") is not None else "–"
        lines.append(f"| {Path(o['video']).name} | {o['category']} | {o['fold']} | {caught} | {o['true_alerts']} | "
                     f"{o['false_alarms']} | {lat} |")
    lines += ["", "## Tuned thresholds (deployed: `" + res["tuned_rules"] + "`)", "",
              "| parameter | default | tuned |", "|---|---|---|"]
    for k, v in res["params_final"].items():
        d = res["params_default"][k]
        mark = " ←" if abs(v - d) > 1e-6 else ""
        lines.append(f"| `{k}` | {d:.3g} | {v:.3g}{mark} |")
    lines += ["", "## Definitions", "",
              "* **Event**: one labelled snatch. An alert counts for it when its contact interval overlaps "
              "[start − 1.5 s, end + 1.5 s] (whole clip if no times were labelled).",
              "* **Recall** = events with at least one alert / events. **Precision** = alerts matching an event "
              "/ all alerts (normal footage and wrong moments inside snatch clips count as false).",
              "* 95% confidence intervals: Wilson score interval.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=str(OUT_DIR / "result.json"))
    ap.add_argument("--out", default=str(OUT_DIR / "acceptance_report.md"))
    args = ap.parse_args()
    res = json.loads(Path(args.result).read_text())
    Path(args.out).write_text(render(res), encoding="utf-8")
    print(f"wrote {args.out}  ->  {'PASS' if res['passes'] else 'NOT MET'}")


if __name__ == "__main__":
    main()
