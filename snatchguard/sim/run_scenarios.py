"""Run every synthetic scenario through the rule engine and print a verdict table.

    python -m snatchguard.sim.run_scenarios [-v] [--seeds 5]
"""

from __future__ import annotations

import argparse
import logging

from snatchguard.core.types import Level
from snatchguard.rules.engine import Decision, RuleEngine
from snatchguard.sim.synthetic import Scenario, all_scenarios


def run(scenario: Scenario, seed: int = 0) -> list[Decision]:
    engine = RuleEngine()
    out: list[Decision] = []
    for frame in scenario.frames(seed=seed):
        out += engine.step(frame)
    return out


def outcome(decisions: list[Decision]) -> Level:
    alerts = [d for d in decisions if d.level is not Level.DISMISSED]
    if any(d.level is Level.CONFIRMED for d in alerts):
        return Level.CONFIRMED
    return Level.SUSPECT if alerts else Level.DISMISSED


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(message)s")

    failures = 0
    for sc in all_scenarios():
        results = []
        for seed in range(args.seeds):
            decisions = run(sc, seed)
            results.append((outcome(decisions), decisions))
        ok = all(r[0] in sc.expect for r in results)
        failures += not ok
        levels = ",".join(r[0].value[0] for r in results)
        scores = [max((d.verdict.score for d in r[1]), default=0.0) for r in results]
        print(f"{'PASS' if ok else 'FAIL'}  {sc.name:48s} got[{levels}] expect={sorted(e.value for e in sc.expect)} "
              f"score={min(scores):.2f}-{max(scores):.2f}")
        if args.verbose or not ok:
            for d in results[0][1]:
                print(f"      {d.level.value} score={d.verdict.score} comps={d.to_payload()['components']} "
                      f"pen={d.verdict.penalties}")
                for e in d.verdict.evidence:
                    print(f"         {'x' if e.passed else '.'} {e.group:8s} {e.key:16s} {e.detail}")
    print(f"\n{failures} scenario(s) failed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
