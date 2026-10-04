"""Cross-validated tuning of the rule thresholds against the >= 90 % precision / recall targets.

Procedure (designed so the reported numbers are not tuned on the data they are measured on):

1. Folds A and B each hold half of the labelled snatch clips and half of the held-out normal videos.
2. Tune on A, measure on B; tune on B, measure on A. The two measurements are pooled: that pooled
   number is the **cross-validated estimate** reported against the targets.
3. Tune once more on A + B: those thresholds are deployed (configs/rules.tuned.yaml).

The search maximises recall while keeping precision >= the target: random sampling over the
parameter ranges, then local refinement around the best point. Rules are replayed from perception
caches (build them first with `python -m snatchguard.evaluation.cache`), in parallel worker processes
that each keep their share of the videos in memory.

    python -m snatchguard.evaluation.tune --config configs/edge.gpu.yaml --trials 150 --workers 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import time
from dataclasses import dataclass
from pathlib import Path

from snatchguard.core.config import DEFAULT_RULES, REPO_ROOT, RuleConfig, load_yaml
from snatchguard.evaluation import cache as cache_mod
from snatchguard.evaluation.metrics import IN_SCOPE, Metrics, merge, score

OUT_DIR = REPO_ROOT / "runtime" / "tuning"

# name -> (low, high). "foot." prefix = override for on-foot attackers only.
SPACE: dict[str, tuple[float, float]] = {
    "confirm_threshold": (0.55, 0.85),
    "suspect_threshold": (0.35, 0.65),
    "neck_radius": (0.15, 0.34),
    "hands_to_neck_radius": (0.14, 0.30),
    "box_contact_upper_frac": (0.30, 0.60),
    "box_contact_min_overlap": (0.04, 0.25),
    "contact_box_factor": (0.40, 0.90),
    "same_depth_tol": (0.15, 0.45),
    "escape_speedup_ratio": (1.2, 2.0),
    "escape_speedup_abs": (0.3, 1.0),
    "escape_fast_departure": (0.7, 1.6),
    "escape_gap_rate": (0.5, 1.2),
    "jerk_dv": (0.8, 2.0),
    "chase_speed": (1.0, 2.0),
    "dwell_min_s": (3.0, 8.0),
    "long_contact_s": (1.2, 3.0),
    "in_traffic_frac": (0.3, 0.8),
    "pedestrian_min_aspect": (1.5, 2.3),
    "pedestrian_min_free_s": (0.5, 1.5),
    "foot.escape_fast_departure": (1.3, 2.4),
    "foot.escape_speedup_ratio": (1.4, 2.4),
}


def normalise(params: dict[str, float]) -> dict[str, float]:
    """Keep the SUSPECT threshold below the CONFIRMED one."""
    p = dict(params)
    if p.get("suspect_threshold", 0) > p.get("confirm_threshold", 1):
        p["suspect_threshold"], p["confirm_threshold"] = p["confirm_threshold"], p["suspect_threshold"]
    return p


def to_rules(base: RuleConfig, params: dict[str, float]) -> RuleConfig:
    params = normalise(params)
    b = {k: v for k, v in params.items() if not k.startswith("foot.")}
    f = {k[5:]: v for k, v in params.items() if k.startswith("foot.")}
    return base.with_overrides(base=b, foot=f)


def defaults(base: RuleConfig) -> dict[str, float]:
    out = {}
    for k in SPACE:
        if k.startswith("foot."):
            out[k] = float((base.raw.get("foot") or {}).get(k[5:], base.raw["base"][k[5:]]))
        else:
            out[k] = float(base.raw["base"][k])
    return out


def objective(m: Metrics, target: float) -> float:
    """Recall, heavily penalised for every point of precision below the target."""
    return m.recall - 3.0 * max(0.0, target - m.precision) + 0.001 * m.precision


# ---- worker processes ----------------------------------------------------------------------------

def _worker(paths: list[str], base_raw: dict, conn) -> None:
    videos = {p: cache_mod.load(Path(p)) for p in paths}
    base = RuleConfig(base_raw)
    while True:
        msg = conn.recv()
        if msg is None:
            return
        params, wanted = msg
        rules = to_rules(base, params)
        conn.send({p: (cv.duration_s, cache_mod.replay(cv, rules)) for p, cv in videos.items() if p in wanted})


class ReplayPool:
    """Each worker owns a fixed share of the cached videos (balanced by size) and keeps them in memory."""

    def __init__(self, paths: list[str], base: RuleConfig, workers: int) -> None:
        ctx = mp.get_context("spawn")
        workers = max(1, min(workers, len(paths)))
        shares: list[list[str]] = [[] for _ in range(workers)]
        loads = [0] * workers
        for p in sorted(paths, key=lambda x: -Path(x).stat().st_size):
            i = loads.index(min(loads))
            shares[i].append(p)
            loads[i] += Path(p).stat().st_size
        self.conns, self.procs = [], []
        for share in shares:
            a, b = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(share, base.raw, b), daemon=True)
            proc.start()
            self.conns.append(a)
            self.procs.append(proc)

    def run(self, params: dict, wanted: set[str]) -> dict[str, tuple[float, list[dict]]]:
        for c in self.conns:
            c.send((params, wanted))
        out = {}
        for c in self.conns:
            out.update(c.recv())
        return out

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(None)
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()


# ---- search ---------------------------------------------------------------------------------------

@dataclass
class Entry:
    row: dict
    path: str


def evaluate(pool: ReplayPool, entries: list[Entry], params: dict, level: str) -> Metrics:
    res = pool.run(params, {e.path for e in entries})
    by_path = {e.path: e.row for e in entries}
    return score({by_path[p]["video"]: (by_path[p], dur, dec) for p, (dur, dec) in res.items()}, level)


def search(pool: ReplayPool, entries: list[Entry], start: dict, trials: int, target: float, level: str,
           seed: int, log_prefix: str = "") -> tuple[dict, Metrics, list[dict]]:
    rng = random.Random(seed)
    history = []

    def run(params: dict) -> tuple[float, Metrics]:
        m = evaluate(pool, entries, params, level)
        obj = objective(m, target)
        history.append({**{k: round(v, 4) for k, v in params.items()}, **m.summary(), "objective": round(obj, 4)})
        return obj, m

    best_p = dict(start)
    best_obj, best_m = run(best_p)
    n_random = int(trials * 0.6)
    t0 = time.time()
    for i in range(1, trials):
        if i < n_random:
            cand = {k: rng.uniform(lo, hi) for k, (lo, hi) in SPACE.items()}
        else:                                   # refine around the best point found so far
            cand = dict(best_p)
            for k in rng.sample(list(SPACE), k=rng.randint(1, 4)):
                lo, hi = SPACE[k]
                cand[k] = min(hi, max(lo, cand[k] + rng.gauss(0, 0.12 * (hi - lo))))
        cand = normalise(cand)
        obj, m = run(cand)
        if obj > best_obj:
            best_obj, best_p, best_m = obj, cand, m
        if i % 10 == 0 or i == trials - 1:
            print(f"{log_prefix}trial {i + 1}/{trials}: best recall {best_m.recall:.3f} precision "
                  f"{best_m.precision:.3f} ({(time.time() - t0) / i:.1f}s/trial)", flush=True)
    return best_p, best_m, history


# ---- the full procedure -----------------------------------------------------------------------------

def run_tuning(entries: list[Entry], base: RuleConfig, trials: int = 150, workers: int = 4, target: float = 0.9,
               level: str = "any", seed: int = 0, out_dir: Path = OUT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = {f: [e for e in entries if e.row.get("fold") == f] for f in ("A", "B")}
    start = defaults(base)
    pool = ReplayPool([e.path for e in entries], base, workers)
    try:
        baseline = {f: evaluate(pool, folds[f], start, level) for f in folds}
        cv_parts, cv_params, histories = [], {}, {}
        for train, test in (("A", "B"), ("B", "A")):
            print(f"== tune on fold {train}, test on fold {test}", flush=True)
            p, m_train, hist = search(pool, folds[train], start, trials, target, level, seed, f"[{train}] ")
            m_test = evaluate(pool, folds[test], p, level)
            print(f"   fold {test}: recall {m_test.recall:.3f} precision {m_test.precision:.3f}", flush=True)
            cv_parts.append(m_test)
            cv_params[train] = p
            histories[train] = hist
        cv = merge(cv_parts)
        print("== final tuning on A+B", flush=True)
        final_p, final_m, hist = search(pool, entries, start, trials, target, level, seed + 1, "[A+B] ")
        histories["AB"] = hist
        default_all = merge([baseline["A"], baseline["B"]])
    finally:
        pool.close()

    tuned = to_rules(base, final_p)
    tuned_path = out_dir / "rules.tuned.yaml"
    tuned.to_yaml(tuned_path)
    result = {
        "target": target, "level": level, "trials": trials,
        "data": {"events": cv.events, "normal_hours": round(cv.normal_hours, 2),
                 "videos": len(entries)},
        "default_rules": default_all.summary(),
        "cross_validated": cv.summary(),
        "final_fit_on_all_data": final_m.summary(),
        "passes": bool(cv.recall >= target and cv.precision >= target),
        "params_default": start, "params_final": final_p, "params_per_fold": cv_params,
        "outcomes": [o.__dict__ | {"alerts": [{k: a[k] for k in ("level", "mode", "score", "contact_start",
                                                                  "decided_at")} for a in o.alerts]}
                     for o in cv.outcomes],
        "tuned_rules": str(tuned_path),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=1, default=str))
    for name, hist in histories.items():
        (out_dir / f"history_{name}.json").write_text(json.dumps(hist))
    return result


def collect_entries(labels: Path, data: Path, edge_cfg: dict, fps: float) -> tuple[list[Entry], list[str]]:
    from snatchguard.evaluation.benchmark import load_labels
    tag = cache_mod.perception_tag(edge_cfg, fps)
    entries, missing = [], []
    for row in load_labels(labels):
        if row.get("fold") not in ("A", "B") or (row["category"] not in IN_SCOPE and row["category"] != "normal"):
            continue
        path = cache_mod.cache_path(str(data / row["video"]), tag)
        (entries.append(Entry(row, str(path))) if path.exists() else missing.append(row["video"]))
    return entries, missing


def main() -> None:
    from snatchguard.evaluation.benchmark import DATA
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "edge.yaml"))
    ap.add_argument("--rules", default=str(DEFAULT_RULES))
    ap.add_argument("--labels", default=str(REPO_ROOT / "data" / "labels" / "snatch1.0.csv"))
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    ap.add_argument("--target", type=float, default=0.90)
    ap.add_argument("--level", choices=["any", "confirmed"], default="any",
                    help="'any': CONFIRMED and SUSPECT both count as alerts (both notify the dashboard)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    entries, missing = collect_entries(Path(args.labels), Path(args.data), load_yaml(args.config), args.fps)
    if missing:
        print(f"WARNING: {len(missing)} labelled videos have no cache yet (run evaluation.cache): {missing[:5]}...")
    if not entries:
        raise SystemExit("no cached videos found")
    res = run_tuning(entries, RuleConfig.load(args.rules), args.trials, args.workers, args.target, args.level,
                     args.seed, Path(args.out))
    cv = res["cross_validated"]
    print(f"\ncross-validated: recall {cv['recall']:.3f} {cv['recall_ci95']}, precision {cv['precision']:.3f} "
          f"{cv['precision_ci95']}, {cv['false_alarms_per_hour']:.2f} false alarms/h -> "
          f"{'PASS' if res['passes'] else 'FAIL'} (target {args.target:.0%})")
    print(f"tuned rules: {res['tuned_rules']}   report: python -m snatchguard.evaluation.acceptance")


if __name__ == "__main__":
    main()
