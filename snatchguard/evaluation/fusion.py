"""Fusion of the rule engine and the learned snatch scorer, shared by the live camera and the evaluation.

The rules decide from geometry over tracks (explainable, but blind when the detector misses the bike or
the clip ends before the getaway); the learned scorer judges the pixels of short time windows (no
tracks needed, but no explanation). `Fuser` combines them in a stream:

* rule CONFIRMED                       -> alert (CONFIRMED)
* rule SUSPECT,   model >= upgrade     -> CONFIRMED;   model >= suspect_gate -> SUSPECT;   else dropped
* rule DISMISSED, model >= rescue      -> SUSPECT (the rules saw the contact, the model saw the snatch)
* no rule alert,  model >= model_threshold for model_k consecutive windows -> model-only alert
* a model-sourced alert within cooldown_s of an earlier alert is merged into it unless it raises the
  level (then it is sent as an upgrade). Rule-vs-rule duplicates are left to the rule engine.

"Model near a decision" = the highest window score overlapping [contact_start - pad, contact_end + pad].
A rule decision waits (at most hold_max_s of stream time) until the model has covered that interval;
when no model windows exist at all, every decision passes through exactly as with the rules alone.

Offline, the same class is fed cached rule decisions and out-of-fold window scores in stream order:

    python -m snatchguard.evaluation.fusion --tune                     # rules-only if no oof files exist
    python -m snatchguard.evaluation.fusion --tune --tune-result runtime/tuning/result.json
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable

import yaml

from snatchguard.core.config import DEFAULT_RULES, REPO_ROOT, RuleConfig, load_yaml
from snatchguard.evaluation.metrics import IN_SCOPE, Metrics, merge, score

FUSION_DEFAULT = REPO_ROOT / "configs" / "fusion.yaml"
FUSION_TUNED = REPO_ROOT / "configs" / "fusion.tuned.yaml"
OOF_DIR = REPO_ROOT / "runtime" / "learned" / "oof"
OUT_DIR = REPO_ROOT / "runtime" / "tuning"
DECISION_CACHE = REPO_ROOT / "runtime" / "fusion_dev" / "decisions"
OFF = 1.01                       # any threshold >= 1 disables its path (scores are probabilities)
RANK = {"DISMISSED": 0, "SUSPECT": 1, "CONFIRMED": 2}
VARIANTS = ("rules", "model", "fused")


@dataclass
class FusionParams:
    suspect_gate: float = 0.0          # keep a rule SUSPECT only if the model near it >= this (0 = always)
    upgrade_threshold: float = OFF     # rule SUSPECT -> CONFIRMED when the model near it >= this
    rescue_threshold: float = OFF      # rule DISMISSED -> SUSPECT when the model near it >= this
    model_threshold: float = OFF       # model-only alert: score >= this ...
    model_k: int = 2                   # ... for this many consecutive windows
    model_confirm_threshold: float = OFF   # model-only alert is CONFIRMED when its peak >= this
    context_pad_s: float = 1.5
    hold_max_s: float = 3.0
    cooldown_s: float = 8.0
    use_rules: bool = True
    use_model: bool = True

    @classmethod
    def from_dict(cls, d: dict | None) -> "FusionParams":
        names = {f.name for f in fields(cls)}
        p = cls(**{k: v for k, v in (d or {}).items() if k in names})
        p.model_k = max(1, int(round(p.model_k)))
        return p

    @classmethod
    def load(cls, path: str | Path | None = None) -> "FusionParams":
        path = Path(path) if path else (FUSION_TUNED if FUSION_TUNED.exists() else FUSION_DEFAULT)
        raw = load_yaml(path) if path.exists() else {}
        return cls.from_dict(raw.get("params", raw))

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass
class FusedAlert:
    payload: dict
    ref: Any = None              # the rule engine's Decision (live: snapshot roles, clip), None for model-only


def _window(w) -> tuple[float, float, float]:
    if isinstance(w, (tuple, list)):
        return float(w[0]), float(w[1]), float(w[2])
    return float(w.t_start), float(w.t_end), float(w.score)


class Fuser:
    """Streaming fusion for one camera / one video. Feed rule decisions (payload dicts) and window scores
    as they become available, call poll(now) to get the alerts that are due."""

    def __init__(self, params: FusionParams | None = None, use_model: bool | None = None,
                 use_rules: bool | None = None, keep_s: float = 120.0) -> None:
        self.p = params or FusionParams()
        self.use_model = self.p.use_model if use_model is None else use_model
        self.use_rules = self.p.use_rules if use_rules is None else use_rules
        self.keep_s = keep_s
        self._ends: list[float] = []                # window t_end, ascending
        self._wins: list[tuple[float, float, float]] = []
        self.covered_until = -math.inf
        self.pending: list[tuple[dict, Any]] = []
        self._run: list[tuple[float, float, float]] = []
        self._run_fired = False
        self._model_ready: list[dict] = []
        self._emitted: list[dict] = []              # recent alerts (for cooldown merges)
        self._next_id = 1
        self.latest_score: float | None = None

    @property
    def model_live(self) -> bool:
        return self.use_model and bool(self._wins)

    # ---- inputs ------------------------------------------------------------------------------------
    def add_rule(self, decision: dict, ref: Any = None) -> None:
        if self.use_rules:
            self.pending.append((decision, ref))

    def add_windows(self, windows: Iterable) -> None:
        if not self.use_model:
            return
        p = self.p
        for w in windows:
            ts, te, s = _window(w)
            if self._ends and te < self._ends[-1]:          # out of order (should not happen): keep sorted
                i = bisect.bisect_right(self._ends, te)
                self._ends.insert(i, te)
                self._wins.insert(i, (ts, te, s))
            else:
                self._ends.append(te)
                self._wins.append((ts, te, s))
            self.covered_until = max(self.covered_until, te)
            self.latest_score = s
            if s >= p.model_threshold:
                self._run.append((ts, te, s))
                if len(self._run) >= p.model_k and not self._run_fired:
                    self._run_fired = True
                    peak = max(self._run, key=lambda x: x[2])
                    self._model_ready.append({
                        "level": "CONFIRMED" if peak[2] >= p.model_confirm_threshold else "SUSPECT",
                        "source": "model", "mode": "unknown", "score": round(peak[2], 4),
                        "model_score": round(peak[2], 4), "rule_score": None,
                        "contact_start": round(peak[0], 3), "contact_end": round(peak[1], 3),
                        "decided_at": round(te, 3), "latency_s": round(te - peak[0], 3),
                        "evidence": [{"key": "model", "group": "model", "passed": True, "value": round(peak[2], 3),
                                      "detail": f"learned scorer {peak[2]:.2f} for {len(self._run)} windows in a row"}],
                    })
            else:
                self._run, self._run_fired = [], False

    # ---- outputs -----------------------------------------------------------------------------------
    def model_near(self, start: float, end: float) -> float | None:
        """Highest window score overlapping [start - pad, end + pad]; None when the model saw nothing there."""
        if not self.model_live:
            return None
        lo, hi = start - self.p.context_pad_s, end + self.p.context_pad_s
        i = bisect.bisect_left(self._ends, lo)
        best = None
        for ts, te, s in self._wins[i:]:
            if ts > hi:
                if te - ts < 60:          # windows have a fixed length: later ones start even later
                    break
                continue
            best = s if best is None else max(best, s)
        return best

    def poll(self, now: float, force: bool = False) -> list[FusedAlert]:
        out: list[FusedAlert] = []
        still = []
        for d, ref in self.pending:
            need = d["contact_end"] + self.p.context_pad_s
            if (force or not self.model_live or self.covered_until >= need
                    or now - d["decided_at"] >= self.p.hold_max_s):
                a = self._resolve(d, ref)
                if a is not None:
                    out.append(a)
            else:
                still.append((d, ref))
        self.pending = still
        ready, self._model_ready = self._model_ready, []
        out += [FusedAlert(m) for m in ready]
        out.sort(key=lambda a: a.payload["decided_at"])
        emitted = [a for a in out if self._admit(a)]
        self._prune(now)
        return emitted

    def flush(self, now: float = math.inf) -> list[FusedAlert]:
        """End of stream: resolve every waiting decision with whatever the model has produced."""
        return self.poll(now, force=True)

    # ---- internals ---------------------------------------------------------------------------------
    def _resolve(self, d: dict, ref: Any) -> FusedAlert | None:
        p = self.p
        level = d["level"]
        ms = self.model_near(d["contact_start"], d["contact_end"])
        source = "rules"
        if ms is None:
            if level == "DISMISSED":
                return None
        elif level == "SUSPECT":
            if ms >= p.upgrade_threshold:
                level, source = "CONFIRMED", "fused"
            elif ms < p.suspect_gate:
                return None
            elif p.suspect_gate > 0:
                source = "fused"
        elif level == "DISMISSED":
            if ms < p.rescue_threshold:
                return None
            level, source = "SUSPECT", "fused"
        payload = {**d, "level": level, "source": source, "rule_level": d["level"],
                   "rule_score": d.get("score"), "model_score": None if ms is None else round(ms, 4)}
        return FusedAlert(payload, ref)

    def _admit(self, a: FusedAlert) -> bool:
        pl = a.payload
        cd = self.p.cooldown_s
        near = [e for e in self._emitted
                if (pl["source"] == "model" or e["source"] == "model")
                and pl["contact_start"] <= e["contact_end"] + cd and pl["contact_end"] >= e["contact_start"] - cd]
        if near:
            top = max(near, key=lambda e: RANK[e["level"]])
            if RANK[pl["level"]] <= RANK[top["level"]]:
                return False
            pl["upgrade_of"] = top["fusion_id"]
        pl["fusion_id"] = self._next_id
        self._next_id += 1
        self._emitted.append(pl)
        return True

    def _prune(self, now: float) -> None:
        if not math.isfinite(now):
            return
        cut = now - self.keep_s
        if self._ends and self._ends[0] < cut:
            i = bisect.bisect_left(self._ends, cut)
            del self._ends[:i], self._wins[:i]
        self._emitted = [e for e in self._emitted if e["contact_end"] >= cut]


def fuse_video(decisions: list[dict], windows: list | None, params: FusionParams,
               use_rules: bool = True) -> list[dict]:
    """Offline: replay one video's rule decisions and window scores through a Fuser in stream order."""
    f = Fuser(params, use_model=windows is not None and params.use_model, use_rules=use_rules and params.use_rules)
    stream = [(_window(w)[1], 0, w) for w in (windows or [])] + [(d["decided_at"], 1, d) for d in decisions]
    stream.sort(key=lambda e: (e[0], e[1]))
    out: list[FusedAlert] = []
    for t, kind, x in stream:
        if kind == 0:
            f.add_windows([x])
        else:
            f.add_rule(x)
        out += f.poll(t)
    out += f.flush()
    return [a.payload for a in out]


# ==== offline data =================================================================================

def video_stem(video_path: str | Path) -> str:
    p = Path(video_path)
    return p.parent.name.replace(" ", "_") + "__" + p.stem


def load_windows(protocol: str, video_path: str | Path, oof_dir: Path = OOF_DIR) -> list | None:
    path = oof_dir / protocol / f"{video_stem(video_path)}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return [tuple(w) for w in d.get("windows", [])]


def load_rules(path: str | Path) -> RuleConfig:
    """A rules file laid over configs/rules.yaml: tuned files written before a parameter was added keep working."""
    from snatchguard.core.config import _deep_merge
    path = Path(path)
    if path.resolve() == DEFAULT_RULES.resolve():
        return RuleConfig.load(path)
    return RuleConfig(_deep_merge(load_yaml(DEFAULT_RULES), load_yaml(path)))


def _rules_key(rules: RuleConfig) -> str:
    """Thresholds and the rule code itself: editing snatchguard/rules/*.py invalidates memoised decisions."""
    h = hashlib.md5(json.dumps(rules.raw, sort_keys=True, default=str).encode())
    for src in sorted((REPO_ROOT / "snatchguard" / "rules").glob("*.py")):
        h.update(src.read_bytes())
    return h.hexdigest()[:10]


def replay_decisions(cache_file: Path, rules: RuleConfig) -> tuple[float, list[dict]]:
    """Every rule decision (DISMISSED too, so fusion can rescue them) for one cached video; memoised on disk."""
    from snatchguard.evaluation import cache as cache_mod
    out = DECISION_CACHE / _rules_key(rules) / (cache_file.name.replace(".pkl.gz", "") + ".json")
    if out.exists() and out.stat().st_mtime >= cache_file.stat().st_mtime:
        d = json.loads(out.read_text(encoding="utf-8"))
        return d["duration_s"], d["decisions"]
    cv = cache_mod.load(cache_file)
    keep = ("level", "mode", "score", "confidence", "lighting", "contact_start", "contact_end", "decided_at",
            "latency_s", "victim_id", "vehicle_id", "flushed_at_end")
    decs = [{k: d[k] for k in keep if k in d} for d in cache_mod.replay(cv, rules)]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"duration_s": cv.duration_s, "decisions": decs}), encoding="utf-8")
    return cv.duration_s, decs


@dataclass
class Item:
    row: dict
    video_path: str               # the source video (for oof stems)
    cache_file: Path


def collect(labels: Path, data: Path, edge_cfg: dict, fps: float, folds: tuple[str, ...] | None = ("A", "B")
            ) -> tuple[list[Item], list[str]]:
    from snatchguard.evaluation import cache as cache_mod
    from snatchguard.evaluation.benchmark import load_labels
    tag = cache_mod.perception_tag(edge_cfg, fps)
    items, missing = [], []
    for row in load_labels(labels):
        if folds and row.get("fold") not in folds:
            continue
        if row["category"] not in IN_SCOPE and row["category"] != "normal":
            continue
        vp = str(data / row["video"])
        cf = cache_mod.cache_path(vp, tag)
        (items.append(Item(row, vp, cf)) if cf.exists() else missing.append(row["video"]))
    return items, missing


@dataclass
class Prepared:
    item: Item
    duration: float
    decisions: list[dict]
    windows: list | None


def prepare(items: list[Item], rules: RuleConfig, protocol: str | None, oof_dir: Path = OOF_DIR) -> list[Prepared]:
    out = []
    for it in items:
        dur, decs = replay_decisions(it.cache_file, rules)
        win = load_windows(protocol, it.video_path, oof_dir) if protocol else None
        out.append(Prepared(it, dur, decs, win))
    return out


def evaluate(prepared: list[Prepared], params: FusionParams, variant: str, level: str = "any") -> Metrics:
    results = {}
    for pr in prepared:
        if variant == "rules":
            alerts = fuse_video(pr.decisions, None, params)
        elif variant == "model":
            alerts = fuse_video([], pr.windows, params, use_rules=False) if pr.windows is not None else []
        else:
            alerts = fuse_video(pr.decisions, pr.windows, params)
        results[pr.item.row["video"]] = (pr.item.row, pr.duration, alerts)
    return score(results, level)


# ==== search ========================================================================================

def _maybe_off(rng: random.Random, lo: float, hi: float, p_off: float) -> float:
    return OFF if rng.random() < p_off else rng.uniform(lo, hi)


def sample(rng: random.Random, variant: str) -> FusionParams:
    if variant == "model":
        return FusionParams(model_threshold=rng.uniform(0.2, 0.99), model_k=rng.randint(1, 4),
                            model_confirm_threshold=_maybe_off(rng, 0.6, 0.99, 0.5),
                            cooldown_s=rng.uniform(3, 20), use_rules=False)
    return FusionParams(
        suspect_gate=0.0 if rng.random() < 0.3 else rng.uniform(0.05, 0.9),
        upgrade_threshold=_maybe_off(rng, 0.5, 0.99, 0.5),
        rescue_threshold=_maybe_off(rng, 0.2, 0.99, 0.3),
        model_threshold=_maybe_off(rng, 0.2, 0.99, 0.3),
        model_k=rng.randint(1, 4),
        model_confirm_threshold=_maybe_off(rng, 0.6, 0.99, 0.6),
        context_pad_s=rng.uniform(0.5, 3.0),
        cooldown_s=rng.uniform(3, 20))


def perturb(rng: random.Random, p: FusionParams, variant: str) -> FusionParams:
    d = p.to_dict()
    keys = (["model_threshold", "model_k", "model_confirm_threshold", "cooldown_s"] if variant == "model" else
            ["suspect_gate", "upgrade_threshold", "rescue_threshold", "model_threshold", "model_k",
             "model_confirm_threshold", "context_pad_s", "cooldown_s"])
    span = {"model_k": 1.5, "cooldown_s": 4.0, "context_pad_s": 0.6}
    for k in rng.sample(keys, k=rng.randint(1, 3)):
        if k == "model_k":
            d[k] = max(1, min(5, d[k] + rng.choice([-1, 1])))
        elif k in span:
            d[k] = max(0.2, d[k] + rng.gauss(0, span[k]))
        else:
            v = d[k] if d[k] < 1 else rng.uniform(0.6, 0.99)
            d[k] = min(0.99, max(0.0, v + rng.gauss(0, 0.08)))
    return FusionParams.from_dict(d)


def objective(m: Metrics, target: float) -> float:
    from snatchguard.evaluation.tune import objective as rules_objective
    return rules_objective(m, target)


def search(prepared: list[Prepared], variant: str, trials: int, target: float, level: str, seed: int,
           prefix: str = "") -> tuple[FusionParams, Metrics]:
    if variant == "rules":
        p = FusionParams(use_model=False)
        return p, evaluate(prepared, p, "rules", level)
    rng = random.Random(seed)
    best_p = FusionParams(use_rules=variant != "model")
    if variant == "model":
        best_p.model_threshold = 0.5
    best_m = evaluate(prepared, best_p, variant, level)
    best_o = objective(best_m, target)
    t0 = time.time()
    for i in range(1, trials):
        cand = sample(rng, variant) if i < int(trials * 0.6) else perturb(rng, best_p, variant)
        m = evaluate(prepared, cand, variant, level)
        o = objective(m, target)
        if o > best_o:
            best_o, best_p, best_m = o, cand, m
        if i % 100 == 0 or i == trials - 1:
            print(f"{prefix}{variant} trial {i + 1}/{trials}: recall {best_m.recall:.3f} precision "
                  f"{best_m.precision:.3f} ({(time.time() - t0) / i * 1000:.0f} ms/trial)", flush=True)
    return best_p, best_m


def _outcomes(m: Metrics) -> list[dict]:
    keep = ("level", "source", "mode", "score", "model_score", "contact_start", "decided_at")
    return [o.__dict__ | {"alerts": [{k: a.get(k) for k in keep} for a in o.alerts]} for o in m.outcomes]


def run(items: list[Item], fold_rules: dict[str, RuleConfig], final_rules: RuleConfig, trials: int, target: float,
        level: str, seed: int, oof_dir: Path = OOF_DIR) -> dict:
    """Cross-validated fusion tuning. fold_rules[f] = the rules tuned on fold f (or the same rules for both)."""
    folds = {f: [it for it in items if it.row.get("fold") == f] for f in ("A", "B")}
    have_model = any(load_windows("snatch_ab", it.video_path, oof_dir) is not None for it in items)
    variants = list(VARIANTS) if have_model else ["rules"]
    result: dict = {"target": target, "level": level, "trials": trials, "model_available": have_model,
                    "deployed_variant": "fused" if have_model else "rules", "variants": {}}
    if have_model:
        missing = [it.row["video"] for it in items if load_windows("snatch_ab", it.video_path, oof_dir) is None]
        result["videos_without_model_scores"] = missing
    for variant in variants:
        parts, per_fold = [], {}
        for train, test in (("A", "B"), ("B", "A")):
            rules = fold_rules[train]
            tr = prepare(folds[train], rules, "snatch_ab" if have_model else None, oof_dir)
            te = prepare(folds[test], rules, "snatch_ab" if have_model else None, oof_dir)
            p, _ = search(tr, variant, trials, target, level, seed, f"[{train}] ")
            m = evaluate(te, p, variant, level)
            print(f"   {variant}: tuned on {train}, fold {test}: recall {m.recall:.3f} precision {m.precision:.3f}",
                  flush=True)
            parts.append(m)
            per_fold[train] = p.to_dict()
        cv = merge(parts)
        allp = prepare(items, final_rules, "snatch_ab" if have_model else None, oof_dir)
        final_p, final_m = search(allp, variant, trials, target, level, seed + 1, "[A+B] ")
        result["variants"][variant] = {
            "cross_validated": cv.summary(), "final_fit_on_all_data": final_m.summary(),
            "passes": bool(cv.recall >= target and cv.precision >= target),
            "params_per_fold": per_fold, "params_final": final_p.to_dict(), "outcomes": _outcomes(cv)}
    any_cv = result["variants"]["rules"]["cross_validated"]
    result["data"] = {"events": any_cv["events"], "normal_hours": any_cv["normal_hours"], "videos": len(items)}
    return result


def write_deployed(result: dict, path: Path = FUSION_TUNED) -> None:
    v = result["deployed_variant"]
    body = {"variant": v, "generated": time.strftime("%Y-%m-%d %H:%M"),
            "cross_validated": {k: result["variants"][v]["cross_validated"][k] for k in ("recall", "precision")},
            "params": result["variants"][v]["params_final"]}
    path.write_text("# written by `python -m snatchguard.evaluation.fusion --tune`; the live edge loads it\n"
                    + yaml.safe_dump(body, sort_keys=False), encoding="utf-8")


def main() -> None:
    from snatchguard.evaluation.benchmark import DATA
    from snatchguard.evaluation.tune import to_rules
    ap = argparse.ArgumentParser(description="Cross-validated fusion of rules and the learned scorer")
    ap.add_argument("--tune", action="store_true", help="run the cross-validated tuning (the only mode)")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "edge.yaml"))
    ap.add_argument("--rules", default=str(DEFAULT_RULES),
                    help="rules file; with --tune-result it is the base the rule tuner started from")
    ap.add_argument("--tune-result", help="runtime/tuning/result.json from evaluation.tune: use the rules tuned "
                                          "on each fold (honest nested CV) and its final rules for deployment")
    ap.add_argument("--labels", default=str(REPO_ROOT / "data" / "labels" / "snatch1.0.csv"))
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--oof", default=str(OOF_DIR))
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--trials", type=int, default=600)
    ap.add_argument("--target", type=float, default=0.90)
    ap.add_argument("--level", choices=["any", "confirmed"], default="any")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR / "fusion.json"))
    ap.add_argument("--deploy-to", default=str(FUSION_TUNED))
    ap.add_argument("--external-labels", default=str(Path.home() / "snatch_web_dataset" / "labels.csv"))
    ap.add_argument("--external-data", default=str(Path.home() / "snatch_web_dataset" / "clips"))
    args = ap.parse_args()
    if not args.tune:
        ap.error("nothing to do: pass --tune")

    cfg = load_yaml(args.config)
    items, missing = collect(Path(args.labels), Path(args.data), cfg, args.fps)
    if missing:
        print(f"WARNING: {len(missing)} labelled videos have no perception cache: {missing[:5]}")
    if not items:
        raise SystemExit("no cached videos (run python -m snatchguard.evaluation.cache first)")
    base = load_rules(args.rules)
    if args.tune_result:
        tr = json.loads(Path(args.tune_result).read_text(encoding="utf-8"))
        fold_rules = {f: to_rules(base, tr["params_per_fold"][f]) for f in ("A", "B")}
        final_rules = to_rules(base, tr["params_final"])
        rules_note = f"per-fold rules from {args.tune_result} (nested cross-validation)"
    else:
        fold_rules = {"A": base, "B": base}
        final_rules = base
        rules_note = f"fixed rules {args.rules}"
        if "tuned" in Path(args.rules).name:
            rules_note += (" (WARNING: if these were tuned on folds A/B, the rules part of the cross-validated "
                           "numbers is optimistic; pass --tune-result for nested CV)")
    print(f"fusion tuning over {len(items)} videos, {rules_note}", flush=True)
    res = run(items, fold_rules, final_rules, args.trials, args.target, args.level, args.seed, Path(args.oof))
    res["rules"] = rules_note

    from snatchguard.evaluation.external import score_external
    ext_labels, ext_data = Path(args.external_labels), Path(args.external_data)
    if ext_labels.exists():
        params = {v: FusionParams.from_dict(res["variants"][v]["params_final"]) for v in res["variants"]}
        res["external"] = score_external(cfg, final_rules, ext_labels, ext_data, args.fps, args.level, params,
                                         Path(args.oof), build_missing=False)

    # Tuned thresholds only reach the live edge when they catch more events than the rules alone in
    # cross-validation; otherwise a search that met the precision target by switching paths off would gate
    # live rule alerts on a model that never earned it. A stale file belongs to an older model: remove it.
    deploy, v = Path(args.deploy_to), res["deployed_variant"]
    gain = (res["variants"][v]["cross_validated"]["events_detected"]
            - res["variants"]["rules"]["cross_validated"]["events_detected"])
    res["deployed"] = v == "rules" or gain > 0
    if res["deployed"]:
        write_deployed(res, deploy)
        note = f"wrote {deploy} (deployed: {v})"
    else:
        deploy.unlink(missing_ok=True)
        note = (f"NOT deployed: '{v}' caught no more events than the rules alone in cross-validation; "
                f"the live edge uses {FUSION_DEFAULT.name} (model shown as a risk score, alerts unchanged)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    for name, r in res["variants"].items():
        cv = r["cross_validated"]
        print(f"{name:6s} cross-validated: recall {cv['recall']:.3f} {cv['recall_ci95']}, precision "
              f"{cv['precision']:.3f} {cv['precision_ci95']}, {cv['false_alarms_per_hour']:.2f} FA/h")
    print(f"wrote {args.out}; {note}")


if __name__ == "__main__":
    main()
