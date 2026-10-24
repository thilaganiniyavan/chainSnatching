"""Build, evaluate and fit the two-stage verifier on real footage, with actor-grounded labels.

    # 1. stage-1 clips + actor-grounded labels from the perception caches, then video features (slow, resumable)
    python -m snatchguard.verifier.train features --out runtime/verifier/features.npz
    # 2. cross-dataset evaluation (Snatch 1.0 <-> Snatch-Web), final fit, model + report
    python -m snatchguard.verifier.train fit --features runtime/verifier/features.npz --model models/verifier/verifier.pt

Labels: a clip is positive only if it contains a contact between the labelled victim and attacker (track IDs in
data/labels/actors.json) inside the event window. Clips that touch a labelled actor at another moment, or fall
near an unclear event, are ignored rather than used as negatives. Every candidate on normal footage is a negative.

Ordinary clips are sampled (positives are all kept, with augmented copies for training only); false alarms per
hour are estimated by dividing the sampled count by the sampling fraction.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from pathlib import Path

import cv2
import numpy as np

from snatchguard.core.config import REPO_ROOT, load_yaml
from snatchguard.evaluation import cache as cache_mod
from snatchguard.evaluation.metrics import wilson
from snatchguard.verifier.candidates import CandidateStage
from snatchguard.verifier.model import NF, Encoder, Head, cut, sample_times

log = logging.getLogger(__name__)

S1_LABELS = REPO_ROOT / "data" / "labels" / "snatch1.0.csv"
S1_DATA = REPO_ROOT / "data" / "samples" / "snatch1.0" / "Chain Snatching Videos"
WEB = Path.home() / "snatch_web_dataset"
ACTORS = REPO_ROOT / "data" / "labels" / "actors.json"
NORMAL_MAX_S = 600.0
WINDOW_PAD_S = 1.5
NEAR_PAD_S = 3.0
# training-only augmentation of positive clips: (time shift s, horizontal flip, crop scale)
VARIANTS = [(0.0, False, 1.0), (-0.5, False, 1.0), (0.5, False, 1.0), (0.0, True, 1.0), (0.0, False, 0.8), (0.0, False, 1.3)]


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def datasets(web: Path) -> list[tuple[str, dict, Path]]:
    out = []
    for row in _rows(S1_LABELS):
        if row["category"] in ("snatch_bike", "snatch_foot") or (row["category"] == "normal" and row["fold"] in ("A", "B")):
            out.append(("S1", row, S1_DATA / row["video"]))
    if (web / "labels.csv").exists():
        for row in _rows(web / "labels.csv"):
            if row["category"] in ("snatch_bike", "snatch_foot", "normal"):
                out.append(("WEB", row, web / "clips" / row["video"]))
    return out


def label_clips(row: dict, clips, actors: dict | None) -> list[int]:
    """1 = the labelled victim-attacker contact in the event window, 0 = negative, -1 = ignore."""
    if row["category"] == "normal":
        return [0] * len(clips)
    if not row.get("event_start_s") or actors is None:
        return [-1] * len(clips)
    es, ee = float(row["event_start_s"]), float(row["event_end_s"] or row["event_start_s"])
    V, A, ok = set(actors["victim"]), set(actors["attacker"]), actors["status"] == "ok"
    ys = []
    for c in clips:
        hit = ok and any((a in V and b in A) or (b in V and a in A) for a, b in c.pairs if a != b)
        touches = any({a, b} & (V | A) for a, b in c.pairs)
        if hit and es - WINDOW_PAD_S <= c.t <= ee + WINDOW_PAD_S:
            ys.append(1)
        elif es - NEAR_PAD_S <= c.t <= ee + NEAR_PAD_S and (not ok or touches):
            ys.append(-1)
        elif touches and ok:
            ys.append(-1)
        else:
            ys.append(0)
    return ys


def build_clips(edge_cfg: str, web: Path, actors_path: Path) -> list[dict]:
    tag = cache_mod.perception_tag(load_yaml(edge_cfg), 8.0, None)
    actors = json.loads(actors_path.read_text())
    videos = []
    for setname, row, path in datasets(web):
        cp = cache_mod.cache_path(str(path), tag)
        if not cp.exists():
            log.warning("no perception cache for %s (run python -m snatchguard.evaluation.cache)", path)
            continue
        cv = cache_mod.load(cp)
        frames = cv.frames if row["category"] != "normal" else [f for f in cv.frames if f.t <= NORMAL_MAX_S]
        stage, clips = CandidateStage(), []
        for f in frames:
            clips += stage.step(f.t, f.detections)
        key = f"{setname}:{row['video']}"
        ys = label_clips(row, clips, actors.get(key))
        videos.append({"key": key, "set": setname, "path": str(path), "category": row["category"],
                       "duration": frames[-1].t if frames else 0.0,
                       "clips": [{"t": c.t, "crop": list(c.crop), "pairs": c.pairs, "y": y} for c, y in zip(clips, ys)]})
    return videos


def _extract(path: str, requests: list[tuple[float, tuple, float, bool, float]]) -> np.ndarray:
    """requests: (onset t, crop, time shift, flip, crop scale) -> (len, NF, SIDE, SIDE, 3) uint8, one decode pass."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    need: dict[int, list] = {}
    for i, (t0, crop, shift, flip, sc) in enumerate(requests):
        x0, y0, side = crop
        cx, cy = x0 + side / 2, y0 + side / 2
        crop2 = (cx - side * sc / 2, cy - side * sc / 2, side * sc)
        for slot, tt in enumerate(sample_times(t0, shift)):
            need.setdefault(int(round(max(tt, 0.0) * fps)), []).append((i, slot, crop2, flip))
    out = np.zeros((len(requests), NF, 112, 112, 3), np.uint8)
    have = np.zeros((len(requests), NF), bool)
    fi, last, scale = -1, max(need) if need else -1, None
    while fi < last:
        if not cap.grab():
            break
        fi += 1
        if fi not in need:
            continue
        ok, img = cap.retrieve()
        if not ok:
            continue
        if scale is None:
            scale = min(1.0, 1280 / img.shape[1])
        for i, slot, crop, flip in need[fi]:
            out[i, slot] = cut(img, crop, 1.0 / scale, flip)
            have[i, slot] = True
    for i in range(len(requests)):          # past the end of the video: repeat the last frame
        for s in range(1, NF):
            if not have[i, s] and have[i, s - 1]:
                out[i, s], have[i, s] = out[i, s - 1], True
    return out


def cmd_features(args) -> None:
    rng = random.Random(args.seed)
    videos = build_clips(args.config, Path(args.web), Path(args.actors))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).with_suffix(".clips.json").write_text(json.dumps(videos))
    pos, norm, evneg = [], {"S1": [], "WEB": []}, []
    for vi, v in enumerate(videos):
        for ci, c in enumerate(v["clips"]):
            if c["y"] == 1:
                pos.append((vi, ci))
            elif c["y"] == 0 and v["category"] == "normal":
                norm[v["set"]].append((vi, ci))
            elif c["y"] == 0:
                evneg.append((vi, ci))
    frac, pick = {}, [(vi, ci, var) for vi, ci in pos for var in range(len(VARIANTS))]
    for s, n in (("S1", args.n_s1), ("WEB", args.n_web)):
        sel = rng.sample(norm[s], min(n, len(norm[s])))
        frac[s] = len(sel) / max(len(norm[s]), 1)
        pick += [(vi, ci, 0) for vi, ci in sel]
    pick += [(vi, ci, 0) for vi, ci in rng.sample(evneg, min(args.n_event, len(evneg)))]
    by_video: dict[int, list] = {}
    for vi, ci, var in pick:
        by_video.setdefault(vi, []).append((ci, var))
    log.info("encoding %d clips from %d videos (%d positives x %d variants); normal sample fractions %s",
             len(pick), len(by_video), len(pos), len(VARIANTS), {k: round(v, 3) for k, v in frac.items()})
    enc = Encoder(threads=args.threads)
    F, VI, CI, VAR = [], [], [], []
    t_start = time.time()
    for n, (vi, items) in enumerate(sorted(by_video.items())):
        v = videos[vi]
        reqs = [(v["clips"][ci]["t"], tuple(v["clips"][ci]["crop"]), *VARIANTS[var]) for ci, var in items]
        t1 = time.time()
        clips = _extract(v["path"], reqs)
        t2 = time.time()
        F.append(enc.encode(clips))
        VI += [vi] * len(items)
        CI += [ci for ci, _ in items]
        VAR += [var for _, var in items]
        log.info("  %3d/%d %-40s %4d clips: decode %4.0fs, encode %4.0fs (total %5.0fs)", n + 1, len(by_video),
                 v["key"][:40], len(items), t2 - t1, time.time() - t2, time.time() - t_start)
    np.savez_compressed(args.out, feats=np.concatenate(F), video=np.array(VI), clip=np.array(CI), variant=np.array(VAR),
                        frac_s1=frac["S1"], frac_web=frac["WEB"])
    log.info("wrote %s", args.out)


# ---------------------------------------------------------------------------
# Snatch 1.0 normal videos that come from the same physical camera (read off the burned-in camera names); they are
# always held out together so a camera is never both trained and tested on.
S1_CAMERAS = [{"s_2", "s_3"}, {"s_27"}, {"s_29"}, {"s_19"}, {"s_21"}, {"s_23", "s_25"}, {"s_5"}, {"s_17"},
              {"s_7", "s_11", "s_13", "s_15"}]


def group_of(video: dict) -> str:
    """Cross-validation group: the physical camera for Snatch 1.0 normals, the source upload for Snatch-Web."""
    name = video["key"].split(":", 1)[1]
    if video["set"] == "S1" and video["category"] == "normal":
        stem = Path(name).stem
        for i, cams in enumerate(S1_CAMERAS):
            if stem in cams:
                return f"S1cam{i}"
        return f"S1:{stem}"
    if video["set"] == "WEB":
        return "WEB:" + Path(name).stem.rsplit("_", 1)[0]      # <youtube id>_<segment> -> youtube id
    return video["key"]


def _events(videos: list[dict], setname: str, actors: dict) -> set[str]:
    """Clearly labelled events of a set (an untracked actor still counts: the system must miss it)."""
    return {v["key"] for v in videos if v["set"] == setname and v["category"] != "normal"
            and actors.get(v["key"], {}).get("status") in ("ok", "untracked")}


def operating_curve(scores, keys, ys, normal, frac: float, hours: float, events: set[str]):
    """(false alarms per hour, events caught) as the threshold sweeps over the scored clips."""
    order = np.argsort(-scores)
    pts, caught, fa = [], set(), 0
    for i in order:
        if normal[i]:
            fa += 1
        elif ys[i] == 1 and keys[i] in events:
            caught.add(keys[i])
        pts.append((fa / frac / hours, len(caught), float(scores[i])))
    return pts


def _envelope(pts):
    """Upper envelope of an operating curve: the first point at each new number of events caught."""
    out = []
    for fa, k, _ in sorted(pts, key=lambda p: (p[0], -p[1])):
        if not out or k > out[-1][1]:
            out.append((fa, k))
    return out


def best_at(pts, budget: float):
    ok = [p for p in pts if p[0] <= budget]
    return max(ok, key=lambda p: (p[1], -p[0])) if ok else (0.0, 0, 1.0)


def cmd_fit(args) -> None:
    D = np.load(args.features)
    videos = json.loads(Path(args.features).with_suffix(".clips.json").read_text())
    actors = json.loads(Path(args.actors).read_text())
    F, VI, CI, VAR = D["feats"], D["video"], D["clip"], D["variant"]
    frac = {"S1": float(D["frac_s1"]), "WEB": float(D["frac_web"])}
    sets = np.array([videos[i]["set"] for i in VI])
    ys = np.array([videos[i]["clips"][c]["y"] for i, c in zip(VI, CI)])
    keys = np.array([videos[i]["key"] for i in VI])
    normal = np.array([videos[i]["category"] == "normal" for i in VI])
    hours = {s: sum(v["duration"] for v in videos if v["set"] == s and v["category"] == "normal") / 3600 for s in frac}
    clips_per_h = {s: sum(len(v["clips"]) for v in videos if v["set"] == s and v["category"] == "normal") / hours[s]
                   for s in frac}
    report = {"hours": hours, "candidate_clips_per_hour": clips_per_h, "sample_fraction": frac, "cross": {}}
    curves = {}
    for tr, te in (("S1", "WEB"), ("WEB", "S1")):
        trm = (sets == tr) & (ys >= 0)
        tem = (sets == te) & (ys >= 0) & (VAR == 0)
        head = Head.fit(F[trm], ys[trm], n_components=args.pca, C=args.C)
        s = head.proba(F[tem])
        ev = _events(videos, te, actors)
        pts = operating_curve(s, keys[tem], ys[tem], normal[tem], frac[te], hours[te], ev)
        curves[te] = pts
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(ys[tem], s)) if len(set(ys[tem])) == 2 else float("nan")
        budgets = {}
        for b in args.budgets:
            fa, k, thr = best_at(pts, b)
            lo, hi = wilson(k, len(ev))
            budgets[str(b)] = {"false_alarms_per_h": round(fa, 2), "caught": k, "events": len(ev),
                               "recall": round(k / max(len(ev), 1), 3), "recall_ci": [round(lo, 3), round(hi, 3)]}
        report["cross"][f"{tr}->{te}"] = {"clip_auc": round(auc, 3), "at_budget": budgets,
                                         "positives_train": int((ys[trm] == 1).sum()), "events_test": len(ev)}
        log.info("train %s -> test %s: clip AUC %.3f  %s", tr, te, auc,
                 "  ".join(f"<={b}/h: {v['caught']}/{v['events']} ({v['false_alarms_per_h']}/h)" for b, v in budgets.items()))
    # pooled 5-fold cross-validation holding out whole cameras / source uploads
    from sklearn.model_selection import GroupKFold
    groups = np.array([group_of(videos[i]) for i in VI])
    oof = np.full(len(F), np.nan)
    usable = ys >= 0
    for trn, tst in GroupKFold(n_splits=5).split(F, ys, groups):
        trn = trn[usable[trn]]
        tst = tst[usable[tst] & (VAR[tst] == 0)]
        if len(set(ys[trn])) < 2:
            continue
        oof[tst] = Head.fit(F[trn], ys[trn], n_components=args.pca, C=args.C).proba(F[tst])
    report["pooled_cv"] = {}
    for te in ("S1", "WEB"):
        m = (sets == te) & ~np.isnan(oof)
        ev = _events(videos, te, actors)
        pts = operating_curve(oof[m], keys[m], ys[m], normal[m], frac[te], hours[te], ev)
        curves[f"cv-{te}"] = pts
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(ys[m], oof[m])) if len(set(ys[m])) == 2 else float("nan")
        budgets = {}
        for b in args.budgets:
            fa, k, _ = best_at(pts, b)
            lo, hi = wilson(k, len(ev))
            budgets[str(b)] = {"false_alarms_per_h": round(fa, 2), "caught": k, "events": len(ev),
                               "recall": round(k / max(len(ev), 1), 3), "recall_ci": [round(lo, 3), round(hi, 3)]}
        report["pooled_cv"][te] = {"clip_auc": round(auc, 3), "at_budget": budgets,
                                   "curve": [(round(p[0], 2), p[1]) for p in _envelope(pts)]}
        log.info("pooled CV, %s: clip AUC %.3f  %s", te, auc,
                 "  ".join(f"<={b}/h: {v['caught']}/{v['events']} ({v['false_alarms_per_h']}/h)" for b, v in budgets.items()))
    for k in ("S1", "WEB"):
        report["cross"][f"{'WEB' if k == 'S1' else 'S1'}->{k}"]["curve"] = [(round(p[0], 2), p[1]) for p in _envelope(curves[k])]

    # final model on everything; threshold = the score that meets the budget in the pooled-CV curves
    allm = ys >= 0
    head = Head.fit(F[allm], ys[allm], n_components=args.pca, C=args.C)
    thr = float(np.mean([best_at(curves[f"cv-{s}"], args.alert_budget)[2] for s in ("S1", "WEB")]))
    head.threshold = thr
    head.meta = {"trained": time.strftime("%Y-%m-%d %H:%M"), "positives": int((ys[allm] == 1).sum()),
                 "negatives": int((ys[allm] == 0).sum()), "alert_budget_per_h": args.alert_budget,
                 "report": report}
    head.save(args.model)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=1))
    log.info("model %s (threshold %.4f), report %s", args.model, thr, args.report)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("--config", default=str(REPO_ROOT / "configs" / "edge.yaml"))
    f.add_argument("--web", default=str(WEB))
    f.add_argument("--actors", default=str(ACTORS))
    f.add_argument("--out", default=str(REPO_ROOT / "runtime" / "verifier" / "features.npz"))
    f.add_argument("--n-s1", type=int, default=1200, help="normal Snatch 1.0 clips to sample")
    f.add_argument("--n-web", type=int, default=600, help="normal Snatch-Web clips to sample")
    f.add_argument("--n-event", type=int, default=400, help="non-event clips from snatch videos to sample")
    f.add_argument("--threads", type=int, default=4)
    f.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("fit")
    p.add_argument("--features", default=str(REPO_ROOT / "runtime" / "verifier" / "features.npz"))
    p.add_argument("--actors", default=str(ACTORS))
    p.add_argument("--model", default=str(REPO_ROOT / "models" / "verifier" / "verifier.pt"))
    p.add_argument("--report", default=str(REPO_ROOT / "runtime" / "verifier" / "report.json"))
    p.add_argument("--pca", type=int, default=32)
    p.add_argument("--C", type=float, default=0.05)
    p.add_argument("--budgets", type=float, nargs="+", default=[1, 10, 100, 1000])
    p.add_argument("--alert-budget", type=float, default=10.0, help="false alarms per hour the threshold targets")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    {"features": cmd_features, "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    main()
