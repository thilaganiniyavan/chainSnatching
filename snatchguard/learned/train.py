"""Train the learned scorer, write out-of-fold window scores and the report.

    python -m snatchguard.learned.train [--device cpu|0] [--normal-max-seconds 600]

* snatch_ab:    fit on Snatch 1.0 fold A, score fold B, and the other way round (the rules' folds).
* deploy:       fit on A+B -> models/learned/snatch_scorer.pt
* web_external: the deployed model scores every web-dataset video. Nothing is trained or tuned on it.

Every choice (regularisation C, Platt calibration, model-only alert threshold and run length) is made by
inner cross-validation over the *training* videos only (whole videos per inner fold).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from snatchguard.core.config import REPO_ROOT
from snatchguard.evaluation.metrics import Metrics, event_window, score, wilson
from snatchguard.learned import extract as ex
from snatchguard.learned.features import FeatureSpec, windows_for
from snatchguard.learned.scorer import Head, alerts_from_windows

OOF_DIR = REPO_ROOT / "runtime" / "learned" / "oof"
MODEL_PATH = REPO_ROOT / "models" / "learned" / "snatch_scorer.pt"
REPORT = REPO_ROOT / "runtime" / "learned" / "report.md"
WEB = Path.home() / "snatch_web_dataset"

WINDOW_S, STRIDE_S = 2.0, 0.5
K_PCA = 32
C_GRID = (0.003, 0.03, 0.3)
POS_MIN_OVERLAP_S = 0.5     # a window is positive when it covers this much of the grab (or half of a shorter grab)
NEG_GAP_S = 3.0             # snatch-clip windows this far from the grab are negatives; in between: ignored
NORMAL_TRAIN_EVERY = 4      # train on every 4th normal window (2 s apart); all windows are scored
PCA_MAX_ROWS = 20000
TARGET = 0.90


@dataclass
class Video:
    row: dict
    path: Path
    duration_s: float
    cap_s: float | None = None
    spans: list = field(default_factory=list)
    X: np.ndarray | None = None
    y: np.ndarray | None = None

    @property
    def name(self) -> str:
        return self.row["video"]

    @property
    def stem(self) -> str:
        return ex.video_stem(self.row["video"])

    @property
    def is_normal(self) -> bool:
        return self.row["category"] == "normal"


# ---- data -----------------------------------------------------------------------------------------

def load_videos(labels: Path, folds: set[str], spec: FeatureSpec, normal_max_seconds: float | None) -> list[Video]:
    out, missing = [], []
    for r in ex.select_rows(labels, folds):
        p = ex.feature_path(r["video"], spec)
        if not p.exists():
            missing.append(r["video"])
            continue
        cap = normal_max_seconds if r["category"] == "normal" else None
        dur = float(ex.load_meta(p)["duration_s"])
        out.append(Video(r, p, min(dur, cap) if cap else dur, cap))
    if missing:
        raise FileNotFoundError(f"no features for {len(missing)} videos (run snatchguard.learned.extract): "
                                + ", ".join(missing[:5]))
    return out


def window_labels(spans: list[tuple[float, float]], row: dict) -> np.ndarray:
    """1 = grab inside, 0 = negative, -1 = not used for training."""
    if row["category"] == "normal":
        return np.zeros(len(spans), np.int8)
    if not row.get("event_start_s"):
        return np.full(len(spans), -1, np.int8)
    start = float(row["event_start_s"])
    end = float(row["event_end_s"] or start)
    need = min(POS_MIN_OVERLAP_S, max(end - start, 0.1) / 2)
    y = np.full(len(spans), -1, np.int8)
    for i, (s, e) in enumerate(spans):
        if min(e, end) - max(s, start) >= need:
            y[i] = 1
        elif e <= start - NEG_GAP_S or s >= end + NEG_GAP_S:
            y[i] = 0
    return y


def fit_pca(videos: list[Video], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA

    metas = [(v, len(ex.load_meta(v.path)["t"])) for v in videos]
    total = sum(n for _, n in metas) or 1
    rows = []
    for v, n in metas:
        d = ex.load_features(v.path)
        emb, t = d["emb"], d["t"]
        if v.cap_s:
            emb = emb[t <= v.cap_s]
        take = max(1, int(PCA_MAX_ROWS / emb.shape[1] * len(emb) / total))
        idx = np.linspace(0, len(emb) - 1, min(take, len(emb))).astype(int) if len(emb) else []
        if len(idx):
            rows.append(emb[idx].reshape(-1, emb.shape[2]).astype(np.float32))
    pca = PCA(n_components=K_PCA, random_state=seed).fit(np.concatenate(rows))
    return pca.mean_.astype(np.float32), pca.components_.astype(np.float32)


def compute_windows(videos: list[Video], pca_mean: np.ndarray, pca_comp: np.ndarray) -> None:
    for v in videos:
        d = ex.load_features(v.path)
        t, emb, mot = d["t"], d["emb"], d["motion"]
        if v.cap_s:
            keep = t <= v.cap_s
            t, emb, mot = t[keep], emb[keep], mot[keep]
        z = np.empty((len(t), emb.shape[1], K_PCA), np.float32)
        for i in range(0, len(t), 2000):
            z[i:i + 2000] = (emb[i:i + 2000].astype(np.float32) - pca_mean) @ pca_comp.T
        v.spans, v.X = windows_for(t, z, mot, WINDOW_S, STRIDE_S, final=True)
        v.y = window_labels(v.spans, v.row)


def training_set(videos: list[Video]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for v in videos:
        idx = np.nonzero(v.y >= 0)[0]
        if v.is_normal:
            idx = idx[::NORMAL_TRAIN_EVERY]
        xs.append(v.X[idx])
        ys.append(v.y[idx])
    return np.concatenate(xs), np.concatenate(ys).astype(int)


# ---- head -----------------------------------------------------------------------------------------

@dataclass
class Linear:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def decision(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.scale) @ self.coef + self.intercept


def fit_linear(X: np.ndarray, y: np.ndarray, C: float) -> Linear:
    from sklearn.linear_model import LogisticRegression

    mean = X.mean(0)
    scale = X.std(0)
    scale[scale < 1e-6] = 1.0
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit((X - mean) / scale, y)
    return Linear(mean.astype(np.float32), scale.astype(np.float32), lr.coef_[0].astype(np.float32),
                  float(lr.intercept_[0]))


def inner_folds(videos: list[Video], n: int, seed: int) -> list[int]:
    """Whole videos per fold, event clips and normal videos spread evenly."""
    rng = random.Random(seed)
    fold = [0] * len(videos)
    for group in ([i for i, v in enumerate(videos) if not v.is_normal], [i for i, v in enumerate(videos) if v.is_normal]):
        rng.shuffle(group)
        for j, i in enumerate(group):
            fold[i] = j % n
    return fold


def inner_traces(videos: list[Video], C: float, n: int, seed: int) -> list[np.ndarray]:
    fold = inner_folds(videos, n, seed)
    traces: list[np.ndarray | None] = [None] * len(videos)
    for k in range(n):
        train = [v for v, f in zip(videos, fold) if f != k]
        X, y = training_set(train)
        if y.min() == y.max():
            continue
        lin = fit_linear(X, y, C)
        for i, v in enumerate(videos):
            if fold[i] == k:
                traces[i] = lin.decision(v.X)
    return [t if t is not None else np.full(len(v.spans), -10.0) for t, v in zip(traces, videos)]


def fit_platt(dec: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1.0, max_iter=1000).fit(dec.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def model_alert_results(videos: list[Video], probs: list[np.ndarray], thr, min_run) -> dict:
    """metrics.score input. thr/min_run: one value, or one per video."""
    res = {}
    for i, (v, p) in enumerate(zip(videos, probs)):
        th = thr[i] if isinstance(thr, (list, tuple)) else thr
        mr = min_run[i] if isinstance(min_run, (list, tuple)) else min_run
        wins = [(s, e, float(x)) for (s, e), x in zip(v.spans, p)]
        res[v.name] = (v.row, v.duration_s, alerts_from_windows(wins, th, mr))
    return res


def choose_threshold(videos: list[Video], probs: list[np.ndarray]) -> tuple[float, int, dict]:
    """Max recall with precision >= TARGET (event level); otherwise max F1. Training videos only.
    Among equally good thresholds take the middle of the range (largest margin), not its edge."""
    allp = np.concatenate(probs)
    cands = set(np.quantile(allp, np.linspace(0.90, 0.99999, 120)).round(6).tolist())
    cands |= {round(float(p.max()), 6) for p in probs if len(p)}
    scored = []
    for thr in sorted(cands):
        for mr in (1, 2):
            m = score(model_alert_results(videos, probs, thr, mr))
            meets = m.precision >= TARGET and m.events_detected > 0
            scored.append(((meets, m.recall if meets else m.f1, m.precision), thr, mr, m))
    best_key = max(k for k, *_ in scored)
    tied: dict[int, list] = {}
    for k, thr, mr, m in scored:
        if k == best_key:
            tied.setdefault(mr, []).append((thr, m))
    mr = max(tied, key=lambda r: (len(tied[r]), -r))
    thr, m = tied[mr][len(tied[mr]) // 2]
    return thr, mr, m.summary()


def fit_head(train: list[Video], spec: FeatureSpec, pca: tuple[np.ndarray, np.ndarray], seed: int,
             label: str) -> tuple[Head, dict]:
    from sklearn.metrics import average_precision_score

    n_events = sum(1 for v in train if not v.is_normal)
    n = max(2, min(5, n_events, sum(1 for v in train if v.is_normal)))
    lab = np.concatenate([v.y for v in train])
    mask = lab >= 0
    ap_by_c, traces_by_c = {}, {}
    for C in C_GRID:
        tr = inner_traces(train, C, n, seed)
        traces_by_c[C] = tr
        ap_by_c[C] = float(average_precision_score(lab[mask], np.concatenate(tr)[mask]))
    C = max(C_GRID, key=lambda c: ap_by_c[c])
    dec = np.concatenate(traces_by_c[C])
    a, b = fit_platt(dec[mask], lab[mask])
    probs = [1 / (1 + np.exp(-(a * d + b))) for d in traces_by_c[C]]
    thr, min_run, inner = choose_threshold(train, probs)
    X, y = training_set(train)
    lin = fit_linear(X, y, C)
    head = Head(spec, WINDOW_S, STRIDE_S, pca[0], pca[1], lin.mean, lin.scale, lin.coef, lin.intercept, a, b,
                thr, min_run, {"trained_on": label, "C": C, "inner_ap": ap_by_c, "inner_event_metrics": inner,
                               "n_train_windows": int(len(y)), "n_pos_windows": int(y.sum()),
                               "created": time.strftime("%Y-%m-%d %H:%M")})
    return head, {"C": C, "inner_ap": ap_by_c, "threshold": thr, "min_run": min_run, "inner": inner,
                  "n_train_windows": int(len(y)), "n_pos_windows": int(y.sum()),
                  "n_train_videos": len(train)}


# ---- outputs --------------------------------------------------------------------------------------

def write_oof(protocol: str, videos: list[Video], probs: list[np.ndarray], heads: list[Head],
              oof_dir: Path = OOF_DIR) -> None:
    d = oof_dir / protocol
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.json"):
        old.unlink()
    for v, p, h in zip(videos, probs, heads):
        (d / f"{v.stem}.json").write_text(json.dumps({
            "video": v.name, "protocol": protocol, "category": v.row["category"], "fold": v.row.get("fold", ""),
            "lighting": v.row.get("lighting", ""), "duration_s": round(v.duration_s, 3),
            "threshold": h.threshold, "min_run": h.min_run, "window_s": h.window_s, "stride_s": h.stride_s,
            "windows": [[s, e, round(float(x), 5)] for (s, e), x in zip(v.spans, p)],
        }), encoding="utf-8")


def window_diagnostics(videos: list[Video], probs: list[np.ndarray]) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.concatenate([v.y for v in videos])
    p = np.concatenate(probs)
    m = y >= 0
    if m.sum() == 0 or y[m].min() == y[m].max():
        return {}
    return {"window_auc": round(float(roc_auc_score(y[m], p[m])), 3),
            "window_ap": round(float(average_precision_score(y[m], p[m])), 3),
            "pos_windows": int((y == 1).sum()), "neg_windows": int((y == 0).sum())}


def clip_rows(videos: list[Video], probs: list[np.ndarray], m: Metrics, heads: list[Head]) -> list[dict]:
    outcome = {o.video: o for o in m.outcomes}
    rows = []
    for v, p, h in zip(videos, probs, heads):
        o = outcome.get(v.name)
        if o is None:
            continue
        win = event_window(v.row, v.duration_s)
        inside = np.array([e >= win[0] and s <= win[1] for s, e in v.spans], bool)
        rows.append({
            "video": v.name, "category": v.row["category"], "fold": v.row.get("fold", ""),
            "lighting": v.row.get("lighting", ""), "caught": o.detected if not v.is_normal else None,
            "true_alerts": o.true_alerts, "false_alarms": o.false_alarms, "threshold": h.threshold,
            "max": float(p.max()) if len(p) else 0.0,
            "max_in_event": float(p[inside].max()) if (not v.is_normal and inside.any()) else None,
            "max_outside": float(p[~inside].max()) if (not v.is_normal and (~inside).any()) else None,
            "argmax_t": v.spans[int(p.argmax())][0] if len(p) else None,
            "peak_in_event": bool(inside[int(p.argmax())]) if (not v.is_normal and len(p)) else None,
            "timed": bool(v.row.get("event_start_s")),
        })
    return rows


def _ci(ci) -> str:
    return f"{ci[0]:.0%}–{ci[1]:.0%}"


def metrics_block(name: str, m: Metrics, diag: dict) -> list[str]:
    n_alerts = m.true_alerts + m.false_alarms
    prec = (f"**{m.precision:.1%}** ({m.true_alerts}/{n_alerts}, 95% CI {_ci(m.precision_ci)})" if n_alerts
            else "n/a (no alerts)")
    out = [f"### {name}", "",
           "| recall (events) | precision (alerts) | false alarms / h of normal | normal footage | window AUC / AP |",
           "|---|---|---|---|---|",
           f"| **{m.recall:.1%}** ({m.events_detected}/{m.events}, 95% CI {_ci(m.recall_ci)}) "
           f"| {prec} | {m.false_alarms_per_hour:.2f} ({m.normal_false_alarms} alarms) | {m.normal_hours:.2f} h "
           f"| {diag.get('window_auc', '–')} / {diag.get('window_ap', '–')} |", ""]
    cats = {}
    for o in m.outcomes:
        if o.category == "normal":
            continue
        c = cats.setdefault(o.category, [0, 0, 0])
        c[0] += 1
        c[1] += o.detected
        c[2] += o.false_alarms
    out += ["| category | events | caught | recall (95% CI) | false alarms inside clips |", "|---|---|---|---|---|"]
    for c, (n, k, fa) in sorted(cats.items()):
        out.append(f"| {c} | {n} | {k} | {k / n:.0%} ({_ci(wilson(k, n))}) | {fa} |")
    return out + [""]


def clip_table(rows: list[dict]) -> list[str]:
    out = ["| video | category | fold | light | caught | true / false alerts | threshold | max score | "
           "max in event window | max outside | peak at (s) | peak in event |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: "–" if x is None else f"{x:.3f}"
    for r in sorted(rows, key=lambda r: (r["category"] == "normal", r["category"], r["video"])):
        caught = "–" if r["caught"] is None else ("yes" if r["caught"] else "**missed**")
        pin = "–" if r["peak_in_event"] is None else ("yes" if r["peak_in_event"] else "no")
        out.append(f"| {r['video']} | {r['category']} | {r['fold']} | {r['lighting']} | {caught} | "
                   f"{r['true_alerts']} / {r['false_alarms']} | {r['threshold']:.3f} | {f(r['max'])} | "
                   f"{f(r['max_in_event'])} | {f(r['max_outside'])} | {r['argmax_t']} | {pin} |")
    return out + [""]


def miss_summary(rows: list[dict]) -> list[str]:
    misses = [r for r in rows if r["caught"] is False]
    if not misses:
        return ["No missed events.", ""]
    below = sum(1 for r in misses if (r["max_in_event"] or 0) < r["threshold"])
    out = [f"{len(misses)} missed: in {below} the score inside the event window never reached the threshold; "
           f"in {len(misses) - below} it did, but the run was shorter than `min_run` windows.", ""]
    timed = [r for r in misses if r["timed"]]
    if timed:
        loc = sum(1 for r in timed if r["peak_in_event"])
        out.append(f"Of the {len(timed)} timed misses, the clip's highest score was inside the event window in {loc} "
                   f"(the model ranks the grab above the rest of the clip, but not above normal footage).")
        out.append("")
    for r in misses:
        inside = "–" if r["max_in_event"] is None else f"{r['max_in_event']:.3f}"
        out.append(f"* `{r['video']}` ({r['category']}): max in event window {inside} vs threshold "
                   f"{r['threshold']:.3f}; clip peak {r['max']:.3f} at {r['argmax_t']} s")
    return out + [""]


def throughput(videos: list[Video], spec: FeatureSpec) -> str:
    """Extraction speed = live speed (same featurizer, every frame decoded); the head costs microseconds."""
    rates = [float(ex.load_meta(v.path)["build_samples_per_s"]) for v in videos]
    if not rates:
        return ""
    med = float(np.median(rates))
    per_window = STRIDE_S * spec.sample_fps
    return (f"median {med:.2f} samples/s over {len(rates)} videos = {med / per_window:.2f} windows/s "
            f"({med / spec.sample_fps:.2f}× real time; real time needs {1 / STRIDE_S:.0f} windows/s).")


# ---- main -----------------------------------------------------------------------------------------

def run(labels: Path, data: Path, web_labels: Path | None, web_data: Path | None, device: str, threads: int,
        normal_max_seconds: float | None, do_extract: bool, model_out: Path, oof_dir: Path, report: Path,
        seed: int = 0) -> dict:
    spec = FeatureSpec()
    if do_extract:
        ex.run(labels, data, {"A", "B"}, spec, device, threads, normal_max_seconds)
    videos = load_videos(labels, {"A", "B"}, spec, normal_max_seconds)
    folds = {f: [v for v in videos if v.row["fold"] == f] for f in ("A", "B")}
    summary: dict = {"spec": spec.to_dict(), "window_s": WINDOW_S, "stride_s": STRIDE_S,
                     "normal_max_seconds": normal_max_seconds, "throughput": throughput(videos, spec)}

    # 1. snatch_ab: out-of-fold
    oof_videos, oof_probs, oof_heads, fold_info = [], [], [], {}
    for test_fold, train_fold in (("B", "A"), ("A", "B")):
        train, test = folds[train_fold], folds[test_fold]
        pca = fit_pca(train, seed)
        compute_windows(train + test, *pca)
        head, info = fit_head(train, spec, pca, seed, f"snatch1.0 fold {train_fold}")
        fold_info[f"train_{train_fold}_test_{test_fold}"] = info
        for v in test:
            oof_videos.append(v)
            oof_probs.append(head.proba(v.X))
            oof_heads.append(head)
        print(f"fold {train_fold}->{test_fold}: C={info['C']} thr={info['threshold']:.4f} "
              f"min_run={info['min_run']} inner={info['inner']}", flush=True)
    write_oof("snatch_ab", oof_videos, oof_probs, oof_heads, oof_dir)
    m_ab = score(model_alert_results(oof_videos, oof_probs, [h.threshold for h in oof_heads],
                                     [h.min_run for h in oof_heads]))
    diag_ab = window_diagnostics(oof_videos, oof_probs)
    rows_ab = clip_rows(oof_videos, oof_probs, m_ab, oof_heads)
    summary["snatch_ab"] = {**m_ab.summary(), **diag_ab, "folds": fold_info}
    print(f"snatch_ab: {m_ab.summary()} {diag_ab}", flush=True)

    # 2. deploy on A+B
    pca = fit_pca(videos, seed)
    compute_windows(videos, *pca)
    head, info = fit_head(videos, spec, pca, seed, "snatch1.0 folds A+B")
    head.save(model_out)
    summary["deploy"] = info
    print(f"deployed {model_out}: C={info['C']} thr={info['threshold']:.4f} min_run={info['min_run']}", flush=True)
    for v in videos:
        v.X = None

    # 3. web_external
    if web_labels and web_labels.exists() and web_data and web_data.exists():
        if do_extract:
            ex.run(web_labels, web_data, {"A", "B"}, spec, device, threads, None)
        web = load_videos(web_labels, {"A", "B"}, spec, None)
        compute_windows(web, head.pca_mean, head.pca_components)
        probs = [head.proba(v.X) for v in web]
        write_oof("web_external", web, probs, [head] * len(web), oof_dir)
        m_web = score(model_alert_results(web, probs, head.threshold, head.min_run))
        diag_web = window_diagnostics(web, probs)
        rows_web = clip_rows(web, probs, m_web, [head] * len(web))
        summary["web_external"] = {**m_web.summary(), **diag_web}
        print(f"web_external: {m_web.summary()} {diag_web}", flush=True)
    else:
        m_web = rows_web = diag_web = None
        print("web dataset not found: skipping web_external", flush=True)

    write_report(report, summary, m_ab, diag_ab, rows_ab, m_web, diag_web, rows_web, normal_max_seconds)
    (report.parent / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    return summary


def write_report(path: Path, summary: dict, m_ab, diag_ab, rows_ab, m_web, diag_web, rows_web,
                 normal_max_seconds) -> None:
    spec = summary["spec"]
    dep = summary["deploy"]
    normals = ("the first {:.0f} min of each normal video".format(normal_max_seconds / 60)
               if normal_max_seconds else "the whole of every normal video")
    lines = [
        "# Learned snatch scorer: measured results", "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Model-only alerts (no rules): runs of consecutive "
        f"{summary['window_s']:g} s windows (stride {summary['stride_s']:g} s) with calibrated score ≥ threshold, "
        "merged into one SUSPECT alert. Scored with `snatchguard.evaluation.metrics.score` (event window = grab "
        "± 1.5 s; every alert on normal footage or at the wrong moment counts as false).", "",
        f"* Backbone: frozen torchvision `{spec['backbone']}` (ImageNet), {spec['input_px']} px, on the whole frame "
        f"plus {spec['tiles']}×{spec['tiles']} overlapping tiles ({spec['tile_frac']:.0%} of the frame each), every "
        f"frame first downscaled to a {spec['max_side']} px long side; {spec['sample_fps']:g} samples/s.",
        "* Window features: only how the picture changes inside the window (embedding spread, jumps and drift after "
        f"PCA to {K_PCA} dims; frame-difference motion energy), never its average appearance.",
        "* Head: standardise → L2 logistic regression (balanced classes) → Platt calibration. C, calibration, "
        "threshold and run length chosen by inner cross-validation over whole training videos.",
        f"* Normal footage used: {normals}.",
        f"* Speed on this laptop's CPU (2 threads, shared with other jobs): {summary.get('throughput') or '–'}", "",
        "## 1. Snatch 1.0, cross-validated (`snatch_ab`: trained on fold A, scored on B, and the reverse)", "",
        "> Caveat: in Snatch 1.0 the snatch clips are short, low-resolution clips from many sources, while the normal "
        "videos are long recordings from a few Hyderabad cameras. Both folds share those cameras, so this number can "
        "be inflated by the model recognising the source rather than the event. Section 2 is the honest test.", "",
        *metrics_block("Out-of-fold", m_ab, diag_ab),
        "Thresholds per fold (chosen on the training fold only): " + ", ".join(
            f"{k}: C={v['C']}, threshold {v['threshold']:.4f}, min run {v['min_run']}, "
            f"{v['n_pos_windows']} positive / {v['n_train_windows']} training windows"
            for k, v in summary["snatch_ab"]["folds"].items()), "",
        "#### Every clip", "", *clip_table(rows_ab),
        "#### Misses", "", *miss_summary(rows_ab),
    ]
    if m_web is not None:
        lines += [
            "## 2. External test: Snatch-Web (different cameras, cities and sources; never trained or tuned on)", "",
            f"Deployed model (trained on Snatch 1.0 A+B): C={dep['C']}, threshold {dep['threshold']:.4f}, "
            f"min run {dep['min_run']}.", "",
            *metrics_block("Snatch-Web", m_web, diag_web),
            "#### Every clip", "", *clip_table(rows_web),
            "#### Misses", "", *miss_summary(rows_web),
        ]
    lines += ["## Files", "",
              "* `models/learned/snatch_scorer.pt` — deployed model (Snatch 1.0 A+B).",
              "* `runtime/learned/oof/snatch_ab/*.json`, `runtime/learned/oof/web_external/*.json` — window scores "
              "for fusion (`[t_start, t_end, score]`).",
              "* `runtime/learned/summary.json` — the numbers above as JSON.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the learned snatch scorer (cross-validated) and deploy it")
    ap.add_argument("--labels", type=Path, default=ex.DEFAULT_LABELS)
    ap.add_argument("--data", type=Path, default=ex.DEFAULT_DATA)
    ap.add_argument("--web-labels", type=Path, default=WEB / "labels.csv")
    ap.add_argument("--web-data", type=Path, default=WEB / "clips")
    ap.add_argument("--device", default="cpu", help="cpu or a CUDA index such as 0 (feature extraction)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--normal-max-seconds", type=float, default=None,
                    help="use only the first N seconds of each normal video")
    ap.add_argument("--no-extract", action="store_true", help="fail instead of extracting missing features")
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    ap.add_argument("--oof-dir", type=Path, default=OOF_DIR)
    ap.add_argument("--report", type=Path, default=REPORT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.labels.expanduser(), args.data.expanduser(), args.web_labels.expanduser(), args.web_data.expanduser(),
        args.device, args.threads, args.normal_max_seconds, not args.no_extract, args.out, args.oof_dir,
        args.report, args.seed)


if __name__ == "__main__":
    main()
