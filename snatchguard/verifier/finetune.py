"""Fine-tune the verifier's video network end to end on a GPU, from a self-contained data package.

The frozen verifier (train.py) learns only a small head on Kinetics features. With a GPU the whole R(2+1)D-18 can
adapt to CCTV grabs. Everything that needs this project's perception (detection, tracking, candidates, labels)
runs once on the development machine and is written out as a package of clip strips, so the GPU machine needs
only PyTorch and the package:

    # development machine: candidates + labels + strips for Snatch 1.0, the Snatch-Web test set and the pool
    python -m snatchguard.verifier.finetune pack --out ~/verifier_data
    # GPU machine: every evaluation protocol, then the final model
    python -m snatchguard.verifier.finetune run --data D:/verifier_data
    # any machine: the head-only (frozen) verifier on the same package, the baseline to beat
    python -m snatchguard.verifier.finetune frozen --data ~/verifier_data

A strip holds STRIP_NF frames (the live clip's NF plus PAD on each side, same spacing) of a square CONTEXT x the
live crop, at STRIP_SIDE px, tiled 4 x 4 in one JPEG. Training draws a random NF-frame window (+-0.7 s), crop
(0.85-1.25 x the live scale), flip, colour change and drawn-on graphics from it; evaluation takes the centre,
which is exactly the live clip.

Labels: actor-grounded (train.label_clips) where actors are labelled (Snatch 1.0 and the Snatch-Web test set).
Training-pool events have event times but no actor IDs: their candidates inside the event window form a bag of
which at least one must score high (multiple-instance learning); candidates near the window are ignored. Bags a
person has checked (webdata/trainpool.py `bags`, train/bag_review.json) instead name the clips that show the grab,
which then count like actor-grounded positives.

Sources: Snatch 1.0 and web footage (the Snatch-Web test set and the pool) look different. Each source's negatives
together are drawn as often as its share of the positives (_neg_weights), so where a clip comes from says nothing
about its label, and the threshold must hold on held-out normal footage of each source (_val_split, _threshold).

Protocols (every threshold is fixed on validation footage held out of training, before the test set is scored):
  external   train on Snatch 1.0 + pool, test on the frozen Snatch-Web test set (other cities and cameras); also
             run with the network frozen (a linear probe), so the gain from fine-tuning is measured, not assumed
  s1-cv      5 folds holding out Snatch 1.0 cameras and event videos; train on the rest + Snatch-Web + pool
  curve      `external` with 0, 1/3 and 2/3 of the pool's event uploads: recall against training events
  final      everything except the validation split; this is the model to deploy

Each run fine-tunes one pretrained network (--arch: R(2+1)D-18 or Swin3D-T) and scores every test clip twice: its
live view alone, and the mean over VIEWS (shifted by +-0.73 s, mirrored), each with its own threshold fixed on the
validation footage. `select` then picks the network and scoring to deploy from the Snatch 1.0 cross-validation only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np

from snatchguard.verifier.model import AFTER_S, BEFORE_S, FT_FORMAT, NF, SIDE, cut

log = logging.getLogger(__name__)

PAD = 2                                   # extra frames on each side of the live window
STRIP_NF = NF + 2 * PAD                   # 16 = a 4 x 4 mosaic
CONTEXT = 1.25                            # strip square = CONTEXT x the live crop
STRIP_SIDE = round(SIDE * CONTEXT)        # 140 px
DT = (BEFORE_S + AFTER_S) / (NF - 1)      # frame spacing of the live clip
POOL_NEAR_S = 4.0                         # pool candidates this close to an event are ignored, not negatives
ARCHS = ("r2plus1d_18", "swin3d_t")       # Kinetics-400 top-1: 67.5 % (native 112 px) and 77.7 % (trained at 224 px)
# test-time views (strip frame offset, mirrored): the live window, PAD frames (0.73 s) earlier and later, and their
# mirror images, i.e. variations training draws from; a clip's score is the mean of its views' logits
VIEWS = tuple((o, f) for o in (PAD, 0, 2 * PAD) for f in (False, True))
PACKAGE_VERSION = 1


def strip_times(t0: float) -> np.ndarray:
    return t0 - BEFORE_S + (np.arange(STRIP_NF) - PAD) * DT


# ---------------------------------------------------------------------------------------------------------------
# pack (development machine)

def pool_labels(row: dict, clips, checked: dict | None = None) -> tuple[list[int], list[bool]]:
    """(y, in_bag) per candidate of a training-pool event video: the bag is the event window.
    `checked` (from the bag review) names the window clips that show the grab: they become positives like
    actor-grounded ones, the rest of the window is ignored; {"grab": []} means no clip shows it."""
    from snatchguard.verifier.train import WINDOW_PAD_S
    if row["category"] == "normal":
        return [0] * len(clips), [False] * len(clips)
    es = float(row["event_start_s"])
    ee = float(row["event_end_s"] or es)
    grab = checked["grab"] if checked is not None else None
    ys, bag = [], []
    for c in clips:
        inside = es - WINDOW_PAD_S <= c.t <= ee + WINDOW_PAD_S
        near = inside or es - POOL_NEAR_S <= c.t <= ee + POOL_NEAR_S
        if grab is not None:
            ys.append(1 if inside and any(abs(c.t - g) < 0.01 for g in grab) else -1 if near else 0)   # index: 3 decimals
            bag.append(False)
        else:
            ys.append(-1 if near else 0)
            bag.append(inside)
    return ys, bag


def _strips(path: str, requests: list[tuple[float, tuple]], out_dir: Path, names: list[str]) -> None:
    """One decode pass; each strip is written (and freed) as soon as its last frame has been read."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    need: dict[int, list] = {}
    last = []
    for i, (t0, crop) in enumerate(requests):
        x0, y0, side = crop
        cx, cy = x0 + side / 2, y0 + side / 2
        big = (cx - side * CONTEXT / 2, cy - side * CONTEXT / 2, side * CONTEXT)
        fis = [int(round(max(t, 0.0) * fps)) for t in strip_times(t0)]
        for slot, fi in enumerate(fis):
            need.setdefault(fi, []).append((i, slot, big))
        last.append(fis[-1])
    open_: dict[int, np.ndarray] = {}
    have: dict[int, np.ndarray] = {}
    order = sorted(range(len(requests)), key=lambda i: last[i])
    done, fi, scale, end = 0, -1, None, max(need) if need else -1

    def flush(upto: int) -> None:
        nonlocal done
        while done < len(order) and last[order[done]] <= upto:
            i = order[done]
            frames, ok = open_.pop(i, None), have.pop(i, None)
            if frames is None:
                frames, ok = np.zeros((STRIP_NF, STRIP_SIDE, STRIP_SIDE, 3), np.uint8), np.zeros(STRIP_NF, bool)
            for s in range(STRIP_NF):            # before the first decoded frame: next one; past the end: last one
                if not ok[s] and ok.any():
                    src = s - 1 if s and ok[s - 1] else int(np.argmax(ok))
                    frames[s], ok[s] = frames[src], True
            mosaic = frames.reshape(4, 4, STRIP_SIDE, STRIP_SIDE, 3).transpose(0, 2, 1, 3, 4)
            mosaic = np.ascontiguousarray(mosaic.reshape(4 * STRIP_SIDE, 4 * STRIP_SIDE, 3)[:, :, ::-1])
            cv2.imwrite(str(out_dir / names[i]), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 88])
            done += 1

    while fi < end:
        if not cap.grab():
            break
        fi += 1
        if fi in need:
            ok, img = cap.retrieve()
            if ok:
                if scale is None:
                    scale = min(1.0, 1280 / img.shape[1])
                for i, slot, big in need[fi]:
                    if i not in open_:
                        open_[i] = np.zeros((STRIP_NF, STRIP_SIDE, STRIP_SIDE, 3), np.uint8)
                        have[i] = np.zeros(STRIP_NF, bool)
                    open_[i][slot] = cut(img, big, 1.0 / scale, side=STRIP_SIDE)
                    have[i][slot] = True
        flush(fi)
    flush(1 << 60)


def cmd_pack(a) -> None:
    from snatchguard.core.config import REPO_ROOT, load_yaml
    from snatchguard.evaluation import cache as cache_mod
    from snatchguard.verifier.candidates import CandidateStage
    from snatchguard.verifier.train import ACTORS, NORMAL_MAX_S, WEB, datasets, group_of, label_clips
    from snatchguard.webdata import trainpool

    out = Path(a.out).expanduser()
    (out / "strips").mkdir(parents=True, exist_ok=True)
    actors = json.loads(Path(a.actors).read_text())
    main_tag = cache_mod.perception_tag(load_yaml(str(REPO_ROOT / "configs" / "edge.yaml")), 8.0, None)
    pool_tag = cache_mod.perception_tag(trainpool.pool_config(), 8.0, None)
    jobs = [(s, row, str(p), main_tag) for s, row, p in datasets(Path(a.web))]
    pool_csv = trainpool.TRAIN / "labels.csv"
    if pool_csv.exists() and not a.no_pool:
        with open(pool_csv, newline="", encoding="utf-8") as f:
            jobs += [("POOL", row, str(trainpool.TRAIN / "clips" / row["video"]), pool_tag) for row in csv.DictReader(f)]
    index_path = out / "index.json"
    bags = json.loads(trainpool.BAGS.read_text()) if trainpool.BAGS.exists() else {}
    videos = []
    t_start = time.time()
    for n, (setname, row, path, tag) in enumerate(jobs, 1):
        key = f"{setname}:{row['video']}"
        cp = cache_mod.cache_path(path, tag)
        if not cp.exists():
            log.warning("no perception cache for %s: skipped", key)
            continue
        cv = cache_mod.load(cp)
        frames = cv.frames if row["category"] != "normal" else [f for f in cv.frames if f.t <= NORMAL_MAX_S]
        stage, clips = CandidateStage(), []
        for f in frames:
            clips += stage.step(f.t, f.detections)
        if setname == "POOL":
            ys, bag = pool_labels(row, clips, bags.get(row["video"]))
            status = ("pool-checked" if row["video"] in bags else "pool") if row["category"] != "normal" else ""
        else:
            ys, bag = label_clips(row, clips, actors.get(key)), [False] * len(clips)
            status = actors.get(key, {}).get("status", "") if row["category"] != "normal" else ""
        vid = hashlib.md5(key.encode()).hexdigest()[:10]
        names = [f"{vid}_{i:04d}.jpg" for i in range(len(clips))]
        todo = [i for i, nm in enumerate(names) if not (out / "strips" / nm).exists()]   # labels change, strips stay
        if todo:
            _strips(path, [(clips[i].t, clips[i].crop) for i in todo], out / "strips", [names[i] for i in todo])
        v = {"key": key, "set": setname, "category": row["category"], "path": path,
             "duration": frames[-1].t if frames else 0.0, "status": status,
             "event": [float(row["event_start_s"]), float(row["event_end_s"] or row["event_start_s"])]
             if row.get("event_start_s") else None,
             "clips": [{"strip": nm, "t": round(c.t, 3), "y": y, "bag": b} for nm, c, y, b in zip(names, clips, ys, bag)]}
        v["group"] = "POOL:" + Path(row["video"]).stem.rsplit("_", 1)[0] if setname == "POOL" else group_of(v)
        videos.append(v)
        if todo:
            log.info("[%d/%d] %-50s %4d clips (%5.0f s)", n, len(jobs), key[:50], len(clips), time.time() - t_start)
    index = {"version": PACKAGE_VERSION, "created": time.strftime("%Y-%m-%d %H:%M"),
             "strip": {"nf": STRIP_NF, "side": STRIP_SIDE, "pad": PAD, "context": CONTEXT, "dt": DT,
                       "live_nf": NF, "live_side": SIDE},
             "videos": videos}
    index_path.write_text(json.dumps(index))
    summary: dict[str, dict] = {}
    for v in videos:
        s = summary.setdefault(v["set"], {"event_videos": 0, "normal_hours": 0.0, "clips": 0, "pos": 0, "bag": 0})
        s["clips"] += len(v["clips"])
        s["pos"] += sum(c["y"] == 1 for c in v["clips"])
        s["bag"] += sum(c["bag"] for c in v["clips"])
        if v["category"] == "normal":
            s["normal_hours"] += v["duration"] / 3600
        else:
            s["event_videos"] += 1
    log.info("package %s: %s", out, json.dumps(summary))


# ---------------------------------------------------------------------------------------------------------------
# run (GPU machine)

class Package:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        idx = json.loads((self.root / "index.json").read_text())
        if idx.get("version") != PACKAGE_VERSION:
            raise SystemExit(f"{self.root}: package version {idx.get('version')}, this code reads {PACKAGE_VERSION}")
        self.videos = idx["videos"]
        rows = [(vi, c) for vi, v in enumerate(self.videos) for c in v["clips"]]
        self.vi = np.array([vi for vi, _ in rows])
        self.strip = [str(self.root / "strips" / c["strip"]) for _, c in rows]
        self.t = np.array([c["t"] for _, c in rows], np.float32)
        self.y = np.array([c["y"] for _, c in rows])
        self.bag = np.array([c["bag"] for _, c in rows])
        self.set = np.array([self.videos[vi]["set"] for vi in self.vi])
        self.source = np.where(self.set == "S1", "S1", "web")
        self.normal = np.array([self.videos[vi]["category"] == "normal" for vi in self.vi])
        self.group = np.array([self.videos[vi]["group"] for vi in self.vi])

    def hours(self, mask_videos: np.ndarray) -> float:
        return sum(v["duration"] for vi, v in enumerate(self.videos) if mask_videos[vi] and v["category"] == "normal") / 3600


def load_strip(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)    # cv2.imread fails on non-ASCII Windows paths
    if img is None:
        raise FileNotFoundError(path)
    s = STRIP_SIDE
    return img[:, :, ::-1].reshape(4, s, 4, s, 3).transpose(0, 2, 1, 3, 4).reshape(STRIP_NF, s, s, 3)


def _graphics(frames: np.ndarray, rng: random.Random) -> np.ndarray:
    """A news-style circle, box or arrow drawn over the clip, so drawn-on graphics cannot predict a grab."""
    col = rng.choice([(255, 0, 0), (255, 255, 0), (255, 255, 255), (0, 255, 0), (255, 64, 0)])
    s, kind = frames.shape[1], rng.choice(["circle", "box", "arrow"])
    cx, cy, r = rng.uniform(0.2, 0.8) * s, rng.uniform(0.2, 0.8) * s, rng.uniform(0.15, 0.4) * s
    dx, dy, th = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.choice([2, 3, 4])
    out = frames.copy()
    start = rng.randrange(0, len(frames) // 2)
    for k in range(start, len(frames)):
        img = np.ascontiguousarray(out[k])
        x, y = int(cx + dx * k), int(cy + dy * k)
        if kind == "circle":
            cv2.ellipse(img, (x, y), (int(r), int(r * 0.8)), 0, 0, 360, col, th)
        elif kind == "box":
            cv2.rectangle(img, (int(x - r), int(y - r)), (int(x + r), int(y + r)), col, th)
        else:
            cv2.arrowedLine(img, (int(x - r), int(y - r)), (x, y), col, th, tipLength=0.3)
        out[k] = img
    return out


def augment(frames: np.ndarray, rng: random.Random, graphics: float = 0.25) -> np.ndarray:
    """(STRIP_NF, S, S, 3) uint8 -> (NF, SIDE, SIDE, 3) uint8 with one random view of the clip; `graphics` is
    the chance of a drawn-on news circle, box or arrow."""
    o = rng.randint(0, 2 * PAD)
    x = frames[o:o + NF]
    c = min(int(round(rng.uniform(0.85 * SIDE, STRIP_SIDE))), STRIP_SIDE)   # 0.85-1.25 x the live crop side
    x0, y0 = rng.randint(0, STRIP_SIDE - c), rng.randint(0, STRIP_SIDE - c)
    x = np.stack([cv2.resize(f[y0:y0 + c, x0:x0 + c], (SIDE, SIDE), interpolation=cv2.INTER_AREA) for f in x])
    if rng.random() < 0.5:
        x = x[:, :, ::-1]
    if rng.random() < graphics:
        x = _graphics(x, rng)
    x = x.astype(np.float32)
    if rng.random() < 0.8:
        x = (x - 128.0) * rng.uniform(0.7, 1.3) + 128.0 + rng.uniform(-30, 30)
        gray = x.mean(axis=3, keepdims=True)
        x = gray + (x - gray) * rng.uniform(0.6, 1.4)
    if rng.random() < 0.15:                                       # night IR footage is grey
        x = np.repeat(x.mean(axis=3, keepdims=True), 3, axis=3)
    if rng.random() < 0.2:
        k = rng.choice([3, 5])
        x = np.stack([cv2.GaussianBlur(f, (k, k), 0) for f in x])
    return np.clip(x, 0, 255).astype(np.uint8)


def view(frames: np.ndarray, offset: int = PAD, flip: bool = False) -> np.ndarray:
    """The live clip seen `offset - PAD` frames later, optionally mirrored (a test-time view, VIEWS)."""
    o = (STRIP_SIDE - SIDE) // 2
    x = frames[offset:offset + NF, o:o + SIDE, o:o + SIDE]
    return np.ascontiguousarray(x[:, :, ::-1] if flip else x)


def centre(frames: np.ndarray) -> np.ndarray:
    return view(frames)


class _Clips:
    """Torch dataset over strips: item = (clip tensor, unit id, unit label)."""

    def __init__(self, pkg: Package, items: list[tuple[int, int, int]], train: bool, seed: int,
                 view: tuple[int, bool] = (PAD, False)) -> None:
        self.pkg, self.items, self.train, self.seed, self.view = pkg, items, train, seed, view

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, k: int):
        import torch
        from snatchguard.verifier.model import _MEAN, _STD
        i, unit, y = self.items[k]
        frames = load_strip(self.pkg.strip[i])
        # news uploads often circle the grab itself: negatives get drawn-on graphics more often than positives,
        # so a circle cannot stand in for the motion
        x = (augment(frames, random.Random(self.seed * 1_000_003 + k), GRAPHICS_NEG if y == 0 else GRAPHICS_POS)
             if self.train else view(frames, *self.view))
        x = ((x.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(3, 0, 1, 2)
        return torch.from_numpy(np.ascontiguousarray(x)), unit, y


def _units(pkg: Package, idx: np.ndarray) -> tuple[list[tuple[bool, list[int]]], list[int]]:
    """Positive units, one per event: (False, its actor-grounded clips) or (True, its pool bag); negative clips."""
    pos: dict[tuple[bool, str], list[int]] = {}
    for i in idx:
        key = pkg.videos[pkg.vi[i]]["key"]
        if pkg.y[i] == 1:
            pos.setdefault((False, key), []).append(int(i))
        elif pkg.bag[i]:
            pos.setdefault((True, key), []).append(int(i))
    neg = [int(i) for i in idx if pkg.y[i] == 0]
    return [(is_bag, clips) for (is_bag, _), clips in pos.items()], neg


BAG_MAX = 6          # clips of one bag per batch (a bag's true grab must be among them)
GRAPHICS_NEG, GRAPHICS_POS = 0.4, 0.1


def _neg_weights(pkg: Package, pos_sources: list[str], neg: list[int], hard: set[int]) -> np.ndarray:
    """Sampling weight of each negative clip: event-video negatives (same scene, other moments) 3x normal-footage
    ones, and each source's negatives together weigh its share of `pos_sources` (the source of every positive unit
    or clip). Otherwise mostly-web positives against mostly-Snatch-1.0 normal footage teach "web footage = snatch":
    the head-only verifier trained that way scored held-out web normal footage 0.94 (median)."""
    base = np.array([3.0 if i in hard else 1.0 for i in neg])
    src = pkg.source[np.asarray(neg, int)]
    w = np.zeros(len(neg))
    for s in set(src.tolist()):
        m = src == s
        w[m] = base[m] * pos_sources.count(s) / len(pos_sources) / base[m].sum()
    return w * len(neg) / w.sum() if w.sum() > 0 else base


def _batches(pos: list[tuple[bool, list[int]]], neg: list[int], weights: np.ndarray, batch: int, steps: int,
             seed: int):
    """Exactly `batch` clips: positive units up to half of it (one actor-grounded clip, or a whole bag of up to
    BAG_MAX clips), the rest negatives drawn by `weights` (_neg_weights). A fixed size keeps cuDNN's tuned kernels."""
    rng = random.Random(seed)
    w = [float(x) for x in weights]
    for _ in range(steps):
        items, unit = [], 0
        while True:
            is_bag, u = rng.choice(pos)
            take = (u if len(u) <= BAG_MAX else rng.sample(u, BAG_MAX)) if is_bag else [rng.choice(u)]
            take = take[:max(1, batch // 2)]
            if items and len(items) + len(take) > batch // 2:
                break
            items += [(i, unit, 1) for i in take]
            unit += 1
        for i in rng.choices(neg, weights=w, k=batch - len(items)):
            items.append((i, unit, 0))
            unit += 1
        yield items


def build_net(pretrained: bool = True, arch: str = "r2plus1d_18"):
    import torch
    if arch == "swin3d_t":
        return _swin3d_t(pretrained)
    if arch != "r2plus1d_18":
        raise ValueError(f"unknown arch {arch!r}: {ARCHS}")
    from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
    net = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None)
    net.fc = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(512, 1))
    return net


def _swin3d_t(pretrained: bool):
    """Swin3D-T under R(2+1)D-18's block names (stem, layer1-4, fc), so _freeze and the optimiser groups apply
    unchanged. Clips here are normalised for R(2+1)D; the first step re-normalises them for Swin's ImageNet
    statistics. Window attention takes the 12 x 112 x 112 live clip as it is."""
    import torch
    from torchvision.models.video import Swin3D_T_Weights, swin3d_t
    from snatchguard.verifier.model import _MEAN, _STD
    w = Swin3D_T_Weights.KINETICS400_V1
    base = swin3d_t(weights=w if pretrained else None)
    tf = w.transforms()
    gain = torch.tensor(_STD) / torch.tensor(tf.std)
    bias = (torch.tensor(_MEAN) - torch.tensor(tf.mean)) / torch.tensor(tf.std)

    class Swin3D(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            f = base.features
            self.stem = torch.nn.Sequential(base.patch_embed, base.pos_drop)
            self.layer1, self.layer2 = f[0], torch.nn.Sequential(f[1], f[2])
            self.layer3, self.layer4 = torch.nn.Sequential(f[3], f[4]), torch.nn.Sequential(f[5], f[6], base.norm)
            self.fc = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(base.head.in_features, 1))
            self.register_buffer("gain", gain.view(1, 3, 1, 1, 1), persistent=False)
            self.register_buffer("bias", bias.view(1, 3, 1, 1, 1), persistent=False)

        def forward(self, x):
            x = self.stem(x * self.gain + self.bias)                       # (B, T, H, W, C) from here on
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            return self.fc(x.mean(dim=(1, 2, 3)))

    return Swin3D()


BLOCKS = ("stem", "layer1", "layer2", "layer3", "layer4", "fc")


def _freeze(net, train_from: str) -> None:
    """Blocks before `train_from` keep their Kinetics weights ("fc": a linear probe on frozen features);
    batch-norm statistics stay fixed everywhere (small batches)."""
    import torch
    for name in BLOCKS[:BLOCKS.index(train_from)]:
        for p in getattr(net, name).parameters():
            p.requires_grad_(False)
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm3d):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)


def train_model(pkg: Package, idx: np.ndarray, a, seed: int, tag: str):
    import torch
    torch.manual_seed(seed)
    dev = a.device
    pos, neg = _units(pkg, idx)
    hard = {i for i in neg if not pkg.normal[i]}
    if not pos or not neg:
        raise RuntimeError(f"{tag}: no positives or no negatives to train on")
    net = build_net(arch=a.arch).to(dev)
    _freeze(net, a.train_from)
    trunk = [p for name in BLOCKS[:-1] for p in getattr(net, name).parameters() if p.requires_grad]
    groups = ([{"params": trunk, "lr": a.lr}] if trunk else []) + [{"params": list(net.fc.parameters()), "lr": a.lr * 10}]
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    warm = max(1, a.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.startswith("cuda"))
    pos_src = [str(pkg.source[u[0]]) for _, u in pos]
    weights = _neg_weights(pkg, pos_src, neg, hard)
    batches = list(_batches(pos, neg, weights, a.batch, a.steps, seed))
    flat = [it for b in batches for it in b]
    bounds = np.cumsum([0] + [len(b) for b in batches])
    ds = _Clips(pkg, flat, train=True, seed=seed)
    sampler = [list(range(bounds[k], bounds[k + 1])) for k in range(len(batches))]
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=a.workers,
                                         pin_memory=dev.startswith("cuda"), persistent_workers=a.workers > 0)
    neg_src = pkg.source[np.asarray(neg, int)]
    log.info("%s: %d positive events (%d bags), %d negatives (%d same-scene); %s; %d steps of %d, training from %s",
             tag, len(pos), sum(1 for is_bag, _ in pos if is_bag), len(neg), len(hard),
             ", ".join(f"{s}: {pos_src.count(s)} events, {int((neg_src == s).sum())} negatives"
                       for s in sorted(set(pos_src) | set(neg_src.tolist()))), a.steps, a.batch, a.train_from)
    net.train()
    _freeze(net, a.train_from)
    t0, run = time.time(), []
    for step, (x, unit, y) in enumerate(loader):
        x = x.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.startswith("cuda")):
            z = net(x).squeeze(1)
        z, unit, y = z.float(), unit.to(dev), y.to(dev).float()
        n_units = int(unit.max()) + 1
        # unit score: smooth maximum over its clips (a bag needs one high clip; a single clip is itself)
        zu = torch.full((n_units,), -1e4, device=dev).scatter_reduce(0, unit, z, "amax")
        zs = zu[unit]
        lse = torch.zeros(n_units, device=dev).index_add(0, unit, torch.exp(4.0 * (z - zs)))
        zu = zu + torch.log(lse / torch.bincount(unit, minlength=n_units).float()) / 4.0
        yu = torch.zeros(n_units, device=dev).scatter_reduce(0, unit, y, "amax")
        bce = torch.nn.functional.binary_cross_entropy_with_logits(zu, yu, reduction="none")
        loss = bce[yu == 1].mean() + bce[yu == 0].mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run.append(loss.item())
        if (step + 1) % 50 == 0 or step + 1 == a.steps:
            el = time.time() - t0
            log.info("  %s step %4d/%d  loss %.3f  (%.0f s, about %.0f s left)", tag, step + 1, a.steps,
                     float(np.mean(run[-50:])), el, el / (step + 1) * (a.steps - step - 1))
    return net


def predict(net, pkg: Package, idx: np.ndarray, a, views=VIEWS[:1]) -> np.ndarray:
    """(len(idx), len(views)) logits; _prob turns them into scores."""
    import torch
    net.eval()
    cols = []
    for v in views:
        ds = _Clips(pkg, [(int(i), 0, 0) for i in idx], train=False, seed=0, view=v)
        loader = torch.utils.data.DataLoader(ds, batch_size=a.batch * 2, num_workers=a.workers,
                                             pin_memory=a.device.startswith("cuda"))
        out = []
        with torch.inference_mode():
            for x, _, _ in loader:
                with torch.autocast("cuda", dtype=torch.float16, enabled=a.device.startswith("cuda")):
                    out.append(net(x.to(a.device)).float().squeeze(1).cpu().numpy())
        cols.append(np.concatenate(out) if out else np.zeros(0, np.float32))
    return np.stack(cols, 1)


def _prob(logits: np.ndarray, tta: bool) -> np.ndarray:
    """Score of each clip: its live view alone, or the mean logit of all its views (tta)."""
    z = logits.mean(1) if tta else logits[:, 0]
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def poisson_ci(k: int, exposure: float) -> tuple[float, float]:
    from scipy.stats import chi2
    lo = 0.0 if k == 0 else chi2.ppf(0.025, 2 * k) / 2
    return lo / exposure, chi2.ppf(0.975, 2 * k + 2) / 2 / exposure


def threshold_for(scores_normal: np.ndarray, hours: float, budget: float) -> float:
    """Lowest threshold at which the validation normal footage gives at most `budget` alerts per hour."""
    k = int(math.floor(budget * hours))
    s = np.sort(scores_normal)[::-1]
    if k >= len(s):
        return float(s[-1]) if len(s) else 0.5
    return float(np.nextafter(s[k], np.float32(1.0)))


def _threshold(pkg: Package, idx: np.ndarray, s: np.ndarray, budget: float) -> tuple[float, dict]:
    """The lowest threshold at which the held-out normal footage of each source (Snatch 1.0, web) gives at most
    `budget` alerts per hour; idx/s: validation normal clips and their scores. One threshold over all of it would
    be set mostly by Snatch 1.0, which has most of the hours and different footage."""
    per, thr = {}, 0.0
    for name in sorted(set(pkg.source[idx].tolist())):
        m = pkg.source[idx] == name
        vm = np.zeros(len(pkg.videos), bool)
        vm[np.unique(pkg.vi[idx[m]])] = True
        per[name] = {"normal_hours": round(pkg.hours(vm), 3), "clips": int(m.sum()),
                     "threshold": threshold_for(s[m], pkg.hours(vm), budget)}
        thr = max(thr, per[name]["threshold"])
    for name, d in per.items():
        d["alerts"] = int((s[pkg.source[idx] == name] >= thr).sum())
    return thr, per


def evaluate(pkg: Package, idx: np.ndarray, s: np.ndarray, thr: float, budgets: list[float]) -> dict:
    """Events caught and false alarms per hour of normal footage, at `thr` and along the whole curve.
    actor: a clip with the labelled victim-attacker contact scores >= thr (events with actor labels);
    overlap: any clip whose onset lies within 1.5 s of the event window does (every timed event)."""
    from snatchguard.evaluation.metrics import wilson
    from sklearn.metrics import roc_auc_score
    vi, lab, t, norm = pkg.vi[idx], pkg.y[idx], pkg.t[idx], pkg.normal[idx]
    vmask = np.zeros(len(pkg.videos), bool)
    vmask[np.unique(vi)] = True
    hours = pkg.hours(vmask)
    vids = [int(v) for v in np.unique(vi)]
    actor_events = {v for v in vids if pkg.videos[v]["category"] != "normal"
                    and pkg.videos[v]["status"] in ("ok", "untracked")}
    timed = {v for v in vids if pkg.videos[v]["category"] != "normal" and pkg.videos[v]["event"]}
    actor_ev = np.where((lab == 1) & np.isin(vi, list(actor_events)), vi, -1)
    lo_t = np.array([pkg.videos[v]["event"][0] - 1.5 if v in timed else np.inf for v in vi])
    hi_t = np.array([pkg.videos[v]["event"][1] + 1.5 if v in timed else -np.inf for v in vi])
    overlap_ev = np.where((t >= lo_t) & (t <= hi_t), vi, -1)

    def point(th: float) -> dict:
        hit = s >= th
        fa = int((hit & norm).sum())
        ka, kt = len(set(actor_ev[hit & (actor_ev >= 0)])), len(set(overlap_ev[hit & (overlap_ev >= 0)]))
        lo, hi = (float(x) for x in poisson_ci(fa, hours)) if hours > 0 else (math.nan, math.nan)
        return {"false_alarms": fa, "false_alarms_per_h": round(fa / hours, 2) if hours > 0 else None,
                "fa_ci": [round(lo, 2), round(hi, 2)] if hours > 0 else None,
                "caught_actor": ka, "caught_overlap": kt,
                "recall_actor_ci": [round(x, 3) for x in wilson(ka, max(len(actor_events), 1))],
                "recall_overlap_ci": [round(x, 3) for x in wilson(kt, max(len(timed), 1))]}

    res = {"normal_hours": round(hours, 3), "events_actor": len(actor_events), "events_timed": len(timed),
           "clips": int(len(idx)), "threshold": thr, "at_threshold": point(thr)}
    m = lab >= 0
    if len(set(lab[m])) == 2:
        res["clip_auc"] = round(float(roc_auc_score(lab[m], s[m])), 3)
    curve, fa, ca, co = [], 0, set(), set()
    for i in np.argsort(-s, kind="stable"):            # one pass, the threshold sweeping down
        fa += int(norm[i])
        if actor_ev[i] >= 0:
            ca.add(int(actor_ev[i]))
        if overlap_ev[i] >= 0:
            co.add(int(overlap_ev[i]))
        if not curve or (len(ca), len(co)) != tuple(curve[-1][1:]):
            curve.append([round(fa / hours, 2) if hours > 0 else 0.0, len(ca), len(co)])
    res["curve"] = curve
    res["best_at_budget"] = {}
    for b in budgets:
        ok = [c for c in curve if c[0] <= b]
        res["best_at_budget"][str(b)] = max(ok, key=lambda c: (c[1], c[2])) if ok else [0.0, 0, 0]
    return res


def _val_split(pkg: Package, idx: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole normal-footage groups (cameras, uploads) worth about `frac` of each source's normal hours, for
    the threshold. Sources are split separately: drawn together, Snatch 1.0's long camera groups took 0.46 of the
    0.51 held-out hours and web footage 0.04. Event videos stay in training: the threshold is set on false alarms."""
    hours: dict[str, float] = {}
    for v in pkg.videos:
        if v["category"] == "normal":
            hours[v["group"]] = hours.get(v["group"], 0.0) + v["duration"] / 3600
    by_set: dict[str, list[str]] = {}
    for g, name in sorted({(pkg.group[i], pkg.source[i]) for i in idx if pkg.normal[i]}):
        by_set.setdefault(name, []).append(g)
    val = set()
    for name, groups in sorted(by_set.items()):
        random.Random(seed).shuffle(groups)
        h, tot = 0.0, sum(hours[g] for g in groups)
        for g in groups:
            if h < frac * tot:
                val.add(g)
                h += hours[g]
    is_val = np.array([pkg.group[i] in val for i in idx])
    return idx[~is_val], idx[is_val]


def _save(net, path: Path, threshold: float, meta: dict, arch: str = "r2plus1d_18",
          threshold_tta: float | None = None, confirm: float | None = None, confirm_tta: float | None = None) -> None:
    """threshold: for the live clip alone; threshold_tta: for the mean over VIEWS (live: shifted clips + mirror);
    confirm(_tta): the stricter CONFIRMED-alert thresholds."""
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    ckpt = {"format": FT_FORMAT, "arch": arch,
            "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "threshold": float(threshold), "meta": meta,
            "clip": {"nf": NF, "side": SIDE, "before_s": BEFORE_S, "after_s": AFTER_S}}
    if confirm is not None:
        ckpt["threshold_confirm"] = float(confirm)
    if threshold_tta is not None:
        ckpt["tta"] = {"threshold": float(threshold_tta), "flip": any(f for _, f in VIEWS),
                       "shifts_s": sorted({round((o - PAD) * DT, 4) for o, _ in VIEWS})}
        if confirm_tta is not None:
            ckpt["tta"]["threshold_confirm"] = float(confirm_tta)
    torch.save(ckpt, tmp)
    tmp.replace(path)


def run_protocol(pkg: Package, name: str, train_idx: np.ndarray, test_idx: np.ndarray | None, a, seed: int,
                 save_to: Path | None = None) -> dict:
    trn, val = _val_split(pkg, train_idx, 0.2, seed)
    net = train_model(pkg, trn, a, seed, name)
    vm = pkg.normal[val]
    vmask = np.zeros(len(pkg.videos), bool)
    vmask[np.unique(pkg.vi[val])] = True
    views = VIEWS if a.tta else VIEWS[:1]
    lv = predict(net, pkg, val[vm], a, views)
    out = {"train_clips": int(len(trn)), "val_normal_hours": round(pkg.hours(vmask), 3)}
    # two alert levels, both fixed here: SUSPECT at `budget` false alarms/h, CONFIRMED at `budget_confirm`
    for sfx, tta in (("", False), ("_tta", True)) if a.tta else (("", False),):
        sv = _prob(lv, tta)
        out["threshold" + sfx], out["val_by_source" + sfx] = _threshold(pkg, val[vm], sv, a.budget)
        out["threshold_confirm" + sfx] = max(_threshold(pkg, val[vm], sv, a.budget_confirm)[0], out["threshold" + sfx])
    if test_idx is not None and len(test_idx):
        lt = predict(net, pkg, test_idx, a, views)
        scores = {}
        for sfx, tta in (("", False), ("_tta", True)) if a.tta else (("", False),):
            s = scores["score" + sfx] = _prob(lt, tta)
            t = out["test" + sfx] = evaluate(pkg, test_idx, s, out["threshold" + sfx], a.budgets)
            t["threshold_confirm"] = out["threshold_confirm" + sfx]
            t["at_confirm"] = evaluate(pkg, test_idx, s, out["threshold_confirm" + sfx], [])["at_threshold"]
            for lvl, p in (("suspect", t["at_threshold"]), ("confirmed", t["at_confirm"])):
                log.info("%s%s %s: %d/%d events (actor), %d/%d (time overlap), %s false alarms/h %s", name,
                         " (views)" if tta else "", lvl, p["caught_actor"], t["events_actor"], p["caught_overlap"],
                         t["events_timed"], p["false_alarms_per_h"], p["fa_ci"])
        np.savez_compressed(Path(a.out) / f"scores_{name}.npz", idx=test_idx, **scores)
    if save_to is not None:
        _save(net, save_to, out["threshold"], {"protocol": name, "trained": time.strftime("%Y-%m-%d %H:%M"),
                                               "steps": a.steps, "budget_per_h": a.budget,
                                               "budget_confirm_per_h": a.budget_confirm,
                                               "val_normal_hours": out["val_normal_hours"]},
              a.arch, out.get("threshold_tta"), out["threshold_confirm"], out.get("threshold_confirm_tta"))
    del net
    return out


def learning_curve(pkg: Package, allidx: np.ndarray, a) -> list[dict]:
    """The external protocol with a fraction of the training-pool event uploads (fixed order): how recall on the
    frozen test set grows with training events, so the events needed for a target can be estimated, not guessed.
    The full pool is the `external` run itself."""
    sets = pkg.set[allidx]
    uploads = sorted({pkg.group[i] for i in allidx if pkg.set[i] == "POOL" and not pkg.normal[i]})
    random.Random(a.seed).shuffle(uploads)
    out = []
    for frac in (0.0, 1 / 3, 2 / 3):
        keep = set(uploads[:round(frac * len(uploads))])
        drop = np.array([pkg.set[i] == "POOL" and not pkg.normal[i] and pkg.group[i] not in keep for i in allidx])
        train = allidx[(sets != "WEB") & ~drop]
        name = f"curve{round(100 * frac)}"
        res = _retry(lambda: run_protocol(pkg, name, train, allidx[sets == "WEB"], a, a.seed), a)
        res["pool_uploads"], res["pool_events"] = len(keep), len({pkg.vi[i] for i in train if pkg.set[i] == "POOL"
                                                                  and not pkg.normal[i]})
        out.append(res)
    return out


def _retry(fn, a):
    """Run fn; on "CUDA out of memory" halve the batch and start that protocol again."""
    import torch
    while True:
        try:
            return fn()
        except torch.cuda.OutOfMemoryError:
            if a.batch <= 4:
                raise
            a.batch //= 2
            torch.cuda.empty_cache()
            log.warning("GPU out of memory: starting again with batch %d", a.batch)


def _pool(folds: list[dict], key: str) -> dict:
    """Events caught and false alarms summed over the cross-validation folds, each at its own threshold."""
    tot = {}
    for f in folds:
        p, t = f[key]["at_threshold"], f[key]
        for k in ("false_alarms", "caught_actor", "caught_overlap"):
            tot[k] = tot.get(k, 0) + p[k]
        for k in ("normal_hours", "events_actor", "events_timed"):
            tot[k] = tot.get(k, 0) + t[k]
    tot["false_alarms_per_h"] = round(tot["false_alarms"] / max(tot["normal_hours"], 1e-9), 2)
    tot["fa_ci"] = [round(x, 2) for x in poisson_ci(tot["false_alarms"], max(tot["normal_hours"], 1e-9))]
    aucs = [f[key]["clip_auc"] for f in folds if "clip_auc" in f[key]]
    tot["mean_clip_auc"] = round(float(np.mean(aucs)), 3) if aucs else None
    return tot


def cmd_run(a) -> None:
    import torch
    from sklearn.model_selection import GroupKFold
    if a.device == "auto":
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.device == "cpu" and not a.quick:
        raise SystemExit("No CUDA GPU visible to PyTorch. Fine-tuning on a CPU would take days; see docs/GPU_VERIFIER.md")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(Path(a.out) / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.getLogger().addHandler(fh)
    if a.device.startswith("cuda"):
        p = torch.cuda.get_device_properties(0)
        log.info("GPU: %s, %.1f GB", p.name, p.total_memory / 2**30)
        torch.backends.cudnn.benchmark = True
    pkg = Package(a.data)
    allidx = np.arange(len(pkg.y))
    if a.quick:                                  # smoke test: a handful of clips, a few steps
        keep = set()
        for s in ("S1", "WEB", "POOL"):
            m = np.where(pkg.set == s)[0]
            keep |= set(m[pkg.y[m] == 1][:4]) | set(m[pkg.bag[m]][:6]) | set(m[pkg.y[m] == 0][:40])
        allidx = np.array(sorted(keep))
    a.tta = not a.no_tta
    report = {"package": str(pkg.root), "device": a.device, "arch": a.arch, "steps": a.steps, "batch": a.batch,
              "lr": a.lr, "train_from": a.train_from, "views": [list(v) for v in (VIEWS if a.tta else VIEWS[:1])],
              "budget_per_h": a.budget, "budget_confirm_per_h": a.budget_confirm, "protocols": {}}
    t0 = time.time()
    sets = pkg.set[allidx]
    if "external" in a.protocols:
        report["protocols"]["external"] = _retry(lambda: run_protocol(
            pkg, "external", allidx[sets != "WEB"], allidx[sets == "WEB"], a, a.seed,
            Path(a.out) / "verifier_ft_external.pt"), a)
        if a.train_from != "fc":         # the same data and protocol with the network frozen: what fine-tuning adds
            probe = copy.copy(a)
            probe.train_from = "fc"
            report["protocols"]["external-probe"] = _retry(lambda: run_protocol(
                pkg, "external-probe", allidx[sets != "WEB"], allidx[sets == "WEB"], probe, a.seed), probe)
    if "curve" in a.protocols:
        report["protocols"]["curve"] = learning_curve(pkg, allidx, a)
    if "s1-cv" in a.protocols:
        s1 = allidx[sets == "S1"]
        folds = []
        for f, (_, te) in enumerate(GroupKFold(n_splits=a.folds).split(s1, groups=pkg.group[s1])):
            test = s1[te]
            train = np.setdiff1d(allidx, test)
            folds.append(_retry(lambda: run_protocol(pkg, f"s1-cv{f}", train, test, a, a.seed + f), a))
        report["protocols"]["s1-cv"] = {"folds": folds}
        for key in ("test", "test_tta") if a.tta else ("test",):
            tot = _pool(folds, key)
            report["protocols"]["s1-cv"]["pooled_at_thresholds" + ("_tta" if key == "test_tta" else "")] = tot
            log.info("s1-cv pooled%s: %d/%d events (actor), %d/%d (overlap), %.1f false alarms/h",
                     " (views)" if key == "test_tta" else "", tot["caught_actor"], tot["events_actor"],
                     tot["caught_overlap"], tot["events_timed"], tot["false_alarms_per_h"])
    if "final" in a.protocols:
        report["protocols"]["final"] = _retry(lambda: run_protocol(pkg, "final", allidx, None, a, a.seed + 100,
                                                                   Path(a.out) / "verifier_ft.pt"), a)
    report["minutes"] = round((time.time() - t0) / 60, 1)
    report["batch_used"] = a.batch
    (Path(a.out) / "report.json").write_text(json.dumps(report, indent=1))
    log.info("done in %.0f min: %s", report["minutes"], Path(a.out) / "report.json")


def cmd_frozen(a) -> None:
    """The head-only verifier (train.py's frozen Kinetics trunk + PCA + logistic regression) on this package, under
    the `external` protocol with the same validation split and threshold rule: what more data gives without a GPU,
    and the baseline the fine-tuned network has to beat. Runs on a CPU (OpenVINO); embeddings are cached."""
    from snatchguard.verifier.model import Encoder, Head
    pkg = Package(a.data)
    rng = np.random.default_rng(a.seed)
    allidx = np.arange(len(pkg.y))
    test = allidx[pkg.set == "WEB"]
    trn, val = _val_split(pkg, allidx[pkg.set != "WEB"], 0.2, a.seed)
    val = val[pkg.normal[val]]
    keep = [trn[(pkg.y[trn] >= 0) & ~pkg.normal[trn]]]          # positives and same-scene negatives: all
    for name in sorted(set(pkg.source[trn].tolist())):          # normal footage: up to n_normal clips per source
        m = trn[pkg.normal[trn] & (pkg.source[trn] == name)]
        keep.append(rng.choice(m, min(a.n_normal, len(m)), replace=False))
    keep = np.concatenate(keep)
    pos, is_neg = keep[pkg.y[keep] == 1], pkg.y[keep] == 0
    cache_path = pkg.root / "frozen_embeddings.npz"
    cache = dict(np.load(cache_path)) if cache_path.exists() else {}
    enc = Encoder(threads=a.threads)
    jobs = [(f"{i}", int(i), False) for i in np.unique(np.concatenate([test, val, keep]))] + \
           [(f"{i}f", int(i), True) for i in pos]
    todo = [j for j in jobs if j[0] not in cache]
    log.info("frozen baseline: %d clips to embed (%d cached), backend %s", len(todo), len(jobs) - len(todo), enc.backend)
    t0 = time.time()
    for b in range(0, len(todo), 16):
        chunk = todo[b:b + 16]
        clips = np.stack([centre(load_strip(pkg.strip[i]))[:, :, ::-1] if flip else centre(load_strip(pkg.strip[i]))
                          for _, i, flip in chunk])
        for (name, _, _), e in zip(chunk, enc.encode(clips)):
            cache[name] = e
        if (b // 16) % 20 == 19 or b + 16 >= len(todo):
            np.savez(cache_path, **cache)
            log.info("  %d/%d embedded (%.0f s)", min(b + 16, len(todo)), len(todo), time.time() - t0)
    emb = lambda idx, flip=False: np.stack([cache[f"{i}{'f' if flip else ''}"] for i in idx])
    X = np.concatenate([emb(keep), emb(pos, True)])
    y = np.concatenate([(pkg.y[keep] == 1).astype(int), np.ones(len(pos), int)])
    w = np.ones(len(y))
    neg = keep[is_neg].tolist()
    w[:len(keep)][is_neg] = _neg_weights(pkg, [str(x) for x in pkg.source[pos]], neg,
                                         {i for i in neg if not pkg.normal[i]})
    head = Head.fit(X, y, n_components=a.pca, C=a.C, sample_weight=w)
    vmask = np.zeros(len(pkg.videos), bool)
    vmask[np.unique(pkg.vi[val])] = True
    thr, per_src = _threshold(pkg, val, head.proba(emb(val)), a.budget)
    res = evaluate(pkg, test, head.proba(emb(test)), thr, a.budgets)
    out = {"protocol": "external, frozen trunk + PCA/logistic head (CPU)", "train_positive_clips": int(len(pos)),
           "train_positive_events": len({int(pkg.vi[i]) for i in pos}), "train_clips": int(len(keep)),
           "val_normal_hours": round(pkg.hours(vmask), 3), "threshold": thr, "val_by_source": per_src, "test": res}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "frozen_external.json").write_text(json.dumps(out, indent=1))
    p = res["at_threshold"]
    log.info("frozen external: %d positive events in training; threshold %.4f -> %d/%d events (actor), %d/%d (time "
             "overlap), %s false alarms/h %s", out["train_positive_events"], thr, p["caught_actor"], res["events_actor"],
             p["caught_overlap"], res["events_timed"], p["false_alarms_per_h"], p["fa_ci"])


def cmd_weights(a) -> None:
    """The pretrained networks into PyTorch's cache, before any run: torch.hub's own download has no timeout, so on
    a flaky connection it can wait forever part-way. Here a stalled download resumes where it stopped, and each
    file must match the hash prefix in its name (as torchvision checks)."""
    import hashlib
    import httpx
    import torch
    from torchvision.models.video import R2Plus1D_18_Weights, Swin3D_T_Weights

    def sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    folder = Path(torch.hub.get_dir()) / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    for w in (R2Plus1D_18_Weights.KINETICS400_V1, Swin3D_T_Weights.KINETICS400_V1):
        name = w.url.rsplit("/", 1)[-1]
        prefix = name.rsplit("-", 1)[-1].split(".")[0]
        dst, part = folder / name, folder / (name + ".part")
        if dst.exists() and sha256(dst).startswith(prefix):
            log.info("%s: already downloaded", name)
            continue
        for attempt in range(a.tries):
            have = part.stat().st_size if part.exists() else 0
            try:
                with httpx.stream("GET", w.url, headers={"Range": f"bytes={have}-"} if have else {},
                                  timeout=httpx.Timeout(30.0), follow_redirects=True) as r:
                    if r.status_code == 416:                   # nothing left to fetch
                        break
                    r.raise_for_status()
                    with open(part, "ab" if r.status_code == 206 else "wb") as f:
                        for chunk in r.iter_bytes(1 << 20):
                            f.write(chunk)
                break
            except (httpx.HTTPError, OSError) as exc:
                got = part.stat().st_size if part.exists() else 0
                log.warning("%s: download interrupted at %.0f MB (%s); resuming (%d/%d)", name, got / 1e6, exc,
                            attempt + 1, a.tries)
                time.sleep(min(30, 2 ** attempt))
        if not part.exists() or not sha256(part).startswith(prefix):
            part.unlink(missing_ok=True)
            raise SystemExit(f"{name}: download failed or corrupt. Run again; or copy {name} into {folder}")
        part.replace(dst)
        log.info("%s: downloaded and checked", name)


def cmd_select(a) -> None:
    """The model to deploy, by a rule fixed before any run: most Snatch 1.0 cross-validation events caught (each
    fold at its own threshold), then the higher mean clip AUC there; ties keep the earlier run and the live view
    alone. The Snatch-Web test set plays no part, so it stays a held-out result for every candidate."""
    import torch
    cands = []
    for k, run in enumerate(a.runs):
        rep_path, model = Path(run) / "report.json", Path(run) / "verifier_ft.pt"
        if not rep_path.exists() or not model.exists():
            log.warning("%s: no report.json or verifier_ft.pt, skipped", run)
            continue
        rep = json.loads(rep_path.read_text())
        cv = rep["protocols"].get("s1-cv", {})
        for tta in (False, True):
            tot = cv.get("pooled_at_thresholds_tta" if tta else "pooled_at_thresholds")
            if tot is not None:
                cands.append(((tot["caught_actor"], tot.get("mean_clip_auc") or 0.0, -k, -int(tta)),
                              run, rep.get("arch", "r2plus1d_18"), tta, tot))
    if not cands:
        raise SystemExit("no run with an s1-cv result and a final model")
    for c in sorted(cands, key=lambda c: c[0], reverse=True):
        log.info("  %-28s %-12s views=%-5s s1-cv %d/%d events, %.1f false alarms/h, clip AUC %s", c[1], c[2],
                 c[3], c[4]["caught_actor"], c[4]["events_actor"], c[4]["false_alarms_per_h"], c[4].get("mean_clip_auc"))
    _, run, arch, tta, tot = max(cands, key=lambda c: c[0])
    ckpt = torch.load(Path(run) / "verifier_ft.pt", map_location="cpu", weights_only=True)
    if tta and "tta" not in ckpt:
        raise SystemExit(f"{run}/verifier_ft.pt has no multi-view threshold")
    ckpt["tta_default"] = tta
    choice = {"run": str(run), "arch": arch, "views": tta, "s1_cv": tot, "selected": time.strftime("%Y-%m-%d %H:%M"),
              "rule": "max s1-cv events caught, then mean clip AUC; ties: earlier run, single view"}
    ckpt["selection"] = json.dumps(choice)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    ext = Path(run) / "verifier_ft_external.pt"
    if ext.exists():        # the same network trained without Snatch-Web: the one to demonstrate on Snatch-Web clips
        demo = torch.load(ext, map_location="cpu", weights_only=True)
        demo["tta_default"] = tta
        torch.save(demo, out.with_name("verifier_ft_external.pt"))
        choice["demo_model"] = str(out.with_name("verifier_ft_external.pt"))
    out.with_name("selection.json").write_text(json.dumps(choice, indent=1))
    log.info("deploy: %s (%s, %s) -> %s", run, arch, "multi-view" if tta else "single view", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    from snatchguard.core.config import REPO_ROOT
    p.add_argument("--out", default=str(Path.home() / "verifier_data"))
    p.add_argument("--web", default=str(Path.home() / "snatch_web_dataset"))
    p.add_argument("--actors", default=str(REPO_ROOT / "data" / "labels" / "actors.json"))
    p.add_argument("--no-pool", action="store_true", help="leave the training pool out")
    r = sub.add_parser("run")
    r.add_argument("--data", required=True, help="the folder made by `pack` (index.json + strips/)")
    r.add_argument("--out", default=str(REPO_ROOT / "runtime" / "verifier_ft"))
    r.add_argument("--protocols", nargs="+", default=["external", "curve", "s1-cv", "final"])
    r.add_argument("--steps", type=int, default=1500)
    r.add_argument("--batch", type=int, default=16)
    r.add_argument("--lr", type=float, default=1e-4)
    r.add_argument("--train-from", default="layer3", choices=BLOCKS[1:],
                   help="first block that is trained (earlier ones keep the Kinetics weights); fc = linear probe")
    r.add_argument("--folds", type=int, default=5)
    r.add_argument("--budget", type=float, default=10.0, help="false alarms per hour the threshold allows")
    r.add_argument("--budget-confirm", type=float, default=1.0,
                   help="false alarms per hour of the stricter CONFIRMED threshold")
    r.add_argument("--budgets", type=float, nargs="+", default=[1, 10, 100])
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--device", default="auto")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--quick", action="store_true", help="smoke test on a few clips (works on a CPU)")
    r.add_argument("--arch", default="r2plus1d_18", choices=ARCHS, help="pretrained video network to fine-tune")
    r.add_argument("--no-tta", action="store_true", help="score the live view only (no shifted / mirrored views)")
    fz = sub.add_parser("frozen", help="CPU baseline: the head-only verifier on this package (external protocol)")
    fz.add_argument("--data", required=True)
    fz.add_argument("--out", default=str(REPO_ROOT / "runtime" / "verifier_ft"))
    fz.add_argument("--n-normal", type=int, default=1500, help="normal-footage clips of each set sampled for training")
    fz.add_argument("--pca", type=int, default=32)
    fz.add_argument("--C", type=float, default=0.05)
    fz.add_argument("--budget", type=float, default=10.0)
    fz.add_argument("--budgets", type=float, nargs="+", default=[1, 10, 100])
    fz.add_argument("--threads", type=int, default=2)
    fz.add_argument("--seed", type=int, default=1)
    wt = sub.add_parser("weights", help="download the pretrained networks (resumes stalled downloads)")
    wt.add_argument("--tries", type=int, default=30)
    sl = sub.add_parser("select", help="pick the model to deploy from finished runs (Snatch 1.0 cross-validation)")
    sl.add_argument("--runs", nargs="+", required=True, help="output folders of `run` (report.json + verifier_ft.pt)")
    sl.add_argument("--out", default=str(REPO_ROOT / "models" / "verifier" / "verifier_ft.pt"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    {"pack": cmd_pack, "run": cmd_run, "frozen": cmd_frozen, "select": cmd_select, "weights": cmd_weights}[a.cmd](a)


if __name__ == "__main__":
    main()
