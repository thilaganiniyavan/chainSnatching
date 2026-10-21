"""Build the night-robust training sets (runs on Google Colab, or any machine with disk + bandwidth).

Detector set  (classes: 0 person, 1 bicycle, 2 motorcycle)
  * ExDark  - real low-light colour photos (HF: vsbgdhdt/ExDark, YOLO format)       -> "lowlight"
  * LLVIP   - real infrared surveillance frames of pedestrians (HF: jsonhash/LLVIP)  -> "ir"
  * VisDrone - elevated / drone views of Asian streets: pedestrians, riders ("people"),
              bicycles and motorcycles ("motor"), many small objects                  -> CCTV-like angles
  * COCO    - daytime street images with people / bikes / motorcycles
  Every non-IR image is then assigned day (30%), synthetic IR (35%) or synthetic low-light
  (35%) with snatchguard.sim.night, and 40% get CCTV degradation (down/up-scaling, heavy
  JPEG, blur) because real CCTV clips are small, re-encoded and soft.
  Target-domain pseudo-labels (teacher-student distillation from your own camera footage)
  come from training/pseudo_label.py and are added into the same folders.
Pose set (COCO-17 keypoints)
  * COCO person keypoints, same day / synthetic-IR / synthetic-low-light mix

Validation is split into val_day / val_night so night performance is measured on its own.

    python training/prepare_dataset.py --out /content/datasets --coco-images 12000 --pose-images 15000
    python training/prepare_dataset.py --out data/datasets --coco-images 300 --pose-images 300 --no-llvip \
        --visdrone-images 0   # quick local check
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snatchguard.sim.night import to_ir, to_lowlight  # noqa: E402

NAMES = ["person", "bicycle", "motorcycle"]
COCO_TO_SG = {0: 0, 1: 1, 3: 2}                       # COCO80 ids -> ours
EXDARK_TO_SG = {10: 0, 0: 1, 9: 2}                    # People, Bicycle, Motorbike
COCO_LABELS = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"
COCO_POSE_LABELS = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-pose.zip"
COCO_IMG = "http://images.cocodataset.org/{split}/{name}"
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def log(msg: str) -> None:
    print(msg, flush=True)


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def _item_rng(key: str) -> random.Random:
    """Reproducible per-image randomness (thread-safe, independent of PYTHONHASHSEED)."""
    return random.Random(zlib.crc32(key.encode()))


def cctv_degrade(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """What re-encoded CCTV footage looks like: low effective resolution, blocky JPEG, soft focus."""
    h, w = img.shape[:2]
    s = rng.uniform(0.3, 0.7)
    small = cv2.resize(img, (max(32, int(w * s)), max(32, int(h * s))), interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(12, 40)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


def night_variant(img: np.ndarray, rng: random.Random, degrade_p: float = 0.4) -> tuple[np.ndarray, str]:
    r = rng.random()
    nprng = np.random.default_rng(rng.randrange(1 << 30))
    strength = rng.uniform(0.6, 1.2)
    if r < 0.30:
        out, kind = img, "day"
    elif r < 0.65:
        out, kind = to_ir(img, nprng, strength), "ir"
    else:
        out, kind = to_lowlight(img, nprng, strength), "lowlight"
    if rng.random() < degrade_p:
        out = cctv_degrade(out, rng)
    return out, kind


class Writer:
    """Writes images + labels into <root>/images/<split> and <root>/labels/<split>."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.counts: dict[str, int] = {}

    def put(self, split: str, name: str, img: np.ndarray, lines: list[str]) -> None:
        (self.root / "images" / split).mkdir(parents=True, exist_ok=True)
        (self.root / "labels" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.root / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (self.root / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        self.counts[split] = self.counts.get(split, 0) + 1


def val_split_for(kind: str) -> str:
    return "val_day" if kind == "day" else "val_night"


# ---------------------------------------------------------------------------
# Detector sources
# ---------------------------------------------------------------------------

def add_exdark(w: Writer, cache: Path, rng: random.Random, val_frac: float) -> None:
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download("vsbgdhdt/ExDark", repo_type="dataset", local_dir=cache / "exdark"))
    n = 0
    for lbl in root.rglob("labels/*.txt"):
        img_path = next((p for p in (lbl.parent.parent / "images").glob(lbl.stem + ".*")), None)
        if img_path is None:
            continue
        lines = []
        for row in lbl.read_text().split("\n"):
            parts = row.split()
            if len(parts) == 5 and int(parts[0]) in EXDARK_TO_SG:
                lines.append(" ".join([str(EXDARK_TO_SG[int(parts[0])])] + parts[1:]))
        if not lines:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        split = "val_night" if rng.random() < val_frac else "train"
        w.put(split, f"exdark_{img_path.stem}", img, lines)
        n += 1
    log(f"ExDark: {n} images with people/bikes")


def add_llvip(w: Writer, cache: Path, rng: random.Random, val_frac: float, max_images: int) -> None:
    from huggingface_hub import hf_hub_download
    z = Path(hf_hub_download("jsonhash/LLVIP", "LLVIP.zip", repo_type="dataset", local_dir=cache / "llvip"))
    out = cache / "llvip" / "extracted"
    if not out.exists():
        log("extracting LLVIP (4 GB)...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
    ann_dir = next(out.rglob("Annotations"), None)
    ir_dir = next((p for p in out.rglob("infrared") if p.is_dir()), None)
    if ann_dir is None or ir_dir is None:
        log(f"LLVIP layout not recognised under {out}; skipping")
        return
    images = {p.stem: p for p in ir_dir.rglob("*.jpg")}
    xmls = sorted(ann_dir.glob("*.xml"))
    rng.shuffle(xmls)
    n = 0
    for x in xmls[:max_images]:
        img_path = images.get(x.stem)
        if img_path is None:
            continue
        tree = ET.parse(x).getroot()
        W = float(tree.findtext("size/width") or 1280)
        H = float(tree.findtext("size/height") or 1024)
        lines = []
        for obj in tree.iter("object"):
            b = obj.find("bndbox")
            x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        img = cv2.imread(str(img_path))
        if img is None or not lines:
            continue
        img = cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1])))
        split = "val_night" if rng.random() < val_frac else "train"
        w.put(split, f"llvip_{x.stem}", img, lines)
        n += 1
    log(f"LLVIP (infrared): {n} images")


VISDRONE = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-{split}.zip"
VISDRONE_TO_SG = {1: 0, 2: 0, 3: 1, 10: 2}          # pedestrian, people(riders/sitting), bicycle, motor


def add_visdrone(w: Writer, cache: Path, rng: random.Random, max_images: int, val_frac: float,
                 splits: tuple[str, ...] = ("train", "val")) -> None:
    n = 0
    for split in splits:
        z = download(VISDRONE.format(split=split), cache / f"visdrone_{split}.zip")
        root = cache / f"visdrone_{split}"
        if not root.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(root)
        ann_dir = next(root.rglob("annotations"))
        img_dir = ann_dir.parent / "images"
        anns = sorted(ann_dir.glob("*.txt"))
        rng.shuffle(anns)
        for ann in anns:
            if n >= max_images:
                break
            img = cv2.imread(str(img_dir / f"{ann.stem}.jpg"))
            if img is None:
                continue
            H, W = img.shape[:2]
            lines = []
            for row in ann.read_text().splitlines():
                f = row.strip().rstrip(",").split(",")
                if len(f) < 6 or f[4] == "0" or int(f[5]) not in VISDRONE_TO_SG:
                    continue
                x, y, bw, bh = (float(v) for v in f[:4])
                if bw < 4 or bh < 4:
                    continue
                lines.append(f"{VISDRONE_TO_SG[int(f[5])]} {(x + bw / 2) / W:.6f} {(y + bh / 2) / H:.6f} "
                             f"{bw / W:.6f} {bh / H:.6f}")
            if not lines:
                continue
            scale = min(1.0, 1280 / max(H, W))
            if scale < 1.0:
                img = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
            r = _item_rng("visdrone" + ann.stem)
            img, kind = night_variant(img, r)
            dst = val_split_for(kind) if r.random() < val_frac else "train"
            w.put(dst, f"visdrone_{kind}_{ann.stem}", img, lines)
            n += 1
    log(f"VisDrone: {n} images")


def _coco_items(labels_root: Path, split: str, keep: dict[int, int], min_needed: set[int]) -> list[tuple[str, list[str]]]:
    items = []
    for lbl in (labels_root / split).glob("*.txt"):
        rows = [r.split() for r in lbl.read_text().split("\n") if r.strip()]
        mapped = [[str(keep[int(r[0])])] + r[1:] for r in rows if int(r[0]) in keep]
        if mapped and any(int(m[0]) in min_needed for m in mapped):
            items.append((lbl.stem, [" ".join(m) for m in mapped]))
    return items


def _fetch_coco(split: str, stem: str) -> np.ndarray | None:
    try:
        with urllib.request.urlopen(COCO_IMG.format(split=split, name=f"{stem}.jpg"), timeout=30) as r:
            data = np.frombuffer(r.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def add_coco_det(w: Writer, cache: Path, rng: random.Random, n_images: int, val_frac: float) -> None:
    z = download(COCO_LABELS, cache / "coco2017labels.zip")
    root = cache / "coco_labels"
    if not root.exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(root)
    labels = next(root.rglob("labels"))
    items = _coco_items(labels, "train2017", COCO_TO_SG, {0, 1, 2})
    moto = [it for it in items if any(l.startswith(("1 ", "2 ")) for l in it[1])]
    moto_stems = {it[0] for it in moto}
    rest = [it for it in items if it[0] not in moto_stems]
    rng.shuffle(moto)
    rng.shuffle(rest)
    # keep every bicycle/motorcycle image we can afford: they are the rare, critical classes
    chosen = moto[: int(n_images * 0.6)]
    chosen += rest[: n_images - len(chosen)]
    log(f"COCO: {len(items)} candidate images ({len(moto)} with bikes); taking {len(chosen)}")

    def work(item):
        stem, lines = item
        img = _fetch_coco("train2017", stem)
        if img is None:
            return
        r = _item_rng(stem)
        img, kind = night_variant(img, r)
        split = val_split_for(kind) if r.random() < val_frac else "train"
        w.put(split, f"coco_{kind}_{stem}", img, lines)

    with ThreadPoolExecutor(32) as ex:
        list(ex.map(work, chosen))


def write_det_yaml(root: Path) -> None:
    """val_cctv (real CCTV frames, teacher-labelled) is added by training/pseudo_label.py when present."""
    for name, val in (("data.yaml", ["val_day", "val_night", "val_cctv"]), ("data_night.yaml", ["val_night"]),
                      ("data_day.yaml", ["val_day"]), ("data_cctv.yaml", ["val_cctv"])):
        cfg = {"path": str(root.resolve()), "train": "images/train",
               "val": [f"images/{v}" for v in val if (root / "images" / v).exists()],
               "names": {i: n for i, n in enumerate(NAMES)}}
        (root / name).write_text(json.dumps(cfg, indent=2))   # JSON is valid YAML


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------

def add_coco_pose(w: Writer, cache: Path, rng: random.Random, n_images: int, val_frac: float) -> None:
    z = download(COCO_POSE_LABELS, cache / "coco2017labels-pose.zip")
    root = cache / "coco_pose_labels"
    if not root.exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(root)
    labels = next(root.rglob("labels"))
    files = list((labels / "train2017").glob("*.txt"))
    rng.shuffle(files)
    files = files[:n_images]
    log(f"COCO pose: taking {len(files)} images")

    def work(lbl: Path):
        img = _fetch_coco("train2017", lbl.stem)
        if img is None:
            return
        r = _item_rng(lbl.stem)
        img, kind = night_variant(img, r)
        split = val_split_for(kind) if r.random() < val_frac else "train"
        w.put(split, f"cocopose_{kind}_{lbl.stem}", img, [r for r in lbl.read_text().split("\n") if r.strip()])

    with ThreadPoolExecutor(32) as ex:
        list(ex.map(work, files))


def write_pose_yaml(root: Path) -> None:
    for name, val in (("data.yaml", ["val_day", "val_night"]), ("data_night.yaml", ["val_night"]),
                      ("data_day.yaml", ["val_day"])):
        cfg = {"path": str(root.resolve()), "train": "images/train",
               "val": [f"images/{v}" for v in val if (root / "images" / v).exists()],
               "kpt_shape": [17, 3], "flip_idx": FLIP_IDX, "names": {0: "person"}}
        (root / name).write_text(json.dumps(cfg, indent=2))


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/datasets")
    ap.add_argument("--cache", default=None, help="download cache (default: <out>/_cache)")
    ap.add_argument("--coco-images", type=int, default=12000)
    ap.add_argument("--pose-images", type=int, default=15000)
    ap.add_argument("--llvip-images", type=int, default=6000)
    ap.add_argument("--visdrone-images", type=int, default=6000)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--no-exdark", action="store_true")
    ap.add_argument("--no-llvip", action="store_true")
    ap.add_argument("--no-pose", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    cache = Path(args.cache) if args.cache else out / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    det = out / "sg_det"
    if det.exists():
        shutil.rmtree(det)
    wd = Writer(det)
    # One source failing (moved file, network hiccup) must not waste a whole Colab session.
    sources = []
    if not args.no_exdark:
        sources.append(("ExDark", lambda: add_exdark(wd, cache, rng, args.val_frac)))
    if not args.no_llvip:
        sources.append(("LLVIP", lambda: add_llvip(wd, cache, rng, args.val_frac, args.llvip_images)))
    if args.visdrone_images:
        sources.append(("VisDrone", lambda: add_visdrone(wd, cache, rng, args.visdrone_images, args.val_frac)))
    if args.coco_images:
        sources.append(("COCO", lambda: add_coco_det(wd, cache, rng, args.coco_images, args.val_frac)))
    failed = []
    for name, fn in sources:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report and carry on with the other sources
            failed.append(name)
            log(f"WARNING: {name} failed ({type(exc).__name__}: {exc}); continuing without it")
    if failed:
        log(f"WARNING: detector set built WITHOUT {failed}")
    write_det_yaml(det)
    log(f"detector set: {wd.counts} -> {det}/data.yaml")

    if not args.no_pose and args.pose_images:
        pose = out / "sg_pose"
        if pose.exists():
            shutil.rmtree(pose)
        wp = Writer(pose)
        add_coco_pose(wp, cache, rng, args.pose_images, args.val_frac)
        write_pose_yaml(pose)
        log(f"pose set: {wp.counts} -> {pose}/data.yaml")


if __name__ == "__main__":
    main()
