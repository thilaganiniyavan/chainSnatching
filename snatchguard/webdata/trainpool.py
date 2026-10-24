"""Training pool: more real snatch events for the learned verifier, kept apart from the frozen test sets.

The clean-only test set (labels.csv) stays frozen and is never trained on. The pool takes what the test set left:
  * segments of downloaded videos nobody has reviewed yet, if the pre-screen calls them CCTV-like,
  * reviewed snatches the test set rejected for editing (circles, captions) or for starting too close to the grab.
Every source video that gave a test clip is excluded whole, since another shot of the same upload can replay the
same incident; dedup.py has already dropped footage repeated from Snatch 1.0 or from another upload. Re-edited
copies (cropped, zoomed, re-timed) can slip past its fingerprints, so `leaks` also matches image features around
every pool event against every test event, and every pool normal clip against the Snatch-Web test clips (another
upload of a test camera), and excludes the whole upload of any match.

    python -m snatchguard.webdata.trainpool sheets            # review images: 8-24 evenly spaced frames per segment
    python -m snatchguard.webdata.trainpool fine KEY T0 T1    # 16 frames between T0 and T1 (source seconds)
    python -m snatchguard.webdata.trainpool build             # cut clips + labels for every decided event, + normals
    python -m snatchguard.webdata.trainpool leaks             # pool clips vs test clips (ORB + RANSAC), then build
    python -m snatchguard.webdata.trainpool cache             # perception for the verifier's candidates (no pose)
    python -m snatchguard.webdata.trainpool bags PACKAGE      # after finetune pack: each event's window clips, as
                                                              # the network sees them, to mark which show the grab

Decisions: review/train_decisions_snatch.csv  key,decision,lighting,event_start_s,event_end_s,notes
    decision: snatch_bike | snatch_foot | no_event | not_cctv | other | unclear   (times: source-video seconds)
Output: ~/snatch_web_dataset/train/  clips/{snatch,normal}/*.mp4, labels.csv (SnatchGuard format), manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from snatchguard.webdata.discover import DATASET
from snatchguard.webdata.review import REVIEW, _tile, decisions, fine, frames_at, reviewed

TRAIN = DATASET / "train"
DECISIONS = REVIEW / "train_decisions_snatch.csv"
FIELDS = ["key", "decision", "lighting", "event_start_s", "event_end_s", "notes"]
LEAD_S, TAIL_S = 15.0, 8.0         # clip bounds around an event (training needs less lead than the clean test)
NORMAL_S = 90.0                    # first normal clip of a segment; the rest, up to NORMAL_MAX_S, is a second one
NORMAL_MAX_S = 300.0               # (perception costs ~1x real time)
NORMAL_MIN_S = 20.0                # shortest normal segment (the clean test set needs 60 s; training does not)
LEAKS = TRAIN / "leak_check.json"
BAGS = TRAIN / "bag_review.json"        # clip video -> {"grab": [onset times of the window clips showing the grab]}
LEAK_INLIERS = 40                  # geometric matches that mean the same footage (unrelated street pairs: <= 25)
# reviewed for the test set but not taken: still real grabs (times known) or worth a second look (no times)
LEGACY_SNATCH = ("snatch_bike", "snatch_foot")
LEGACY_RECHECK = ("too_short", "not_clean_auto", "ambiguous")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def test_sources() -> set[str]:
    """Uploads never trained on: those of the Snatch-Web test clips, and any found to repeat a test event."""
    with open(DATASET / "manifest.csv", newline="", encoding="utf-8") as f:
        out = {r["source_id"] for r in csv.DictReader(f)}
    for clip, r in _load(LEAKS).items():
        if r["best"] and r["best"][0][0] >= LEAK_INLIERS:
            out.add(Path(clip).stem.rsplit("_", 1)[0])
    return out


def train_decisions() -> dict[str, dict]:
    if not DECISIONS.exists():
        return {}
    with open(DECISIONS, newline="", encoding="utf-8") as f:
        return {r["key"]: r for r in csv.DictReader(f)}


def raw(key: str, kind: str = "snatch") -> Path:
    return DATASET / "raw" / kind / f"{key.split('#')[0]}.mp4"


def pool() -> list[tuple[str, dict]]:
    """Segments still to review for training: unreviewed CCTV-like ones, and legacy ones without event times."""
    dd = _load(DATASET / "meta" / "dedup_snatch.json")
    pre = _load(REVIEW / "prescreen_snatch.json")
    skip, done, old = test_sources(), set(train_decisions()), decisions("snatch")
    seen = reviewed("snatch")
    out = []
    for k, v in dd.items():
        if v.get("status") != "ok" or k in done or k.split("#")[0] in skip or v["end"] - v["start"] < 3.0:
            continue          # 'check' = possible Snatch 1.0 repeat: never train on it
        if k in seen:
            if old.get(k, {}).get("decision") not in LEGACY_RECHECK:
                continue
        else:
            p = pre.get(k)
            if not p or p["frames_with_person"] < 2 or (p["median_person_height"] or 1) > 0.45:
                continue
            if p["max_bikes"] == 0 and p["max_persons"] < 2:
                continue
        out.append((k, v))
    return sorted(out)


def by_promise(segs: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """Most promising first: every video's best segment before any video's second one; best = a raw upload
    titled as CCTV of a snatching, people and a two-wheeler in view, 6-60 s long."""
    import re
    from snatchguard.webdata.download import CCTV_WORDS
    meta = {}
    for line in (DATASET / "meta" / "candidates.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        meta[d["id"]] = d
    pre = _load(REVIEW / "prescreen_snatch.json")

    def score(key: str, seg: dict) -> float:
        m, p = meta.get(key.split("#")[0], {}), pre.get(key, {})
        title = m.get("title", "")
        return (1.0 * bool(CCTV_WORDS.search(title)) + 1.0 * (m.get("source") == "raw")
                + 1.0 * (p.get("max_bikes", 0) > 0) + 0.5 * (p.get("frames_with_person", 0) >= 0.5 * p.get("frames", 1))
                + 1.0 * (6.0 <= seg["end"] - seg["start"] <= 60.0)
                + 0.5 * bool(re.search(r"chain|snatch|giật|กระชาก|ragut|jambret|arrebat|motochorro", title, re.I)))

    per_video: dict[str, list] = {}
    for key, seg in segs:
        per_video.setdefault(key.split("#")[0], []).append((score(key, seg), key, seg))
    ranked = []
    for items in per_video.values():
        for rank, (sc, key, seg) in enumerate(sorted(items, key=lambda x: -x[0])):
            ranked.append((rank, -sc, key, seg))
    return [(key, seg) for _, _, key, seg in sorted(ranked, key=lambda x: (x[0], x[1], x[2]))]


def rows_for(dur: float) -> int:
    """Rows of 8 frames for a segment: about one frame per 1.5-2 s, at most 3 rows."""
    return 1 if dur <= 16 else 2 if dur <= 40 else 3


def sheets(max_rows: int = 6, tag: str = "", limit: int = 0) -> None:
    """Review images: each segment as 1-3 rows of 8 evenly spaced frames (source-video seconds on each), labelled
    [n]; index.json maps n to the segment key (keys are not typed off the image: I/l and O/0 look alike)."""
    segs = by_promise(pool())[:limit] if limit else by_promise(pool())
    out_dir = REVIEW / f"train_sheets{'_' + tag if tag else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    old, index, rows, sheet = decisions("snatch"), [], [], 0

    def write() -> None:
        nonlocal rows, sheet
        if rows:
            cv2.imwrite(str(out_dir / f"sheet_{sheet:03d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82])
            rows, sheet = [], sheet + 1

    for num, (key, seg) in enumerate(segs):
        n = rows_for(seg["end"] - seg["start"])
        if len(rows) + n > max_rows:
            write()
        ts = list(np.linspace(seg["start"], seg["end"] - 0.1, 8 * n))
        imgs = frames_at(raw(key), ts)
        tiles = [_tile(im if im is not None else np.zeros((135, 240, 3), np.uint8),
                       f"[{num}] {t:.1f}" if i == 0 else f"{t:.1f}s", 240, 135) for i, (im, t) in enumerate(zip(imgs, ts))]
        rows += [np.hstack(tiles[r * 8:(r + 1) * 8]) for r in range(n)]
        index.append({"n": num, "sheet": sheet, "key": key, "start": seg["start"], "end": seg["end"],
                      "old_note": old.get(key, {}).get("notes", "")})
        if num % 10 == 9:
            (out_dir / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    write()
    (out_dir / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(segs)} segments -> {sheet} sheets in {out_dir}")


def accepted() -> dict[str, dict]:
    """Every event the training pool uses: legacy reviewed snatches with times, then this pool's decisions."""
    skip, out = test_sources(), {}
    for k, d in decisions("snatch").items():
        if d["decision"] in LEGACY_SNATCH and d.get("event_start_s") and k.split("#")[0] not in skip \
                and (d.get("clean") or "").lower() != "yes":
            out[k] = {f: d.get(f, "") for f in FIELDS}
    for k, d in train_decisions().items():
        out[k] = d
    return {k: d for k, d in out.items()
            if d["decision"] in LEGACY_SNATCH and d.get("event_start_s") and k.split("#")[0] not in skip}


def _cut(src: Path, start: float, end: float, out: Path) -> None:
    import imageio_ffmpeg
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{start:.2f}",
                        "-to", f"{end:.2f}", "-i", str(src), "-an", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "21", "-pix_fmt", "yuv420p", str(out)], check=True)


def build() -> None:
    """Cut each accepted event (LEAD_S before to TAIL_S after) and each CCTV-like normal segment into train/."""
    labels, manifest = [], []
    dd = _load(DATASET / "meta" / "dedup_snatch.json")
    for key, d in sorted(accepted().items()):
        seg, es = dd[key], float(d["event_start_s"])
        ee = float(d["event_end_s"] or es)
        start, end = max(seg["start"], es - LEAD_S), min(seg["end"], ee + TAIL_S)
        name = f"snatch/{key.replace('#', '_')}.mp4"
        _cut(raw(key), start, end, TRAIN / "clips" / name)
        labels.append({"video": name, "category": d["decision"], "lighting": d["lighting"],
                       "event_start_s": f"{es - start:.2f}", "event_end_s": f"{ee - start:.2f}", "fold": "T",
                       "notes": d.get("notes", "")})
        manifest.append({"clip": name, "source_id": key.split("#")[0], "segment": key,
                         "source_start_s": f"{start:.2f}", "source_end_s": f"{end:.2f}"})
    ddn, pre = _load(DATASET / "meta" / "dedup_normal.json"), _load(REVIEW / "prescreen_normal.json")
    skip = test_sources()
    for key, seg in sorted(ddn.items()):
        p = pre.get(key)
        if seg.get("status") != "ok" or key.split("#")[0] in skip or seg["end"] - seg["start"] < NORMAL_MIN_S \
                or not p or p["frames_with_person"] < 3 or (p["median_person_height"] or 1) > 0.45:
            continue
        parts = [("", seg["start"], min(seg["end"], seg["start"] + NORMAL_S))]
        if seg["end"] - seg["start"] >= NORMAL_S + NORMAL_MIN_S:      # a new name: caches are keyed by file name
            parts.append(("-b", seg["start"] + NORMAL_S, min(seg["end"], seg["start"] + NORMAL_MAX_S)))
        for suffix, start, end in parts:
            name = f"normal/{key.replace('#', '_')}{suffix}.mp4"
            _cut(raw(key, "normal"), start, end, TRAIN / "clips" / name)
            labels.append({"video": name, "category": "normal", "lighting": "", "event_start_s": "",
                           "event_end_s": "", "fold": "T",
                           "notes": "web street footage (pre-screened, not reviewed frame by frame)"})
            manifest.append({"clip": name, "source_id": key.split("#")[0], "segment": key,
                             "source_start_s": f"{start:.2f}", "source_end_s": f"{end:.2f}"})
    for name, rows in (("labels.csv", labels), ("manifest.csv", manifest)):
        with open(TRAIN / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    n_ev = sum(1 for r in labels if r["category"] != "normal")
    print(f"{n_ev} snatch clips, {len(labels) - n_ev} normal clips -> {TRAIN}")


def _features(path: Path, es: float, ee: float) -> list:
    """ORB keypoints on 5 frames around an event, frame borders masked (logos, captions, tickers)."""
    cap = cv2.VideoCapture(str(path))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1.0)
    orb, out = cv2.ORB_create(1500), []
    for t in (es - 1.5, (es + ee) / 2, ee + 1.0, max(0.2, es - 5.0), min(dur - 0.2, ee + 4.0)):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0.0) * 1000)
        ok, img = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (640, int(g.shape[0] * 640 / g.shape[1])))
        h, w = g.shape
        mask = np.zeros_like(g)
        mask[int(0.12 * h):int(0.85 * h), int(0.08 * w):int(0.92 * w)] = 255
        kp, des = orb.detectAndCompute(g, mask)
        if des is not None and len(kp) >= 30:
            out.append((np.float32([k.pt for k in kp]), des))
    return out


def _inliers(a: list, b: list, bf) -> int:
    best = 0
    for pa, da in a:
        for pb, db in b:
            good = [m[0] for m in bf.knnMatch(da, db, k=2) if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
            if len(good) >= 15:
                _, inl = cv2.findHomography(pa[[m.queryIdx for m in good]], pb[[m.trainIdx for m in good]],
                                            cv2.RANSAC, 4.0)
                if inl is not None:
                    best = max(best, int(inl.sum()))
    return best


def _middle(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    return cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1.0) / 2


def leaks() -> None:
    """Match every pool event against every test event (Snatch-Web test set, Snatch 1.0), and every pool normal
    clip against every Snatch-Web test clip (its middle frames for a normal one); keep the two best."""
    from snatchguard.core.config import REPO_ROOT
    s1 = REPO_ROOT / "data" / "samples" / "snatch1.0" / "Chain Snatching Videos"
    tests, web = [], []
    for name, labels, root in (("WEB", DATASET / "labels.csv", DATASET / "clips"),
                               ("S1", REPO_ROOT / "data" / "labels" / "snatch1.0.csv", s1)):
        with open(labels, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["category"] != "normal" and r["event_start_s"]:
                    es = float(r["event_start_s"])
                    tests.append((f"{name}:{r['video']}", _features(root / r["video"], es, float(r["event_end_s"] or es))))
                elif r["category"] == "normal" and name == "WEB":
                    mid = _middle(root / r["video"])
                    web.append((f"{name}:{r['video']}", _features(root / r["video"], mid, mid)))
    web += [t for t in tests if t[0].startswith("WEB:")]
    res = _load(LEAKS)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    with open(TRAIN / "labels.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["video"] not in res]
    for r in rows:
        if r["category"] == "normal":
            es = ee = _middle(TRAIN / "clips" / r["video"])
            against = web
        else:
            es, against = float(r["event_start_s"]), tests
            ee = float(r["event_end_s"] or es)
        mine = _features(TRAIN / "clips" / r["video"], es, ee)
        best = sorted(((_inliers(mine, feats, bf), key) for key, feats in against), reverse=True)[:2]
        res[r["video"]] = {"best": best}
        flag = "  <-- same footage: upload excluded" if best and best[0][0] >= LEAK_INLIERS else ""
        print(f"{r['video']}: {best}{flag}", flush=True)
        LEAKS.write_text(json.dumps(res, indent=1), encoding="utf-8")


def bag_sheets(package: Path, per_sheet: int = 4) -> None:
    """Review images of every unchecked pool event's window clips (frames 4, 6, 8, 10 of each strip: the onset
    to 1.5 s after it), numbered [event.clip]; index.json maps the numbers to clip videos and onset times."""
    from snatchguard.verifier.finetune import load_strip
    idx = json.loads((package / "index.json").read_text())
    done = _load(BAGS)
    events = [v for v in idx["videos"] if v["set"] == "POOL" and v["category"] != "normal"
              and v["key"].split(":", 1)[1] not in done and any(c["bag"] for c in v["clips"])]
    out_dir = REVIEW / "bag_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for s in range(0, len(events), per_sheet):
        rows = []
        for e, v in enumerate(events[s:s + per_sheet], s):
            tiles = []
            for c_i, c in enumerate(c for c in v["clips"] if c["bag"]):
                frames = load_strip(str(package / "strips" / c["strip"]))
                strip = np.hstack([frames[k][:, :, ::-1] for k in (4, 6, 8, 10)])
                strip = np.ascontiguousarray(strip)
                cv2.rectangle(strip, (0, 0), (150, 16), (0, 0, 0), -1)
                cv2.putText(strip, f"[{e}.{c_i}] t={c['t']:.1f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                tiles.append(np.pad(strip, ((0, 0), (0, 6), (0, 0))))
                index.append({"n": f"{e}.{c_i}", "video": v["key"].split(":", 1)[1], "t": c["t"]})
            while len(tiles) % 3:
                tiles.append(np.zeros_like(tiles[0]))
            head = np.zeros((18, 3 * tiles[0].shape[1], 3), np.uint8)
            ev = v["event"]
            cv2.putText(head, f"event {e}: {v['key'].split(':', 1)[1]}  grab {ev[0]:.1f}-{ev[1]:.1f} s", (4, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            rows.append(head)
            rows += [np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)]
        cv2.imwrite(str(out_dir / f"bags_{s // per_sheet:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out_dir / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"{len(events)} events -> {(len(events) + per_sheet - 1) // per_sheet} sheets in {out_dir}")


def pool_config() -> dict:
    """The live perception without pose: the verifier's candidates need detections and tracks only."""
    from snatchguard.core.config import REPO_ROOT, load_yaml
    cfg = load_yaml(str(REPO_ROOT / "configs" / "edge.yaml"))
    cfg["models"] = dict(cfg["models"], pose=None)
    return cfg


def cache() -> None:
    from snatchguard.evaluation import cache as cache_mod
    cfg = pool_config()
    with open(TRAIN / "labels.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: r["category"] == "normal")          # events first
    for n, r in enumerate(rows, 1):
        path = str(TRAIN / "clips" / r["video"])
        if cache_mod.cache_path(path, cache_mod.perception_tag(cfg, 8.0, None)).exists():
            continue
        cv = cache_mod.get(path, cfg, 8.0)
        print(f"[{n}/{len(rows)}] {r['video']}: {cv.duration_s:.0f} s at {cv.build_fps:.1f} fps", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sh = sub.add_parser("sheets")
    sh.add_argument("--tag", default="")
    sh.add_argument("--limit", type=int, default=0, help="only the N most promising segments")
    fn = sub.add_parser("fine")
    fn.add_argument("key")
    fn.add_argument("start", type=float)
    fn.add_argument("end", type=float)
    fn.add_argument("--out", type=Path, default=REVIEW / "train_fine.jpg")
    sub.add_parser("build")
    sub.add_parser("leaks")
    sub.add_parser("cache")
    bg = sub.add_parser("bags")
    bg.add_argument("package", type=Path)
    a = ap.parse_args()
    if a.cmd == "sheets":
        sheets(tag=a.tag, limit=a.limit)
    elif a.cmd == "fine":
        fine("snatch", a.key, a.start, a.end, a.out)
    elif a.cmd == "build":
        build()
    elif a.cmd == "leaks":
        leaks()
    elif a.cmd == "bags":
        bag_sheets(a.package.expanduser())
    else:
        cache()


if __name__ == "__main__":
    main()
