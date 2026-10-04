"""List / fetch individual videos from the Snatch 1.0 dataset (Zenodo) without downloading whole archives.

The dataset is 7 zip parts (12.5 GB). Zip files keep their table of contents at
the end, so with HTTP range requests we can list every part and pull out single
videos.

    python scripts/snatch_dataset.py download --to D:/downloads        # all 7 parts, resumable
    python scripts/snatch_dataset.py extract --zips D:/downloads        # unpack into data/samples/snatch1.0
    python scripts/snatch_dataset.py import --from /media/usb/snatch      # zips, folders or loose videos
    python scripts/snatch_dataset.py verify                              # every labelled video present?
    python scripts/snatch_dataset.py list
    python scripts/snatch_dataset.py get "Snatch 1.0/Chain Snatching Videos/Snatch Theft/17_0.mp4"
    python scripts/snatch_dataset.py get-all --category "Snatch Theft" --max 10

Dataset: Roy et al., "Snatch theft detection in unconstrained surveillance videos
using action attribute modelling", Pattern Recognition Letters 2018. CC-BY-SA 4.0.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import httpx

RECORD = "https://zenodo.org/records/10812645/files/Snatch%201.0-20201231T071009Z-00{n}.zip?download=1"
PARTS = range(1, 8)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "samples" / "snatch1.0"
INDEX = ROOT / "data" / "raw" / "snatch_index.json"


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url: str, client: httpx.Client) -> None:
        self.url, self.client = url, client
        for attempt in range(8):
            try:
                r = client.head(url, follow_redirects=True)
                r.raise_for_status()
                break
            except httpx.HTTPError:
                if attempt == 7:
                    raise
                time.sleep(3 * (attempt + 1))
        self.size = int(r.headers["content-length"])
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        for attempt in range(8):
            try:
                r = self.client.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, follow_redirects=True)
                r.raise_for_status()
                data = r.content
                self.pos += len(data)
                return data
            except httpx.HTTPError:
                time.sleep(2 * (attempt + 1))
        raise IOError(f"range read failed: {self.url} {self.pos}-{end}")

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def build_index(client: httpx.Client) -> dict:
    if INDEX.exists():
        return json.loads(INDEX.read_text())
    index = {}
    for n in PARTS:
        with zipfile.ZipFile(HttpRangeFile(RECORD.format(n=n), client)) as zf:
            for info in zf.infolist():
                if not info.is_dir():
                    index[info.filename] = {"part": n, "size": info.file_size}
        print(f"part {n}: {sum(1 for v in index.values() if v['part'] == n)} files")
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(index, indent=1))
    return index


def fetch(name: str, index: dict, client: httpx.Client) -> Path:
    dest = OUT / Path(name).relative_to(Path(name).parts[0])
    if dest.exists() and dest.stat().st_size == index[name]["size"]:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    rf = io.BufferedReader(HttpRangeFile(RECORD.format(n=index[name]["part"]), client), buffer_size=8 << 20)
    with zipfile.ZipFile(rf) as zf, zf.open(name) as src, open(dest, "wb") as out:
        while chunk := src.read(8 << 20):
            out.write(chunk)
    print(f"saved {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def download_all(dest: Path) -> None:
    """All 7 archive parts, resuming partial files after connection drops."""
    dest.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=httpx.Timeout(60.0, connect=20.0), follow_redirects=True) as client:
        for n in PARTS:
            url = RECORD.format(n=n)
            out = dest / f"Snatch 1.0-20201231T071009Z-00{n}.zip"
            total = int(client.head(url).headers["content-length"])
            for attempt in range(200):
                have = out.stat().st_size if out.exists() else 0
                if have >= total:
                    break
                print(f"part {n}: {have / 1e9:.2f} / {total / 1e9:.2f} GB (attempt {attempt + 1})", flush=True)
                try:
                    with client.stream("GET", url, headers={"Range": f"bytes={have}-"}) as r, open(out, "ab") as f:
                        for chunk in r.iter_bytes(8 << 20):
                            f.write(chunk)
                except httpx.HTTPError as exc:
                    print(f"  interrupted ({exc}); resuming", flush=True)
                    time.sleep(5)
            print(f"part {n}: complete", flush=True)


def extract(zips: Path) -> None:
    """Unpack the downloaded parts into data/samples/snatch1.0 (skips files already there)."""
    parts = sorted(zips.glob("Snatch 1.0-*.zip")) or sorted(zips.glob("*.zip"))
    if not parts:
        raise SystemExit(f"no Snatch 1.0 zip files in {zips}")
    for z in parts:
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                rel = Path(info.filename)
                rel = rel.relative_to(rel.parts[0]) if rel.parts[0].startswith("Snatch") else rel
                dest = OUT / rel
                if dest.exists() and dest.stat().st_size == info.file_size:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(dest, "wb") as out:
                    while chunk := src.read(8 << 20):
                        out.write(chunk)
                print(f"extracted {rel}", flush=True)
    print(f"done -> {OUT}")


def import_videos(src: Path, labels: Path, move: bool = False) -> None:
    """Bring Snatch 1.0 videos in from anywhere: zip parts, an extracted folder or loose files.
    Videos are recognised by file name (every labelled name is unique) and placed in the expected layout."""
    import csv
    import shutil
    for folder in sorted({z.parent for z in src.rglob("*.zip")}):
        extract(folder)
    with open(labels, newline="", encoding="utf-8") as f:
        wanted = {Path(r["video"]).name: r["video"] for r in csv.DictReader(f)}
    placed = skipped = 0
    for path in src.rglob("*"):
        if path.suffix.lower() not in (".avi", ".mp4", ".mkv", ".mov") or path.name not in wanted:
            continue
        dest = OUT / "Chain Snatching Videos" / wanted[path.name]
        if dest.exists() and dest.stat().st_size == path.stat().st_size:
            skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        (shutil.move if move else shutil.copy2)(str(path), dest)
        placed += 1
        print(f"{'moved' if move else 'copied'} {wanted[path.name]}", flush=True)
    print(f"{placed} placed, {skipped} already present")


def verify(labels: Path, deep: bool = False) -> bool:
    import csv
    with open(labels, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = [r["video"] for r in rows if not (OUT / "Chain Snatching Videos" / r["video"]).exists()]
    for m in missing:
        print(f"missing: {m}")
    print(f"{len(rows) - len(missing)}/{len(rows)} labelled videos present in {OUT}")
    broken = []
    if deep:
        import cv2
        total_h = 0.0
        for r in rows:
            path = OUT / "Chain Snatching Videos" / r["video"]
            if not path.exists():
                continue
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            ok_all = cap.isOpened() and fps > 0 and n > 0
            for frac in (0.0, 0.5, 0.9):
                if not ok_all:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac))
                ok_all = cap.read()[0]
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            dur = n / fps if fps else 0
            total_h += dur / 3600
            print(f"  {'ok ' if ok_all else 'BAD'} {r['video']:28s} {w}x{h} {fps:5.1f} fps {dur / 60:6.1f} min  [{r['category']}, fold {r.get('fold', '')}]")
            if not ok_all and r["category"] in ("snatch_bike", "snatch_foot", "normal"):
                broken.append(r["video"])         # out-of-scope clips are never used, so they don't count
        print(f"{total_h:.1f} h of video; {len(broken)} unreadable{': ' + ', '.join(broken) if broken else ''}")
    return not missing and not broken


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    dl = sub.add_parser("download")
    dl.add_argument("--to", type=Path, required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--zips", type=Path, required=True)
    im = sub.add_parser("import")
    im.add_argument("--from", dest="src", type=Path, required=True)
    im.add_argument("--move", action="store_true", help="move instead of copy (saves disk space)")
    im.add_argument("--labels", type=Path, default=ROOT / "data" / "labels" / "snatch1.0.csv")
    ve = sub.add_parser("verify")
    ve.add_argument("--labels", type=Path, default=ROOT / "data" / "labels" / "snatch1.0.csv")
    ve.add_argument("--deep", action="store_true", help="also open every video and read frames from it")
    sub.add_parser("list")
    g = sub.add_parser("get")
    g.add_argument("names", nargs="+")
    ga = sub.add_parser("get-all")
    ga.add_argument("--category", default="Snatch", help="folder name to match, e.g. Snatch or Normal")
    ga.add_argument("--max", type=int, default=5)
    ga.add_argument("--max-mb", type=float, default=400, help="skip files bigger than this")
    args = ap.parse_args()
    if args.cmd == "download":
        return download_all(args.to)
    if args.cmd == "extract":
        return extract(args.zips)
    if args.cmd == "import":
        import_videos(args.src, args.labels, args.move)
        raise SystemExit(0 if verify(args.labels) else 1)
    if args.cmd == "verify":
        raise SystemExit(0 if verify(args.labels, args.deep) else 1)

    with httpx.Client(timeout=httpx.Timeout(120.0, connect=20.0)) as client:
        index = build_index(client)
        if args.cmd == "list":
            for name, v in sorted(index.items()):
                print(f"part {v['part']}  {v['size'] / 1e6:8.1f} MB  {name}")
        elif args.cmd == "get":
            for name in args.names:
                fetch(name, index, client)
        else:
            picks = [n for n, v in sorted(index.items(), key=lambda kv: kv[1]["size"])
                     if f"/{args.category}/" in n and v["size"] / 1e6 <= args.max_mb][: args.max]
            for name in picks:
                fetch(name, index, client)


if __name__ == "__main__":
    main()
