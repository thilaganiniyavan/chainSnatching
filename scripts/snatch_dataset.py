"""List / fetch individual videos from the Snatch 1.0 dataset (Zenodo) without downloading whole archives.

The dataset is 7 zip parts (12.5 GB). Zip files keep their table of contents at
the end, so with HTTP range requests we can list every part and pull out single
videos.

    python scripts/snatch_dataset.py list
    python scripts/snatch_dataset.py get "Snatch 1.0/Chain Snatching Videos/Snatch/s_10.avi"
    python scripts/snatch_dataset.py get-all --category Snatch --max 10

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


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    g = sub.add_parser("get")
    g.add_argument("names", nargs="+")
    ga = sub.add_parser("get-all")
    ga.add_argument("--category", default="Snatch", help="folder name to match, e.g. Snatch or Normal")
    ga.add_argument("--max", type=int, default=5)
    ga.add_argument("--max-mb", type=float, default=400, help="skip files bigger than this")
    args = ap.parse_args()

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
