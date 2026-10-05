"""Stage 2: download prioritised candidates (video stream only, <= 720p), slowly and resumably.

Snatch candidates: titles that mention CCTV / camera footage first, 10 s - 5 min long.
Normal candidates: only the first `--normal-minutes` of each video (bundled ffmpeg).
Metadata (title, channel, upload date, licence, URL) is kept next to each file for provenance.

    python -m snatchguard.webdata.download --kind snatch --max 400
    python -m snatchguard.webdata.download --kind normal --max 60 --normal-minutes 10
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

from snatchguard.webdata.discover import DATASET

CCTV_WORDS = re.compile(r"cctv|camera|caught|captured|footage|video|सीसीटीवी|कैमरे|కెమెరా|సీసీ|சிசிடிவி|கேமரா|ಸಿಸಿಟಿವಿ", re.I)


def priority(item: dict) -> tuple:
    title = item["title"]
    return (0 if CCTV_WORDS.search(title) else 1,
            0 if re.search(r"bike|scooty|motorcycle|बाइक|బైక్|பைக்", title, re.I) else 1,
            -(item.get("view_count") or 0))


def main() -> None:
    import imageio_ffmpeg
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["snatch", "normal"], required=True)
    ap.add_argument("--max", type=int, default=300)
    ap.add_argument("--normal-minutes", type=float, default=10)
    args = ap.parse_args()

    cands = [json.loads(l) for l in (DATASET / "meta" / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    cands = [c for c in cands if c["kind"] == args.kind]
    if args.kind == "snatch":
        cands = [c for c in cands if 10 <= (c.get("duration") or 0) <= 300]
    else:
        cands = [c for c in cands if (c.get("duration") or 0) >= 120]
    cands.sort(key=priority)
    out_dir = DATASET / "raw" / args.kind
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = DATASET / "meta" / f"downloaded_{args.kind}.txt"
    urls = [c["url"] for c in cands[: args.max]]
    (DATASET / "meta" / f"queue_{args.kind}.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
    print(f"{len(cands)} eligible {args.kind} candidates; downloading up to {len(urls)}", flush=True)

    cmd = [sys.executable, "-m", "yt_dlp", "-a", str(DATASET / "meta" / f"queue_{args.kind}.txt"),
           "-f", "bv*[height<=720][vcodec^=avc1]/bv*[height<=720][vcodec^=avc]/b[height<=720][vcodec^=avc1]",
           "-o", str(out_dir / "%(id)s.%(ext)s"), "--write-info-json", "--no-playlist",
           "--download-archive", str(archive), "--ignore-errors", "--no-overwrites",
           "--sleep-interval", "2", "--max-sleep-interval", "6", "--retries", "5",
           "--ffmpeg-location", imageio_ffmpeg.get_ffmpeg_exe(), "--no-warnings", "--progress", "--newline"]
    if args.kind == "normal":
        cmd += ["--download-sections", f"*0-{int(args.normal_minutes * 60)}"]
    subprocess.run(cmd)
    print("done", flush=True)


if __name__ == "__main__":
    main()
