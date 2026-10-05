"""Stage 2: download prioritised candidates (video stream only, <= 720p), slowly and resumably.

Snatch candidates: titles that mention CCTV / camera footage first, 10 s - 5 min long.
Normal candidates: videos up to `--normal-max-minutes` long, downloaded whole (the build step uses at most
10 minutes of each).
Metadata (title, channel, upload date, licence, URL) is kept next to each file for provenance.

    python -m snatchguard.webdata.download --kind snatch --max 400
    python -m snatchguard.webdata.download --kind normal --max 80
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

from snatchguard.webdata.discover import DATASET, source_of

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
    ap.add_argument("--normal-max-minutes", type=float, default=30,
                    help="normal videos up to this long are downloaded whole (ffmpeg section cuts are unreliable)")
    ap.add_argument("--source", choices=["any", "raw", "news"], default="any",
                    help="raw = police / owner / CCTV-channel uploads only (news packages are edited)")
    args = ap.parse_args()

    cands = [json.loads(l) for l in (DATASET / "meta" / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    cands = [c for c in cands if c["kind"] == args.kind]
    if args.source != "any":
        cands = [c for c in cands if source_of(c) == args.source]
    cands = [c for c in cands if not re.search(r"\bai\b", c.get("channel") or "", re.I)]   # no AI-named channels
    if args.kind == "snatch":
        cands = [c for c in cands if 10 <= (c.get("duration") or 0) <= 300]
    else:
        cands = [c for c in cands if 120 <= (c.get("duration") or 0) <= args.normal_max_minutes * 60]
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
    subprocess.run(cmd)
    print("done", flush=True)


if __name__ == "__main__":
    main()
