"""Analyse one video file offline (deterministic, as fast as the CPU allows).

Frames are sampled at `--fps` (what the edge box would process), run through
the exact live pipeline, and every decision is printed with its evidence.

    python -m snatchguard.evaluation.analyse VIDEO
    python -m snatchguard.evaluation.analyse VIDEO --night ir --debug out.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from snatchguard.core.config import REPO_ROOT, RuleConfig, load_yaml
from snatchguard.core.types import Level
from snatchguard.edge.pipeline import FramePipeline, draw_debug
from snatchguard.sim.night import make_transform


@dataclass
class VideoResult:
    video: str
    duration_s: float
    frames: int
    proc_fps: float
    decisions: list[dict] = field(default_factory=list)
    lighting_modes: dict[str, int] = field(default_factory=dict)

    def alerts(self, levels=("CONFIRMED", "SUSPECT")) -> list[dict]:
        return [d for d in self.decisions if d["level"] in levels]


def _abs(p: str) -> str:
    return p if Path(p).is_absolute() else str(REPO_ROOT / p)


def build_pipeline(edge_cfg: dict, lighting: str = "auto", rules_path: str | None = None,
                   zone: list[float] | None = None) -> FramePipeline:
    models = dict(edge_cfg["models"])
    models["detector"] = _abs(models["detector"])
    models["pose"] = _abs(models["pose"]) if models.get("pose") else None
    rules = RuleConfig.load(_abs(rules_path or edge_cfg.get("rules", "configs/rules.yaml")))
    return FramePipeline(models, _abs(edge_cfg.get("tracker", "configs/tracker.yaml")), rules, lighting, zone)


def analyse_video(path: str, edge_cfg: dict, fps: float = 8.0, night: str | None = None, lighting: str = "auto",
                  debug_out: str | None = None, max_seconds: float | None = None,
                  start_s: float = 0.0, rules_path: str | None = None, progress: bool = False,
                  zone: list[float] | None = None) -> VideoResult:
    pipe = build_pipeline(edge_cfg, lighting, rules_path, zone)
    transform = make_transform(night)
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if start_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    step = max(1, round(src_fps / fps))
    writer = None
    result = VideoResult(path, 0.0, 0, 0.0)
    idx, t_start = 0, time.perf_counter()
    while True:
        ok = cap.grab()
        if not ok:
            break
        idx += 1
        if (idx - 1) % step:
            continue
        ok, img = cap.retrieve()
        if not ok:
            break
        t = start_s + (idx - 1) / src_fps
        if max_seconds is not None and t - start_s > max_seconds:
            break
        if img.shape[1] > 1280:
            img = cv2.resize(img, (1280, int(img.shape[0] * 1280 / img.shape[1])), interpolation=cv2.INTER_AREA)
        if transform is not None:
            img = transform(img)
        res = pipe.process(img, t)
        result.frames += 1
        result.lighting_modes[res.mode.value] = result.lighting_modes.get(res.mode.value, 0) + 1
        for d in res.decisions:
            p = d.to_payload()
            p["video_time_s"] = round(t, 2)
            result.decisions.append(p)
        if debug_out:
            dbg = draw_debug(res.enhanced, pipe.engine, res.mode, zone=zone)
            for d in res.decisions:
                if d.level is not Level.DISMISSED:
                    cv2.putText(dbg, f"{d.level.value} {d.verdict.score:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 0, 255), 3)
            if writer is None:
                writer = cv2.VideoWriter(debug_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (dbg.shape[1], dbg.shape[0]))
            writer.write(dbg)
        if progress and result.frames % 200 == 0:
            el = time.perf_counter() - t_start
            print(f"  {path.split('/')[-1]}: {t:7.1f}s video, {result.frames / el:4.1f} fps, "
                  f"{len(result.alerts())} alerts", flush=True)
        result.duration_s = t - start_s
    for d in pipe.engine.flush():          # the clip ended while a contact was still being judged
        payload = d.to_payload()
        payload["video_time_s"] = round(result.duration_s + start_s, 2)
        payload["flushed_at_end"] = True
        result.decisions.append(payload)
    if writer is not None:
        writer.release()
    result.proc_fps = result.frames / max(time.perf_counter() - t_start, 1e-6)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "edge.yaml"))
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--night", choices=["none", "ir", "lowlight"], default="none")
    ap.add_argument("--lighting", default="auto")
    ap.add_argument("--debug", help="write an annotated debug video here (.mp4)")
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--zone", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                    help="detection zone as frame fractions, e.g. 0 0 0.7 0.7")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    res = analyse_video(args.video, load_yaml(args.config), args.fps, args.night, args.lighting, args.debug,
                        args.max_seconds, args.start, progress=True, zone=args.zone)
    print(f"\n{res.video}: {res.duration_s:.1f}s analysed, {res.frames} frames at {res.proc_fps:.1f} fps, "
          f"lighting {res.lighting_modes}")
    for d in res.decisions:
        passed = [e["key"] for e in d["evidence"] if e["passed"]]
        print(f"  t={d['video_time_s']:6.1f}s {d['level']:9s} score={d['score']:.2f} conf={d['confidence']} "
              f"victim={d['victim_id']} bike={d['vehicle_id']} evidence={passed} penalties={d['penalties']}")
    if not res.decisions:
        print("  no contact encounters")
    if args.verbose:
        print(json.dumps(res.decisions, indent=1)[:4000])


if __name__ == "__main__":
    main()
