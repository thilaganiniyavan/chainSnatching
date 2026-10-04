"""Everything from raw data to the acceptance report, in one resumable command (Windows or Linux, NVIDIA GPU).

    python -m training.run_all                          # full profile (YOLO26s, about 12-18 h on an RTX 4060)
    python -m training.run_all --profile quick          # YOLO26n, smaller data (about 4-6 h): a first result
    python -m training.run_all --zips D:/downloads      # also unpack the Snatch 1.0 zips first
    python -m training.run_all --baseline               # also measure the pretrained models (before/after table)
    python -m training.run_all --from cache             # re-run from a step (e.g. after editing rules.yaml)
    python -m training.run_all --only report

Steps (each writes runtime/pipeline/<step>.done and is skipped next time unless --force):
  check      GPU, disk space, Python packages
  data       Snatch 1.0 present (optionally extracted from --zips)
  models     pretrained YOLO26 detector / pose and the YOLO26x teacher
  baseline   (optional) caches + tuning with the pretrained models, for the before/after comparison
  datasets   ExDark + LLVIP + VisDrone + COCO, with night and CCTV-degradation variants
  pseudo     teacher pseudo-labels on the Snatch 1.0 `pseudo` fold (never the evaluation videos)
  train_det  fine-tune the detector
  compare    day vs night mAP, pretrained vs fine-tuned
  train_pose fine-tune the pose model
  install    copy the trained weights into models/ and write configs/edge.gpu.yaml
  cache      run perception once over every evaluation video
  tune       cross-validated rule tuning against the 90 % targets
  night      the same held-out footage as simulated IR / low-light night, scored with the tuned rules
  report     acceptance report (runtime/tuning/acceptance_report.md)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "pipeline"
DATASETS = ROOT / "data" / "datasets"
RUNS = ROOT / "runtime" / "train_runs"
MODELS = ROOT / "models"

PROFILES = {
    "full": dict(det_size="s", det_imgsz=960, det_epochs=50, pose_size="s", pose_imgsz=640, pose_epochs=30,
                 coco=12000, visdrone=6000, llvip=6000, exdark=True, pose_images=15000, pseudo_every=2.0,
                 pseudo_max=400, teacher="yolo26x.pt", edge_imgsz=960, pose_infer_imgsz=960, trials=150,
                 device="0", cpu_ok=False, night_max_seconds=600),
    "quick": dict(det_size="n", det_imgsz=640, det_epochs=25, pose_size="n", pose_imgsz=640, pose_epochs=15,
                  coco=4000, visdrone=2000, llvip=2000, exdark=True, pose_images=5000, pseudo_every=4.0,
                  pseudo_max=150, teacher="yolo26x.pt", edge_imgsz=768, pose_infer_imgsz=768, trials=80,
                  device="0", cpu_ok=False, night_max_seconds=600),
    # Plumbing test only (any machine, CPU is fine): tiny data, 1 epoch. Proves every step runs.
    "smoke": dict(det_size="n", det_imgsz=320, det_epochs=1, pose_size="n", pose_imgsz=320, pose_epochs=1,
                  coco=40, visdrone=0, llvip=0, exdark=False, pose_images=30, pseudo_every=30.0, pseudo_max=2,
                  teacher="yolo26n.pt", edge_imgsz=320, pose_infer_imgsz=320, trials=3, device="cpu",
                  cpu_ok=True, cache_max_seconds=60),
}
STEPS = ["check", "data", "models", "baseline", "datasets", "pseudo", "train_det", "compare", "train_pose",
         "install", "cache", "tune", "night", "report"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(step: str, args: list[str]) -> None:
    """Run a module/script with live output, also written to runtime/pipeline/logs/<step>.log."""
    (STATE / "logs").mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, *args]
    log(f"$ {' '.join(cmd)}")
    with open(STATE / "logs" / f"{step}.log", "a", encoding="utf-8") as lf:
        lf.write(f"\n=== {time.ctime()} {' '.join(cmd)}\n")
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1, env=env)
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
        if proc.wait() != 0:
            raise SystemExit(f"step '{step}' failed (exit {proc.returncode}); see {STATE / 'logs' / (step + '.log')}")


def edge_config(det: str, pose: str, prof: dict, rules: str = "configs/rules.yaml") -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "edge.yaml").read_text(encoding="utf-8"))
    cfg["models"].update({"detector": det, "pose": pose, "imgsz": prof["edge_imgsz"],
                          "pose_imgsz": prof["pose_infer_imgsz"], "device": "auto", "pose_mode": "full",
                          "max_pose_crops": 4})
    cfg["processing"]["target_fps"] = 10
    cfg["rules"] = rules
    cfg["central"]["url"] = "http://LAPTOP-IP:8000"
    cfg["cameras"] = [{"id": "cam-01", "name": "Snatch 1.0 replay", "source": "playlist:demo", "loop": True,
                       "realtime": True, "lighting": "auto", "simulate_night": "none"}]
    return cfg


def write_yaml(path: Path, cfg: dict, header: str) -> None:
    path.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


# ---------------------------------------------------------------------------------------------------

def step_check(a, prof) -> None:
    try:
        import torch
    except ImportError:
        raise SystemExit("PyTorch is not installed: run scripts/windows/setup.ps1 (or see README)")
    if not torch.cuda.is_available():
        if prof["cpu_ok"]:
            log("no CUDA GPU: fine for the smoke profile (plumbing test only)")
            return
        raise SystemExit("PyTorch cannot see a CUDA GPU. Install the CUDA build of PyTorch "
                         "(scripts/windows/setup.ps1 does this) and an up-to-date NVIDIA driver.")
    gpu = torch.cuda.get_device_properties(0)
    free = shutil.disk_usage(ROOT).free / 1e9
    log(f"GPU {gpu.name}, {gpu.total_memory / 1e9:.1f} GB; torch {torch.__version__}; free disk {free:.0f} GB")
    if free < 60:
        log("WARNING: under 60 GB free. The full run needs about 50 GB more (training sets ~25 GB, videos 12 GB, "
            "models and caches); free some space or it may stop with 'No space left on device'.")
    import ultralytics
    log(f"ultralytics {ultralytics.__version__}")


def step_data(a, prof) -> None:
    if a.zips:
        run("data", ["scripts/snatch_dataset.py", "extract", "--zips", str(a.zips)])
    try:
        run("data", ["scripts/snatch_dataset.py", "verify"])
    except SystemExit:
        if not prof["cpu_ok"]:
            raise
        log("some videos missing: fine for the smoke profile")


def step_models(a, prof) -> None:
    MODELS.mkdir(exist_ok=True)
    from ultralytics import YOLO
    for name in {f"yolo26{prof['det_size']}.pt", f"yolo26{prof['pose_size']}-pose.pt", prof["teacher"]}:
        target = MODELS / name
        if not target.exists():
            YOLO(name)                                  # downloads into the working directory
            shutil.move(name, target) if Path(name).exists() else None
        log(f"model ready: {target}")


def step_baseline(a, prof) -> None:
    cfg = edge_config(f"models/yolo26{prof['det_size']}.pt", f"models/yolo26{prof['pose_size']}-pose.pt", prof)
    path = ROOT / "configs" / "edge.baseline.yaml"
    write_yaml(path, cfg, "# Generated by training/run_all.py: pretrained (not fine-tuned) models, for comparison.\n")
    run("baseline", ["-m", "snatchguard.evaluation.cache", "--config", str(path)])
    run("baseline", ["-m", "snatchguard.evaluation.tune", "--config", str(path), "--trials", str(prof["trials"]),
                     "--out", str(ROOT / "runtime" / "tuning_baseline")])
    run("baseline", ["-m", "snatchguard.evaluation.acceptance",
                     "--result", str(ROOT / "runtime" / "tuning_baseline" / "result.json"),
                     "--out", str(ROOT / "runtime" / "tuning_baseline" / "acceptance_report.md")])


def step_datasets(a, prof) -> None:
    extra = ([] if prof["exdark"] else ["--no-exdark"]) + ([] if prof["llvip"] else ["--no-llvip"])
    run("datasets", ["training/prepare_dataset.py", "--out", str(DATASETS), "--coco-images", str(prof["coco"]),
                     "--visdrone-images", str(prof["visdrone"]), "--llvip-images", str(prof["llvip"]),
                     "--pose-images", str(prof["pose_images"]), *extra])


def step_pseudo(a, prof) -> None:
    run("pseudo", ["training/pseudo_label.py", "--out", str(DATASETS / "sg_det"), "--teacher",
                   str(MODELS / prof["teacher"]), "--every", str(prof["pseudo_every"]), "--max-per-video",
                   str(prof["pseudo_max"]), "--night-aug"])


def step_train_det(a, prof) -> None:
    run("train_det", ["training/train.py", "det", "--data", str(DATASETS / "sg_det"), "--size", prof["det_size"],
                      "--imgsz", str(prof["det_imgsz"]), "--epochs", str(prof["det_epochs"]),
                      "--batch", "-1" if prof["device"] != "cpu" else "8",
                      "--device", prof["device"], "--project", str(RUNS), "--export-to", str(RUNS / "export")])


def step_compare(a, prof) -> None:
    run("compare", ["training/train.py", "compare", "--kind", "det", "--size", prof["det_size"], "--data",
                    str(DATASETS / "sg_det"), "--imgsz", str(prof["det_imgsz"]), "--weights",
                    str(RUNS / "det" / "weights" / "best.pt"), "--project", str(RUNS)])


def step_train_pose(a, prof) -> None:
    run("train_pose", ["training/train.py", "pose", "--data", str(DATASETS / "sg_pose"), "--size", prof["pose_size"],
                       "--imgsz", str(prof["pose_imgsz"]), "--epochs", str(prof["pose_epochs"]),
                       "--batch", "-1" if prof["device"] != "cpu" else "8",
                       "--device", prof["device"], "--project", str(RUNS), "--export-to", str(RUNS / "export")])


def step_install(a, prof) -> None:
    det, pose = RUNS / "det" / "weights" / "best.pt", RUNS / "pose" / "weights" / "best.pt"
    for p in (det, pose):
        if not p.exists():
            raise SystemExit(f"missing {p}: training did not finish")
    shutil.copy(det, MODELS / "det_finetuned.pt")
    shutil.copy(pose, MODELS / "pose_finetuned.pt")
    cfg = edge_config("models/det_finetuned.pt", "models/pose_finetuned.pt", prof)
    write_yaml(ROOT / "configs" / "edge.gpu.yaml", cfg,
               "# Generated by training/run_all.py: fine-tuned models on the NVIDIA GPU.\n"
               "# Set central.url to the laptop running the dashboard (or pass --central).\n")
    log("installed models/det_finetuned.pt, models/pose_finetuned.pt -> configs/edge.gpu.yaml")


def step_cache(a, prof) -> None:
    cap = ["--max-seconds", str(prof["cache_max_seconds"])] if prof.get("cache_max_seconds") else []
    run("cache", ["-m", "snatchguard.evaluation.cache", "--config", "configs/edge.gpu.yaml", *cap])


def step_tune(a, prof) -> None:
    run("tune", ["-m", "snatchguard.evaluation.tune", "--config", "configs/edge.gpu.yaml",
                 "--trials", str(a.trials or prof["trials"]), "--target", str(a.target)])


def step_night(a, prof) -> None:
    cap = str(prof.get("cache_max_seconds") or prof["night_max_seconds"])
    for mode in ("ir", "lowlight"):
        run("night", ["-m", "snatchguard.evaluation.cache", "--config", "configs/edge.gpu.yaml", "--night", mode,
                      "--max-seconds", cap])
    run("night", ["-m", "snatchguard.evaluation.night", "--config", "configs/edge.gpu.yaml",
                  "--rules", str(ROOT / "runtime" / "tuning" / "rules.tuned.yaml")])


def step_report(a, prof) -> None:
    run("report", ["-m", "snatchguard.evaluation.acceptance"])
    tuned = ROOT / "runtime" / "tuning" / "rules.tuned.yaml"
    if tuned.exists():
        shutil.copy(tuned, ROOT / "configs" / "rules.tuned.yaml")
        cfg_path = ROOT / "configs" / "edge.gpu.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        cfg["rules"] = "configs/rules.tuned.yaml"
        write_yaml(cfg_path, cfg, "# Generated by training/run_all.py: fine-tuned models + tuned rules.\n")
        log("deployed configs/rules.tuned.yaml in configs/edge.gpu.yaml")
    res = json.loads((ROOT / "runtime" / "tuning" / "result.json").read_text(encoding="utf-8"))
    cv = res["cross_validated"]
    log(f"RESULT (cross-validated): recall {cv['recall']:.1%} {cv['recall_ci95']}, precision {cv['precision']:.1%} "
        f"{cv['precision_ci95']}, {cv['false_alarms_per_hour']:.2f} false alarms/h -> "
        f"{'PASS' if res['passes'] else 'NOT MET'}")
    log(f"report: {ROOT / 'runtime' / 'tuning' / 'acceptance_report.md'}")


FUNCS = {s: globals()[f"step_{s}"] for s in STEPS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", choices=list(PROFILES), default="full")
    ap.add_argument("--zips", type=Path, help="folder with the 7 downloaded Snatch 1.0 zip parts")
    ap.add_argument("--baseline", action="store_true", help="also evaluate the pretrained models first")
    ap.add_argument("--from", dest="start", choices=STEPS)
    ap.add_argument("--only", choices=STEPS)
    ap.add_argument("--force", action="store_true", help="re-run steps already marked done")
    ap.add_argument("--trials", type=int, help="tuning trials (default from profile)")
    ap.add_argument("--target", type=float, default=0.90)
    a = ap.parse_args()
    os.chdir(ROOT)                      # relative paths and Ultralytics downloads land in the repo
    prof = PROFILES[a.profile]
    STATE.mkdir(parents=True, exist_ok=True)

    steps = [a.only] if a.only else STEPS[STEPS.index(a.start):] if a.start else STEPS
    if "baseline" in steps and not a.baseline and not a.only:
        steps.remove("baseline")
    for s in steps:
        marker = STATE / f"{s}.done"
        always = s in ("check",)
        if marker.exists() and not (a.force or a.only or a.start == s or always):
            log(f"skip {s} (done {marker.read_text(encoding='utf-8').strip()}; --force to redo)")
            continue
        log(f"=== {s}")
        t0 = time.time()
        FUNCS[s](a, prof)
        marker.write_text(f"{time.ctime()} in {(time.time() - t0) / 60:.0f} min, profile {a.profile}", encoding="utf-8")
    log("all requested steps finished")


if __name__ == "__main__":
    main()
