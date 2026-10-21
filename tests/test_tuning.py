"""Tuning pipeline end to end on synthetic caches: caches -> cross-validated tuning -> acceptance report."""

from __future__ import annotations

from pathlib import Path

import pytest

from snatchguard.core.config import RuleConfig
from snatchguard.evaluation import cache as cache_mod
from snatchguard.evaluation.acceptance import render
from snatchguard.evaluation.metrics import score, wilson
from snatchguard.evaluation.tune import Entry, run_tuning
from snatchguard.sim import synthetic as syn


def _cache(tmp: Path, name: str, sc, seed: int = 0) -> str:
    frames = list(sc.frames(seed=seed))
    cv = cache_mod.CachedVideo(name, "synthetic", 8.0, 8.0, frames[-1].t, frames)
    path = tmp / f"{name}.pkl.gz"
    cache_mod.save(cv, path)
    return str(path)


@pytest.fixture()
def entries(tmp_path):
    snatches = [("bike1", syn.snatch_from_behind(), "snatch_bike", "A"),
                ("bike2", syn.snatch_head_on(), "snatch_bike", "B"),
                ("foot1", syn.foot_snatch_from_behind(), "snatch_foot", "A"),
                ("foot2", syn.foot_snatch_head_on(), "snatch_foot", "B")]
    normals = [("n1", syn.passing_bike(), "A"), ("n2", syn.friends_chat_then_leave(), "B"),
               ("n3", syn.jogger_passes_close(), "A"), ("n4", syn.crowd_brush(), "B")]
    out = []
    for name, sc, cat, fold in snatches:
        out.append(Entry({"video": name, "category": cat, "fold": fold, "event_start_s": "", "event_end_s": ""},
                         _cache(tmp_path, name, sc)))
    for name, sc, fold in normals:
        out.append(Entry({"video": name, "category": "normal", "fold": fold}, _cache(tmp_path, name, sc)))
    return out


def test_cache_roundtrip_preserves_decisions(tmp_path):
    sc = syn.snatch_from_behind()
    frames = list(sc.frames(seed=0))
    path = tmp_path / "x.pkl.gz"
    cache_mod.save(cache_mod.CachedVideo("x", "t", 8.0, 8.0, frames[-1].t, frames), path)
    a = cache_mod.replay(cache_mod.load(path), RuleConfig.load())
    b = cache_mod.replay(cache_mod.CachedVideo("x", "t", 8.0, 8.0, frames[-1].t, list(sc.frames(seed=0))),
                         RuleConfig.load())
    assert [(d["level"], d["score"]) for d in a] == [(d["level"], d["score"]) for d in b]
    assert any(d["level"] == "CONFIRMED" for d in a)


def test_event_scoring():
    row = {"video": "v", "category": "snatch_bike", "event_start_s": "10", "event_end_s": "12"}
    hit = {"level": "CONFIRMED", "contact_start": 10.5, "contact_end": 10.9, "decided_at": 13.0}
    wrong_time = {"level": "SUSPECT", "contact_start": 40.0, "contact_end": 40.5, "decided_at": 42.0}
    normal_fa = {"level": "SUSPECT", "contact_start": 5.0, "contact_end": 5.5, "decided_at": 7.0}
    m = score({"v": (row, 60.0, [hit, wrong_time]),
               "n": ({"video": "n", "category": "normal"}, 3600.0, [normal_fa])}, "any")
    assert (m.events, m.events_detected, m.true_alerts, m.false_alarms) == (1, 1, 1, 2)
    assert m.false_alarms_per_hour == pytest.approx(1.0)
    assert score({"v": (row, 60.0, [hit, wrong_time])}, "confirmed").false_alarms == 0
    lo, hi = wilson(9, 10)
    assert 0.55 < lo < 0.6 and hi > 0.98


def test_cross_validated_tuning_and_report(entries, tmp_path):
    res = run_tuning(entries, RuleConfig.load(), trials=4, workers=2, target=0.9, out_dir=tmp_path / "tune")
    cv = res["cross_validated"]
    assert cv["events"] == 4
    assert 0.0 <= cv["recall"] <= 1.0 and 0.0 <= cv["precision"] <= 1.0
    assert Path(res["tuned_rules"]).exists()
    RuleConfig.load(res["tuned_rules"])                      # the deployed file loads
    report = render(res)
    assert "cross-validated" in report.lower() and "Every clip" in report
    # the scripted snatches are easy: the defaults already catch them without false alarms
    assert res["default_rules"]["recall"] == 1.0 and res["default_rules"]["false_alarms"] == 0


def test_report_includes_night_section(entries, tmp_path):
    res = run_tuning(entries, RuleConfig.load(), trials=2, workers=1, target=0.9, out_dir=tmp_path / "t")
    night = {"modes": {"ir": {**res["cross_validated"], "by_category": {"snatch_bike": {"events": 2, "caught": 1}}}}}
    text = render(res, night=night)
    assert "## Night" in text and "IR (infrared camera)" in text and "1/2" in text


# ---- tiled detection and training plumbing ----------------------------------------------------------

def test_tiles_cover_frame_with_overlap():
    from snatchguard.edge.perception import tile_regions

    tiles = tile_regions(1280, 720, 2, 2, 0.2)
    assert len(tiles) == 4
    assert min(t[0] for t in tiles) == 0 and max(t[2] for t in tiles) == 1280
    assert min(t[1] for t in tiles) == 0 and max(t[3] for t in tiles) == 720
    (ax1, _, ax2, _), (bx1, _, bx2, _) = tiles[0], tiles[1]
    assert bx1 < ax2 and abs((ax2 - bx1) / (ax2 - ax1) - 0.2) < 0.01
    assert tile_regions(300, 200, 1, 1, 0.2) == [(0, 0, 300, 200)]


def test_merge_tiled_keeps_one_box_per_object_across_passes():
    import numpy as np

    from snatchguard.edge.perception import merge_tiled

    rows = np.array([[100, 100, 150, 200, 0.9, 0, 0],    # full frame
                     [101, 99, 151, 201, 0.8, 0, 1],     # same person from a tile
                     [100, 100, 150, 200, 0.7, 2, 1],    # motorcycle at the same place: other class, kept
                     [110, 110, 140, 190, 0.6, 0, 1]])   # a second person inside the first one's box (a grab): kept
    out = merge_tiled(rows)
    assert len(out) == 3 and out.shape[1] == 6


def test_perception_tag_unchanged_without_tiles():
    cfg = {"models": {"detector": "models/x.pt", "pose": None, "imgsz": 512}, "tracker": "configs/tracker.yaml"}
    plain = cache_mod.perception_tag(cfg, 8.0)
    cfg["models"]["tiles"] = [2, 2]
    tiled = cache_mod.perception_tag(cfg, 8.0)
    assert tiled != plain and tiled.endswith("-t2x2")
    del cfg["models"]["tiles"]
    assert cache_mod.perception_tag(cfg, 8.0) == plain


class _Arr:
    def __init__(self, a):
        self.a = a

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self, rows):
        self.data = _Arr(rows)

    def __len__(self):
        return len(self.data.a)


class _BlobDetector:
    """Stand-in detector: one 'person' box around the bright pixels of whatever crop it is given."""

    def predict(self, img, **_):
        import numpy as np

        ys, xs = np.nonzero(img[:, :, 0] > 128)
        rows = np.zeros((0, 6), dtype=np.float32)
        if len(xs):
            rows = np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 0.9, 0]], dtype=np.float32)
        return [type("R", (), {"boxes": _Boxes(rows)})()]


def test_tiled_tracking_keeps_one_identity():
    pytest.importorskip("ultralytics")
    import numpy as np

    from snatchguard.core.config import REPO_ROOT
    from snatchguard.core.types import ObjClass
    from snatchguard.edge.perception import Perception

    p = object.__new__(Perception)
    p.det, p.tracker_cfg, p.imgsz, p.device, p._prec = _BlobDetector(), str(REPO_ROOT / "configs" / "tracker.yaml"), 512, "cpu", {}
    p.class_of, p.tiles, p.tile_overlap, p._tracker, p.det_ms = {0: ObjClass.PERSON}, (2, 2), 0.2, None, 0.0
    ids = []
    for k in range(4):     # a person crossing the tile seams (x 569-711, y 320-400): cut boxes must not double it
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        img[300:420, 600 + 6 * k:660 + 6 * k] = 255
        dets = p.detect_track(img, 0.1)
        assert len(dets) == 1
        ids.append(dets[0].track_id)
    assert len(set(ids)) == 1
    p.reset_tracking()


def test_training_resumes_only_with_the_same_settings(tmp_path):
    pytest.importorskip("ultralytics")
    import json

    from training.train import AUG, SETTINGS_FILE, _same_run

    wanted = {"model": "yolo26s.pt", "imgsz": 960, "epochs": 50, **AUG}
    assert not _same_run(tmp_path, wanted)
    (tmp_path / SETTINGS_FILE).write_text(json.dumps(wanted))
    assert _same_run(tmp_path, dict(wanted))
    assert not _same_run(tmp_path, {**wanted, "model": "yolo26m.pt"})
    assert AUG["mosaic"] > 0 and AUG["close_mosaic"] > 0


def test_staying_silent_never_beats_catching_something():
    from snatchguard.evaluation.metrics import Metrics
    from snatchguard.evaluation.tune import objective

    silent = Metrics("any", events=25, events_detected=0, true_alerts=0, false_alarms=0, normal_hours=2.0,
                     normal_false_alarms=0)
    weak = Metrics("any", events=25, events_detected=6, true_alerts=7, false_alarms=22, normal_hours=2.0,
                   normal_false_alarms=21)
    good = Metrics("any", events=25, events_detected=23, true_alerts=23, false_alarms=1, normal_hours=2.0,
                   normal_false_alarms=1)
    assert objective(weak, 0.9) > objective(silent, 0.9)
    assert objective(good, 0.9) > objective(weak, 0.9)


def test_pipeline_rebuilds_lost_training_data_only_while_training_needs_it(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    from training import run_all

    monkeypatch.setattr(run_all, "STATE", tmp_path / "pipeline")
    monkeypatch.setattr(run_all, "DATASETS", tmp_path / "datasets")
    run_all.STATE.mkdir()
    prof = run_all.PROFILES["quick"]
    for s in ("datasets", "pseudo"):
        (run_all.STATE / f"{s}.done").write_text("done")
    # a fresh runtime: markers kept on Drive, training data gone, training not finished
    assert run_all.lost_outputs("datasets", prof) == ["train_det", "compare", "train_pose"]
    assert run_all.lost_outputs("pseudo", prof) == ["train_det", "compare", "train_pose"]
    assert run_all.lost_outputs("cache", prof) == []
    # the data is back
    train = run_all.DATASETS / "sg_det" / "images" / "train"
    train.mkdir(parents=True)
    (train / "pseudo_s_1_0001.jpg").write_bytes(b"")
    for d in ("sg_det", "sg_pose"):
        (run_all.DATASETS / d / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
        (run_all.DATASETS / d / "data.yaml").write_text("")
    assert run_all.lost_outputs("datasets", prof) == run_all.lost_outputs("pseudo", prof) == []
    # training finished: nothing needs the data any more
    for d in ("sg_det", "sg_pose"):
        (run_all.DATASETS / d / "data.yaml").unlink()
    for s in run_all.TRAINING:
        (run_all.STATE / f"{s}.done").write_text("done")
    assert run_all.lost_outputs("datasets", prof) == []


def test_step_log_keeps_the_final_state_of_progress_bars(tmp_path, monkeypatch, capsys):
    pytest.importorskip("yaml")
    from training import run_all

    monkeypatch.setattr(run_all, "STATE", tmp_path)
    run_all.run("demo", ["-c", "import sys; sys.stdout.write('epoch 1%\\repoch 50%\\repoch 100%\\nAP 0.61\\r\\ndone')"])
    assert "epoch 1%\repoch 50%\repoch 100%\n" in capsys.readouterr().out      # the console still redraws the bar
    log = (tmp_path / "logs" / "demo.log").read_text(encoding="utf-8")
    assert log.endswith("epoch 100%\nAP 0.61\ndone\n")
