"""GPU fine-tuning package: strip timing, training-pool labels, batches, thresholds, evaluation, checkpoint loading."""

from __future__ import annotations

import random
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from snatchguard.verifier.candidates import ClipCandidate
from snatchguard.verifier.finetune import (PAD, STRIP_NF, STRIP_SIDE, _batches, _neg_weights, _strips, _threshold,
                                           _units, _val_split, centre, evaluate, load_strip, pool_labels, strip_times,
                                           threshold_for)
from snatchguard.verifier.model import NF, SIDE, sample_times


def test_strip_centre_is_the_live_clip():
    t = strip_times(10.0)
    assert len(t) == STRIP_NF
    assert np.allclose(t[PAD:PAD + NF], sample_times(10.0))


def _counter_video(path, fps: float = 25.0, seconds: float = 6.0) -> None:
    """Every frame's brightness is its frame index (mod 250)."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 120))
    for i in range(int(fps * seconds)):
        w.write(np.full((120, 160, 3), i % 250, np.uint8))
    w.release()


def test_strips_take_the_right_frames(tmp_path):
    video = tmp_path / "v.avi"
    _counter_video(video, seconds=8.0)                               # frames 0-199
    _strips(str(video), [(3.0, (40.0, 20.0, 80.0)), (0.5, (0.0, 0.0, 400.0)), (7.0, (40.0, 20.0, 80.0))],
            tmp_path, ["a.jpg", "b.jpg", "c.jpg"])
    a = load_strip(str(tmp_path / "a.jpg"))
    assert a.shape == (STRIP_NF, STRIP_SIDE, STRIP_SIDE, 3)
    got = a[:, STRIP_SIDE // 2, STRIP_SIDE // 2].mean(axis=1)
    assert np.abs(got - np.round(strip_times(3.0) * 25.0)).max() <= 3      # codec noise only
    b = load_strip(str(tmp_path / "b.jpg"))                      # starts before t = 0: the first frame repeats
    assert b[0, STRIP_SIDE // 2, STRIP_SIDE // 2].mean() <= 3
    assert b[0, 2, 2].max() <= 3                                 # outside the frame: black
    c = load_strip(str(tmp_path / "c.jpg"))                      # runs past the end: its last frame repeats
    tail = c[:, STRIP_SIDE // 2, STRIP_SIDE // 2].mean(axis=1)
    last = int(np.searchsorted(np.round(strip_times(7.0) * 25.0), 199, side="right")) - 1   # frame 192
    assert np.abs(tail[last:] - tail[last]).max() <= 1 and np.abs(tail[last] - 192) <= 3
    assert centre(a).shape == (NF, SIDE, SIDE, 3)


def test_pool_labels_bag_the_event_window_and_ignore_its_surroundings():
    row = {"category": "snatch_bike", "event_start_s": "10.0", "event_end_s": "11.0"}
    clips = [ClipCandidate(t, (0, 0, 100), [(1, 2)]) for t in (2.0, 7.0, 9.0, 12.4, 14.0, 20.0)]
    ys, bag = pool_labels(row, clips)
    assert bag == [False, False, True, True, False, False]
    assert ys == [0, -1, -1, -1, -1, 0]
    assert pool_labels({"category": "normal"}, clips) == ([0] * 6, [False] * 6)


class _Pkg:
    """The parts of Package that sampling and evaluation read."""

    def __init__(self, videos: list[dict]) -> None:
        self.videos = videos
        rows = [(vi, c) for vi, v in enumerate(videos) for c in v["clips"]]
        self.vi = np.array([vi for vi, _ in rows])
        self.t = np.array([c["t"] for _, c in rows], np.float32)
        self.y = np.array([c["y"] for _, c in rows])
        self.bag = np.array([c["bag"] for _, c in rows])
        self.normal = np.array([videos[vi]["category"] == "normal" for vi in self.vi])
        self.set = np.array([videos[vi].get("set", "S1") for vi in self.vi])
        self.source = np.where(self.set == "S1", "S1", "web")
        self.group = np.array([videos[vi].get("group", videos[vi]["key"]) for vi in self.vi])

    def hours(self, mask_videos) -> float:
        return sum(v["duration"] for vi, v in enumerate(self.videos) if mask_videos[vi] and v["category"] == "normal") / 3600


def _video(key, category, clips, status="ok", event=None, duration=60.0, set_="S1", group=None):
    return {"key": key, "category": category, "status": status, "event": event, "duration": duration,
            "set": set_, "group": group or key, "clips": [{"t": t, "y": y, "bag": b} for t, y, b in clips]}


def test_units_and_batches():
    pkg = _Pkg([_video("A", "snatch_bike", [(5.0, 1, False), (5.2, 1, False), (30.0, 0, False)], event=[5.0, 6.0]),
                _video("B", "snatch_bike", [(9.0, -1, True), (9.5, -1, True), (10.0, -1, True), (40.0, 0, False)],
                       status="pool", event=[9.0, 10.0]),
                _video("N", "normal", [(t, 0, False) for t in range(0, 60, 5)], status="")])
    pos, neg = _units(pkg, np.arange(len(pkg.y)))
    assert sorted((is_bag, len(c)) for is_bag, c in pos) == [(False, 2), (True, 3)]
    assert len(neg) == 2 + 12
    bag = next(c for is_bag, c in pos if is_bag)
    for items in _batches(pos, neg, np.ones(len(neg)), batch=16, steps=20, seed=1):
        assert len(items) == 16                                       # fixed batch size
        units = {}
        for i, u, y in items:
            units.setdefault(u, []).append((i, y))
        assert all(len({y for _, y in v}) == 1 for v in units.values())   # one label per unit
        positives = [v for v in units.values() if v[0][1] == 1]
        assert 1 <= sum(len(v) for v in positives) <= 8               # at most half the batch
        for v in positives:                                           # a bag comes whole
            assert len(v) == 1 or sorted(i for i, _ in v) == sorted(bag)


def test_negatives_of_each_source_weigh_its_share_of_the_positives():
    pkg = _Pkg([_video("A", "snatch_bike", [(5.0, 1, False), (30.0, 0, False)], event=[5.0, 6.0]),
                _video("N1", "normal", [(t, 0, False) for t in range(0, 60, 2)], status=""),
                _video("B", "snatch_bike", [(9.0, 1, False), (40.0, 0, False)], event=[9.0, 10.0], set_="POOL"),
                _video("C", "snatch_bike", [(9.0, 1, False)], event=[9.0, 10.0], set_="POOL"),
                _video("D", "snatch_bike", [(9.0, 1, False)], event=[9.0, 10.0], set_="POOL"),
                _video("N2", "normal", [(t, 0, False) for t in range(0, 60, 20)], status="", set_="POOL")])
    pos, neg = _units(pkg, np.arange(len(pkg.y)))
    hard = {i for i in neg if not pkg.normal[i]}
    w = _neg_weights(pkg, [str(pkg.source[u[0]]) for _, u in pos], neg, hard)
    src = pkg.source[np.array(neg)]
    assert w[src == "S1"].sum() / w.sum() == pytest.approx(1 / 4)           # 1 of the 4 events is Snatch 1.0
    assert w[src == "web"].sum() / w.sum() == pytest.approx(3 / 4)
    s1 = [k for k, i in enumerate(neg) if src[k] == "S1"]
    assert w[s1[0]] == pytest.approx(3 * w[s1[1]])                          # same-scene negative 3x a normal one
    only_s1 = _neg_weights(pkg, ["S1"], neg, hard)
    assert only_s1[src == "web"].sum() == 0                                  # no web positives: no web negatives


def test_validation_holds_out_footage_of_each_source():
    vids = [_video(f"S{k}", "normal", [(t, 0, False) for t in range(0, 600, 10)], status="", duration=600.0,
                   group=f"cam{k}") for k in range(5)]
    vids += [_video(f"P{k}", "normal", [(t, 0, False) for t in range(0, 60, 10)], status="", set_="POOL")
             for k in range(10)]
    pkg = _Pkg(vids)
    trn, val = _val_split(pkg, np.arange(len(pkg.y)), 0.2, seed=1)
    assert {"S1", "web"} == set(pkg.source[val].tolist()) and len(set(pkg.group[val]) & set(pkg.group[trn])) == 0
    s = np.where(pkg.source[val] == "web", 0.9, 0.1).astype(np.float32)
    s[0] = 0.95
    thr, per = _threshold(pkg, val, s, budget=10.0)
    assert per["web"]["alerts"] / per["web"]["normal_hours"] <= 10 and per["S1"]["alerts"] <= 1
    assert thr == max(d["threshold"] for d in per.values())


def test_threshold_allows_the_budget_on_validation_footage():
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.05], np.float32)
    thr = threshold_for(s, hours=0.2, budget=10.0)                # 2 alerts allowed
    assert (s >= thr).sum() == 2
    assert (s >= threshold_for(s, hours=0.05, budget=10.0)).sum() == 0


def test_evaluate_counts_events_and_false_alarms():
    pkg = _Pkg([_video("E1", "snatch_bike", [(10.0, 1, False), (30.0, 0, False)], event=[10.0, 11.0]),
                _video("E2", "snatch_foot", [(10.5, 0, False), (50.0, 0, False)], status="untracked", event=[10.0, 11.0]),
                _video("N", "normal", [(t, 0, False) for t in range(0, 3600, 360)], status="", duration=3600.0)])
    idx = np.arange(len(pkg.y))
    s = np.full(len(idx), 0.1, np.float32)
    s[0] = 0.95          # E1's actor clip
    s[2] = 0.9           # E2: a clip inside the event window, not the actors' contact
    s[4] = 0.92          # one normal clip
    res = evaluate(pkg, idx, s, thr=0.85, budgets=[1, 10])
    p = res["at_threshold"]
    assert (res["events_actor"], res["events_timed"]) == (2, 2)
    assert (p["caught_actor"], p["caught_overlap"], p["false_alarms"]) == (1, 2, 1)
    assert p["false_alarms_per_h"] == pytest.approx(1.0)
    assert res["best_at_budget"]["1"][1:] == [1, 2]
    fas = [c[0] for c in res["curve"]]
    assert fas == sorted(fas)


torch = pytest.importorskip("torch")


def test_finetuned_checkpoint_scores_like_its_network(tmp_path):
    from snatchguard.verifier.finetune import _save, build_net
    from snatchguard.verifier.model import _MEAN, _STD, Verifier
    torch.manual_seed(0)
    net = build_net(pretrained=False).eval()
    _save(net, tmp_path / "ft.pt", threshold=0.7, meta={"protocol": "test"})
    v = Verifier(tmp_path / "ft.pt", threads=1)
    assert v.threshold == pytest.approx(0.7) and v.head is None
    clips = np.random.default_rng(0).integers(0, 255, size=(2, NF, SIDE, SIDE, 3), dtype=np.uint8)
    x = torch.from_numpy(((clips / 255.0 - _MEAN) / _STD).astype(np.float32).transpose(0, 4, 1, 2, 3))
    with torch.inference_mode():
        want = torch.sigmoid(net(x)).squeeze(1).numpy()
    assert np.allclose(v.score(clips), want, atol=1e-4)


def test_augment_keeps_the_clip_shape():
    from snatchguard.verifier.finetune import augment
    frames = np.random.default_rng(0).integers(0, 255, size=(STRIP_NF, STRIP_SIDE, STRIP_SIDE, 3), dtype=np.uint8)
    for k in range(20):
        out = augment(frames, random.Random(k))
        assert out.shape == (NF, SIDE, SIDE, 3) and out.dtype == np.uint8
    assert SimpleNamespace  # noqa: B018 - keep the import used if tests change
