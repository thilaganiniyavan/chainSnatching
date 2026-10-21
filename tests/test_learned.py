"""Learned scorer: window bookkeeping, live == offline, model file, alert merging, labels, OOF files.
No video files or model downloads: a fake backbone stands in for the CNN."""

from __future__ import annotations

import json

import numpy as np
import pytest

from snatchguard.evaluation.cache import cache_path
from snatchguard.learned import extract as ex
from snatchguard.learned.features import (FeatureSpec, FrameFeaturizer, Sampler, views, window_dim, window_feature,
                                          window_spans, windows_for)
from snatchguard.learned.scorer import Head, SnatchScorer, WindowScore, alerts_from_windows

pytest.importorskip("torch")

SPEC = FeatureSpec(input_px=32, max_side=96, sample_fps=4.0)
K = 4


class FakeBackbone:
    dim = 12

    def embed(self, crops, px):
        return np.stack([np.concatenate([c.reshape(-1, 3).mean(0) / 255.0] * 4) for c in crops]).astype(np.float32)


def _head(tmp_path=None, threshold=0.5, min_run=1) -> Head:
    rng = np.random.default_rng(0)
    f = window_dim(K)
    return Head(SPEC, 2.0, 0.5, rng.normal(size=FakeBackbone.dim).astype(np.float32),
                rng.normal(size=(K, FakeBackbone.dim)).astype(np.float32), np.zeros(f, np.float32),
                np.ones(f, np.float32), rng.normal(size=f).astype(np.float32) * 0.1, -0.2, 1.0, 0.0,
                threshold, min_run, {"note": "test"})


def _frames(n: int, fps: float = 25.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        img = np.full((72, 128, 3), 40, np.uint8)
        x = (i * 3) % 100
        img[20:50, x:x + 20] = rng.integers(100, 255, size=3, dtype=np.uint8)
        yield i / fps, img


def test_views_whole_frame_plus_tiles():
    img = np.zeros((100, 200, 3), np.uint8)
    v = views(img, SPEC)
    assert len(v) == SPEC.n_views == 5
    assert v[0].shape == (100, 200, 3)
    assert all(t.shape == (60, 120, 3) for t in v[1:])


def test_sampler_picks_the_requested_rate():
    s = Sampler(4.0)
    taken = [t for t in np.arange(0, 10, 1 / 25) if s.take(float(t))]
    assert 39 <= len(taken) <= 41
    assert np.allclose(np.diff(taken), 0.24, atol=0.05)


def test_window_spans_grid_and_short_clip():
    t = np.arange(0, 5.0, 0.25)
    spans = window_spans(t, 2.0, 0.5)
    assert spans[0] == (0.0, 2.0) and spans[1] == (0.5, 2.5)
    assert spans[-1][1] <= t[-1] + 0.125 + 1e-6
    short = np.arange(0, 1.0, 0.25)
    assert window_spans(short, 2.0, 0.5) == [(0.0, 0.875)]
    assert window_spans(short, 2.0, 0.5, final=False) == []
    assert window_spans(np.array([0.0]), 2.0, 0.5) == []


def test_window_feature_dimension():
    z = np.random.default_rng(0).normal(size=(8, 5, K)).astype(np.float32)
    m = np.random.default_rng(1).random((8, 5, 3)).astype(np.float32)
    assert window_feature(z, m).shape == (window_dim(K),)
    assert window_feature(z[:1], m[:1]).shape == (window_dim(K),)


def test_live_push_matches_offline_windows(tmp_path):
    head = _head()
    path = tmp_path / "m.pt"
    head.save(path)
    scorer = SnatchScorer(path, backbone=FakeBackbone())
    live: list[WindowScore] = []
    for t, img in _frames(130):          # 5.2 s at 25 fps
        live += scorer.push(img, t)
    live += scorer.flush()

    feat = FrameFeaturizer(SPEC, FakeBackbone())
    sampler = Sampler(SPEC.sample_fps)
    ts, zs, ms = [], [], []
    for t, img in _frames(130):
        if sampler.take(t):
            e, m = feat(img)
            ts.append(t)
            zs.append(head.project(e))
            ms.append(m)
    spans, X = windows_for(np.array(ts), np.stack(zs), np.stack(ms), 2.0, 0.5)
    assert [(w.t_start, w.t_end) for w in live] == spans
    assert np.allclose([w.score for w in live], head.proba(X), atol=1e-5)
    assert all(0.0 <= w.score <= 1.0 for w in live)


def test_short_clip_gets_one_window_on_flush_and_reset_restarts(tmp_path):
    path = tmp_path / "m.pt"
    _head().save(path)
    scorer = SnatchScorer(path, backbone=FakeBackbone())
    out = []
    for t, img in _frames(30):           # 1.2 s: shorter than one window
        out += scorer.push(img, t)
    assert out == []
    out = scorer.flush()
    assert len(out) == 1 and out[0].t_start == 0.0
    assert scorer.flush() == []
    scorer.reset()
    again = []
    for t, img in _frames(80):
        again += scorer.push(img, t + 100.0)
    assert again and again[0].t_start == 100.0


def test_model_file_roundtrip(tmp_path):
    head = _head(threshold=0.123, min_run=2)
    path = tmp_path / "sub" / "m.pt"
    head.save(path)
    back = Head.load(path)
    assert back.spec == SPEC and back.threshold == pytest.approx(0.123) and back.min_run == 2
    X = np.random.default_rng(3).normal(size=(5, window_dim(K))).astype(np.float32)
    assert np.allclose(back.proba(X), head.proba(X), atol=1e-6)
    assert back.meta == {"note": "test"}


def test_alerts_merge_consecutive_windows():
    wins = [(0, 2, 0.1), (0.5, 2.5, 0.8), (1, 3, 0.9), (1.5, 3.5, 0.2), (2, 4, 0.7), (2.5, 4.5, 0.1)]
    a = alerts_from_windows(wins, 0.5)
    assert [(x["contact_start"], x["contact_end"], x["decided_at"]) for x in a] == [(0.5, 3, 2.5), (2, 4, 4)]
    assert all(x["level"] == "SUSPECT" and x["source"] == "model" for x in a)
    assert a[0]["score"] == 0.9
    a2 = alerts_from_windows([WindowScore(*w) for w in wins], 0.5, min_run=2)
    assert len(a2) == 1 and a2[0]["decided_at"] == 3


def test_window_labels_positive_negative_ignored():
    from snatchguard.learned.train import window_labels

    spans = [(s, s + 2.0) for s in np.arange(0, 14, 0.5)]
    row = {"category": "snatch_bike", "event_start_s": "8.0", "event_end_s": "9.0"}
    y = window_labels(spans, row)
    by_start = dict(zip([s for s, _ in spans], y))
    assert by_start[7.5] == 1 and by_start[8.0] == 1
    assert by_start[0.0] == 0 and by_start[3.0] == 0 and by_start[12.0] == 0
    assert by_start[5.0] == -1 and by_start[10.0] == -1
    assert set(window_labels(spans, {"category": "snatch_foot", "event_start_s": ""})) == {-1}
    assert set(window_labels(spans, {"category": "normal"})) == {0}


def test_stem_matches_perception_cache_naming():
    for v in ("Snatch Theft/17_0.mp4", "Normal/s_2.avi", "snatch/0eOz5O3FQM8_0.mp4"):
        assert ex.video_stem(v) == cache_path(v, "tag").name.removesuffix(".pkl.gz")


def test_oof_json_format(tmp_path):
    from snatchguard.learned.train import Video, write_oof

    v = Video({"video": "Snatch Theft/17_0.mp4", "category": "snatch_bike", "fold": "A", "lighting": "day"},
              tmp_path / "x.npz", 2.2, spans=[(0.0, 2.0), (0.5, 2.5)])
    write_oof("snatch_ab", [v], [np.array([0.1, 0.7])], [_head()], tmp_path)
    d = json.loads((tmp_path / "snatch_ab" / "Snatch_Theft__17_0.json").read_text())
    assert d["video"] == "Snatch Theft/17_0.mp4" and d["protocol"] == "snatch_ab"
    assert d["windows"] == [[0.0, 2.0, 0.1], [0.5, 2.5, 0.7]]
    assert d["duration_s"] == 2.2 and "threshold" in d and "min_run" in d


def test_fit_head_on_separable_synthetic_windows():
    pytest.importorskip("sklearn")
    from snatchguard.learned.train import Video, fit_head

    rng = np.random.default_rng(0)
    f = window_dim(K)
    videos = []
    for i in range(6):
        spans = [(s, s + 2.0) for s in np.arange(0, 12, 0.5)]
        X = rng.normal(size=(len(spans), f)).astype(np.float32)
        row = {"video": f"Snatch Theft/{i}.mp4", "category": "snatch_bike", "event_start_s": "8.0",
               "event_end_s": "9.0", "fold": "A"}
        v = Video(row, None, 12.0, spans=spans, X=X)
        from snatchguard.learned.train import window_labels
        v.y = window_labels(spans, row)
        v.X[v.y == 1, :5] += 4.0
        videos.append(v)
    for i in range(4):
        spans = [(s, s + 2.0) for s in np.arange(0, 200, 0.5)]
        v = Video({"video": f"Normal/n{i}.avi", "category": "normal", "fold": "A"}, None, 200.0, spans=spans,
                  X=rng.normal(size=(len(spans), f)).astype(np.float32))
        v.y = np.zeros(len(spans), np.int8)
        videos.append(v)
    pca = (np.zeros(FakeBackbone.dim, np.float32), np.eye(K, FakeBackbone.dim, dtype=np.float32))
    head, info = fit_head(videos, SPEC, pca, 0, "synthetic")
    assert info["inner"]["recall"] >= 0.8 and info["inner"]["precision"] >= 0.8
    p = head.proba(np.concatenate([v.X for v in videos]))
    assert ((p >= 0) & (p <= 1)).all()


def _fake_features(path, duration, event=None, seed=0):
    """Feature file like extract.extract_video writes, with a strong change during `event`."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 0.25)
    emb = rng.normal(size=(len(t), 5, 16)).astype(np.float32) * 0.1
    motion = rng.random((len(t), 5, 3)).astype(np.float32) * 0.05
    if event:
        m = (t >= event[0]) & (t <= event[1])
        emb[m] += rng.normal(size=(m.sum(), 5, 16)) * 3.0
        motion[m] += 0.8
    ex.save_features({"t": t, "emb": emb.astype(np.float16), "motion": motion, "duration_s": np.float64(duration),
                      "src_fps": np.float64(25), "max_seconds": np.float64(-1),
                      "build_samples_per_s": np.float64(5.0), "spec_tag": np.array("x")}, path)


def test_train_run_end_to_end_on_synthetic_features(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    from snatchguard.learned import train as tr

    monkeypatch.setattr(ex, "FEATURES_DIR", tmp_path / "features")
    monkeypatch.setattr(tr, "K_PCA", 8)
    spec = FeatureSpec()
    rows = ["video,category,lighting,event_start_s,event_end_s,fold,notes"]
    for i in range(8):
        fold = "AB"[i % 2]
        name = f"Snatch Theft/c{i}.mp4"
        rows.append(f"{name},snatch_{'bike' if i < 4 else 'foot'},day,6.0,7.0,{fold},")
        _fake_features(ex.feature_path(name, spec), 12.0, (6.0, 7.0), seed=i)
    for i in range(6):
        fold = "AB"[i % 2]
        name = f"Normal/n{i}.avi"
        rows.append(f"{name},normal,day,,,{fold},")
        _fake_features(ex.feature_path(name, spec), 300.0, seed=100 + i)
    labels = tmp_path / "labels.csv"
    labels.write_text("\n".join(rows) + "\n")
    web_rows = ["video,category,lighting,event_start_s,event_end_s,fold,notes",
                "snatch/w0.mp4,snatch_bike,night,5.0,6.0,A,", "normal/w1.mp4,normal,day,,,B,"]
    _fake_features(ex.feature_path("snatch/w0.mp4", spec), 15.0, (5.0, 6.0), seed=50)
    _fake_features(ex.feature_path("normal/w1.mp4", spec), 120.0, seed=51)
    web = tmp_path / "web.csv"
    web.write_text("\n".join(web_rows) + "\n")

    out = tr.run(labels, tmp_path, web, tmp_path, "cpu", 1, None, False, tmp_path / "m.pt", tmp_path / "oof",
                 tmp_path / "report.md")
    assert out["snatch_ab"]["recall"] >= 0.75 and out["snatch_ab"]["precision"] >= 0.75
    assert out["web_external"]["events"] == 1
    assert len(list((tmp_path / "oof" / "snatch_ab").glob("*.json"))) == 14
    assert len(list((tmp_path / "oof" / "web_external").glob("*.json"))) == 2
    head = Head.load(tmp_path / "m.pt")
    assert head.meta["trained_on"] == "snatch1.0 folds A+B" and 0 < head.threshold < 1
    text = (tmp_path / "report.md").read_text()
    assert "Snatch-Web" in text and "Every clip" in text
