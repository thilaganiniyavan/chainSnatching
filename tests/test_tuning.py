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
