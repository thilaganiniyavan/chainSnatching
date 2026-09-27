"""Edge building blocks: lighting, night simulation, tamper, outbox, evidence."""

from __future__ import annotations

import numpy as np

from snatchguard.core.types import Detection, Lighting, ObjClass
from snatchguard.edge.evidence import EvidenceRecorder
from snatchguard.edge.lighting import Enhancer, LightingAnalyzer
from snatchguard.edge.outbox import Outbox
from snatchguard.edge.tamper import TamperDetector
from snatchguard.sim.night import to_ir, to_lowlight


def _street(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((360, 640, 3), (120, 150, 170), np.uint8)
    img[200:] = (90, 90, 95)
    for _ in range(40):                       # buildings / signs / texture
        x, y = rng.integers(0, 600), rng.integers(0, 200)
        img[y:y + 30, x:x + 30] = rng.integers(0, 255, 3)
    return img


def test_lighting_classifies_day_ir_and_lowlight():
    la = LightingAnalyzer()
    day = _street()
    assert la.classify(day) is Lighting.DAY
    assert la.classify(to_ir(day, np.random.default_rng(0))) is Lighting.IR
    assert la.classify(to_lowlight(day, np.random.default_rng(0))) is Lighting.LOWLIGHT


def test_lighting_mode_needs_confirmation():
    la = LightingAnalyzer(every_s=1.0, confirm=3)
    ir = to_ir(_street(), np.random.default_rng(0))
    modes = [la.update(ir, t) for t in range(5)]
    assert modes[0] is Lighting.DAY and modes[1] is Lighting.DAY   # no flicker on one frame
    assert modes[-1] is Lighting.IR


def test_enhancer_brightens_lowlight():
    dark = to_lowlight(_street(), np.random.default_rng(1))
    out = Enhancer().apply(dark, Lighting.LOWLIGHT)
    assert out.mean() > dark.mean() * 1.3
    assert Enhancer().apply(dark, Lighting.DAY) is dark


def test_tamper_blackout_and_recovery():
    td = TamperDetector(persist_s=3, learn_s=5)
    img = _street()
    raised_all, cleared_all = set(), set()
    for t in range(10):
        r, c = td.update(img + np.uint8(t % 2), float(t))       # tiny changes so it is not "frozen"
        raised_all |= r
    assert not raised_all
    black = np.zeros_like(img)
    for t in range(10, 16):
        r, _ = td.update(black + np.uint8(t % 2), float(t))
        raised_all |= r
    assert "blackout" in raised_all
    for t in range(16, 20):
        _, c = td.update(img + np.uint8(t % 2), float(t))
        cleared_all |= c
    assert "blackout" in cleared_all


def test_tamper_camera_moved():
    td = TamperDetector(persist_s=3, learn_s=5)
    for t in range(8):
        td.update(_street(0) + np.uint8(t % 2), float(t))
    other = np.ascontiguousarray(np.fliplr(_street(7)))
    raised = set()
    for t in range(8, 14):
        r, _ = td.update(other + np.uint8(t % 2), float(t))
        raised |= r
    assert "moved" in raised


def test_tamper_frozen_feed():
    td = TamperDetector(frozen_s=3)
    img = _street()
    raised = set()
    for t in range(6):
        r, _ = td.update(img, float(t))
        raised |= r
    assert "frozen" in raised


def test_tamper_does_not_fire_on_day_to_night_switch():
    td = TamperDetector(persist_s=3, learn_s=5)
    day = _street()
    for t in range(8):
        td.update(day + np.uint8(t % 2), float(t), "day")
    ir = to_ir(day, np.random.default_rng(0))
    raised = set()
    for t in range(8, 20):
        r, _ = td.update(ir + np.uint8(t % 2), float(t), "ir")
        raised |= r
    assert "moved" not in raised


def test_outbox_retry_and_persistence(tmp_path):
    db = tmp_path / "o.db"
    ob = Outbox(db)
    i = ob.put("event", {"event_id": "a"}, {"snapshot": "x.jpg"})
    items = ob.due()
    assert [x.id for x in items] == [i] and items[0].payload["event_id"] == "a"
    ob.retry_later(i, 0, "down")
    assert ob.due() == []                    # backed off
    ob.close()
    ob2 = Outbox(db)                          # survives a restart
    assert ob2.pending() == 1
    ob2.done(i)
    assert ob2.pending() == 0


def test_evidence_clip_and_snapshot(tmp_path):
    rec = EvidenceRecorder(tmp_path, pre_s=1.0, post_s=0.5, width=320)
    det = [Detection(1, ObjClass.PERSON, np.array([100, 100, 150, 250.0]), 0.9)]
    for i in range(30):
        img = _street(i % 3)
        rec.push(img, i * 0.1, det)
        if i == 15:
            snap = rec.snapshot("ev1", 1.5, {1: "victim"}, "TEST")
            rec.schedule_clip("ev1", 1.2, 1.6, {1: "victim"}, "TEST")
    rec.close()
    done = dict(rec.finished())
    assert snap.exists()
    assert done["ev1"] is not None and done["ev1"].exists() and done["ev1"].stat().st_size > 1000
