"""Two-stage verifier: contact candidates, clip cutting, the head's file format, the live stage, training labels."""

from __future__ import annotations

import time

import numpy as np
import pytest

from snatchguard.core.types import Detection, ObjClass
from snatchguard.verifier.candidates import (CandidateStage, ClipCandidate, ContactDetector, MIN_SEP_S, crop_for,
                                             square_iou)
from snatchguard.verifier.model import AFTER_S, NF, SIDE, cut, sample_times
from snatchguard.verifier.train import label_clips


def person(tid: int, x: float, y: float = 100.0, h: float = 100.0) -> Detection:
    return Detection(tid, ObjClass.PERSON, np.array([x, y, x + 0.36 * h, y + h]), 0.9)


def bike(tid: int, x: float, y: float = 140.0) -> Detection:
    return Detection(tid, ObjClass.MOTORCYCLE, np.array([x, y, x + 80.0, y + 60.0]), 0.9)


def test_contact_starts_once_and_again_after_separating():
    det = ContactDetector()
    starts = []
    t = 0.0
    # approach, stay in contact, separate for longer than MIN_SEP_S, come back
    plan = [(200.0, 10)] + [(130.0, 10)] + [(300.0, int(MIN_SEP_S * 8) + 4)] + [(130.0, 5)]
    for x, n in plan:
        for _ in range(n):
            starts += det.step(t, [person(1, 100.0), person(2, x)])
            t += 0.125
    assert len(starts) == 2
    assert {(c.a, c.b) for c in starts} == {(1, 2)}


def test_contact_starts_again_after_one_of_the_pair_was_lost():
    # Snatch 1.0 7_0: in contact, attacker undetected for 2 s, back in contact for the grab
    det = ContactDetector()
    starts = []
    t = 0.0
    for dets, n in [([person(1, 100.0), person(2, 130.0)], 4), ([person(1, 100.0)], 16),
                    ([person(1, 100.0), person(2, 130.0)], 4)]:
        for _ in range(n):
            starts += det.step(t, dets)
            t += 0.125
    assert [c.t for c in starts] == [0.0, 2.5]


def test_jitter_around_the_threshold_does_not_start_new_contacts():
    det = ContactDetector()
    starts = []
    for k in range(40):     # gap alternates just inside / just outside CLOSE but never past OPEN
        x = 136.0 + (20.0 if k % 2 else 0.0)
        starts += det.step(k * 0.125, [person(1, 100.0), person(2, x)])
    assert len(starts) == 1


def test_rider_on_own_bike_is_not_a_pair():
    det = ContactDetector()
    out = []
    for k in range(10):
        out += det.step(k * 0.125, [Detection(5, ObjClass.PERSON, np.array([105.0, 110.0, 155.0, 190.0]), 0.9),
                                    bike(6, 100.0)])           # rider box IoU 0.4 with the bike
    assert out == []


def test_simultaneous_contacts_in_one_place_become_one_clip():
    stage = CandidateStage()
    dets = [person(1, 100.0), person(2, 130.0), person(3, 160.0)]
    clips = stage.step(0.0, dets)
    assert len(clips) == 1
    assert len(clips[0].pairs) >= 2


def test_far_apart_contacts_stay_separate_clips():
    stage = CandidateStage()
    dets = [person(1, 100.0), person(2, 130.0), person(3, 900.0), person(4, 930.0)]
    assert len(stage.step(0.0, dets)) == 2


def test_crop_is_square_and_padded():
    x0, y0, side = crop_for([np.array([100.0, 100.0, 140.0, 200.0]), np.array([150.0, 110.0, 190.0, 210.0])])
    assert side == pytest.approx(220.0)                  # 2 x the larger side (110)
    assert square_iou((0, 0, 10), (0, 0, 10)) == pytest.approx(1.0)
    assert square_iou((0, 0, 10), (20, 20, 10)) == 0.0


def test_cut_pads_outside_the_frame_and_flips():
    img = np.zeros((100, 200, 3), np.uint8)
    img[:, :100] = (0, 0, 255)                           # left half red (BGR)
    out = cut(img, (-50.0, 0.0, 100.0))                 # half of the square lies left of the frame
    assert out.shape == (SIDE, SIDE, 3)
    assert out[:, : SIDE // 2 - 2].max() == 0            # padding is black
    assert out[SIDE // 2, SIDE - 3, 0] > 200             # inside: red, now RGB
    flipped = cut(img, (-50.0, 0.0, 100.0), flip=True)
    assert flipped[SIDE // 2, 2, 0] > 200


def test_clip_window_spans_before_and_after_the_onset():
    ts = sample_times(10.0)
    assert len(ts) == NF and ts[0] == pytest.approx(8.5) and ts[-1] == pytest.approx(12.5)


def test_training_labels_are_actor_grounded():
    row = {"category": "snatch_bike", "event_start_s": "5.0", "event_end_s": "6.0"}
    actors = {"status": "ok", "victim": [1], "attacker": [2]}
    clips = [ClipCandidate(5.2, (0, 0, 100), [(1, 2)]),        # the labelled pair in the window
             ClipCandidate(5.2, (0, 0, 100), [(3, 4)]),        # bystanders at the same time: a negative
             ClipCandidate(5.5, (0, 0, 100), [(1, 3)]),        # the victim with someone else, near the event
             ClipCandidate(20.0, (0, 0, 100), [(3, 4)])]       # bystanders, long after
    assert label_clips(row, clips, actors) == [1, 0, -1, 0]
    assert label_clips(row, clips, {"status": "unclear", "victim": [], "attacker": []}) == [-1, -1, -1, 0]
    assert label_clips({"category": "normal"}, clips, None) == [0, 0, 0, 0]


torch = pytest.importorskip("torch")


def test_head_roundtrip(tmp_path):
    pytest.importorskip("sklearn")
    from snatchguard.verifier.model import Head
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 512)).astype(np.float32)
    y = np.r_[np.ones(10), np.zeros(50)]
    x[:10] += 1.5
    head = Head.fit(x, y, n_components=8)
    head.threshold = 0.42
    head.save(tmp_path / "v.pt")
    back = Head.load(tmp_path / "v.pt")
    assert back.threshold == pytest.approx(0.42)
    assert np.allclose(back.proba(x), head.proba(x), atol=1e-5)
    assert head.proba(x[:10]).mean() > head.proba(x[10:]).mean()


class _StubVerifier:
    threshold = 0.5

    def __init__(self, score: float) -> None:
        self.value = score
        self.seen: list[tuple] = []

    def score(self, clips: np.ndarray) -> np.ndarray:
        self.seen.append(clips.shape)
        return np.full(len(clips), self.value, np.float32)


def _run_live(score: float):
    from snatchguard.verifier.live import VerifierStage
    stub = _StubVerifier(score)
    stage = VerifierStage(stub, "test")
    img = np.full((240, 320, 3), 128, np.uint8)
    got, t = [], 0.0
    for k in range(int((AFTER_S + 1.5) * 8)):
        x = 220.0 - 12.0 * min(k, 8)                # person 2 walks up to person 1, then stays
        got += stage.step(t, img, [person(1, 100.0), person(2, x)])
        t += 0.125
        time.sleep(0.002)
    deadline = time.time() + 5.0
    while not got and time.time() < deadline and score >= 0.5:
        time.sleep(0.05)
        got += stage.step(t, img, [person(1, 100.0), person(2, 130.0)])
    stage.stop()
    return got, stub


def test_live_stage_verifies_a_contact_after_its_window():
    got, stub = _run_live(0.9)
    assert len(got) == 1 and got[0].score == pytest.approx(0.9)
    assert stub.seen and stub.seen[0][1:] == (NF, SIDE, SIDE, 3)


def test_live_stage_stays_quiet_below_threshold():
    got, stub = _run_live(0.1)
    assert got == []


def _contact_frames(stage, n: int, t: float = 0.0) -> float:
    img = np.full((240, 320, 3), 128, np.uint8)
    for k in range(n):
        stage.step(t, img, [person(1, 100.0), person(2, 220.0 - 12.0 * min(k, 8))])
        t += 0.125
    return t


def test_scene_change_verifies_the_clip_it_cut_short():
    from snatchguard.verifier.live import VerifierStage
    stub = _StubVerifier(0.9)
    stage = VerifierStage(stub, "test")
    _contact_frames(stage, 12)                  # contact starts, the scene ends before AFTER_S has passed
    stage.reset()
    got = stage.finish(timeout=5.0)
    stage.stop()
    assert len(got) == 1 and stub.seen[0][1:] == (NF, SIDE, SIDE, 3)


def test_end_of_stream_returns_the_alerts_still_owed():
    from snatchguard.verifier.live import VerifierStage
    stage = VerifierStage(_StubVerifier(0.9), "test")
    _contact_frames(stage, 12)
    got = stage.finish(timeout=5.0)
    stage.stop()
    assert len(got) == 1


class _SlowVerifier(_StubVerifier):
    def score(self, clips: np.ndarray) -> np.ndarray:
        time.sleep(0.3)
        return super().score(clips)


def test_clips_that_wait_too_long_are_dropped_not_alerted_late():
    from snatchguard.verifier.live import _Runner
    runner = _Runner(_SlowVerifier(0.9), "test", max_lag_s=0.1)
    runner.start()
    frames = np.zeros((NF, SIDE, SIDE, 3), np.uint8)
    clip = ClipCandidate(0.0, (0, 0, 100), [(1, 2)])
    runner.offer(clip, frames, 0.0)             # taken at once
    time.sleep(0.05)
    for _ in range(6):                          # arrive while the network is busy, wait > max_lag_s
        runner.offer(clip, frames, 0.0)
    runner.wait_idle(5.0)
    runner.stop()
    assert runner.scored == 1 and runner.dropped == 6


def test_verifier_process_scores_like_the_in_process_one():
    from pathlib import Path
    model = Path(__file__).resolve().parents[1] / "models" / "verifier" / "verifier.pt"
    if not model.exists():
        pytest.skip("no trained verifier (python -m snatchguard.verifier.train)")
    from snatchguard.verifier.model import Verifier
    from snatchguard.verifier.serve import ProcessVerifier
    clips = np.random.default_rng(1).integers(0, 255, size=(2, NF, SIDE, SIDE, 3), dtype=np.uint8)
    pv = ProcessVerifier(model, threads=1, nice=0)
    try:
        got = pv.score(clips)
    finally:
        pv.close()
    ref = Verifier(model, threads=1)
    assert pv.threshold == pytest.approx(ref.threshold)
    assert np.allclose(got, ref.score(clips), atol=1e-4)


def test_multi_view_live_stage_cuts_one_clip_per_shift_and_waits_for_the_last():
    from snatchguard.verifier.live import VerifierStage
    stub = _StubVerifier(0.9)
    stub.shifts = (-0.5, 0.0, 0.5)
    stage = VerifierStage(stub, "test")
    img = np.full((240, 320, 3), 128, np.uint8)
    t, onset, offered = 0.0, None, None
    for k in range(int((AFTER_S + 2.0) * 8)):
        x = 220.0 - 12.0 * min(k, 8)
        stage.step(t, img, [person(1, 100.0), person(2, x)])
        if onset is None and stage.candidates:
            onset = t
        if offered is None and not stage._pending and onset is not None:
            offered = t
        t += 0.125
    deadline = time.time() + 5.0
    while not stub.seen and time.time() < deadline:
        time.sleep(0.02)
    stage.stop()
    assert stub.seen and stub.seen[0] == (1, 3, NF, SIDE, SIDE, 3)
    assert offered - onset >= AFTER_S + 0.5 and offered - onset < AFTER_S + 0.5 + 0.2


def test_live_stage_takes_the_confirm_threshold_from_the_model_unless_configured():
    from snatchguard.verifier.live import VerifierStage
    stub = _StubVerifier(0.9)
    stub.threshold_confirm = 0.95
    a, b = VerifierStage(stub, "a"), VerifierStage(stub, "b", threshold_confirm=0.97)
    try:
        assert a.threshold_confirm == pytest.approx(0.95) and b.threshold_confirm == pytest.approx(0.97)
        assert VerifierStage(_StubVerifier(0.9), "c").threshold_confirm is None
    finally:
        for st in (a, b):
            st.stop()
