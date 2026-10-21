"""Rule engine behaviour on scripted scenes (no models, no video)."""

from __future__ import annotations

import numpy as np
import pytest

from snatchguard.core.types import Level, Lighting
from snatchguard.rules.engine import RuleEngine
from snatchguard.sim import synthetic as syn
from snatchguard.sim.run_scenarios import engine_for, outcome, run

SCENARIOS = syn.all_scenarios()


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
@pytest.mark.parametrize("seed", range(3))
def test_scenario_verdict(scenario, seed):
    assert outcome(run(scenario, seed)) in scenario.expect


@pytest.mark.parametrize("fps,jitter,drop", [(6, 2.5, 0.15), (8, 1.5, 0.08), (12, 2.0, 0.1)])
def test_no_false_alarms_and_all_snatches_found_across_frame_rates(fps, jitter, drop):
    """Operating envelope: >= 6 fps processing with noisy tracks and dropped frames."""
    wrong = []
    for sc in syn.all_scenarios():
        sc.fps = fps
        for seed in range(4):
            engine, out = engine_for(sc), []
            for f in sc.frames(seed=seed, jitter_px=jitter, drop_prob=drop):
                out += engine.step(f)
            if outcome(out) not in sc.expect:
                wrong.append((sc.name, seed, outcome(out).value))
    assert not wrong


def test_benign_scenes_never_confirm_even_at_low_fps():
    """Degradation must go towards missing alerts, never towards false CONFIRMED alarms."""
    benign = [s for s in syn.all_scenarios() if Level.CONFIRMED not in s.expect]
    for sc in benign:
        for fps, jitter, drop in [(4, 4.0, 0.3), (5, 3.0, 0.2)]:
            sc.fps = fps
            for seed in range(5):
                engine, out = engine_for(sc), []
                for f in sc.frames(seed=seed, jitter_px=jitter, drop_prob=drop):
                    out += engine.step(f)
                assert outcome(out) is not Level.CONFIRMED, (sc.name, fps, seed)


def test_snatch_alert_explains_itself():
    decisions = run(syn.snatch_from_behind(), 0)
    alert = next(d for d in decisions if d.level is Level.CONFIRMED)
    payload = alert.to_payload()
    passed = {e["key"] for e in payload["evidence"] if e["passed"]}
    assert "contact" in passed
    assert passed & {"bike_speedup", "fast_departure", "left_scene"}
    assert "hands_to_neck" in passed
    assert payload["latency_s"] < 5.0
    assert payload["victim_id"] == 1 and payload["vehicle_id"] == 10


def test_bystander_gets_no_second_alert():
    decisions = run(syn.snatch_with_bystander(), 0)
    alerts = [d for d in decisions if d.level is not Level.DISMISSED]
    assert len(alerts) == 1 and alerts[0].victim_id == 1


def test_riders_are_never_victims():
    decisions = run(syn.snatch_from_behind(), 0)
    assert all(d.victim_id not in (11, 12) for d in decisions)


def test_victim_is_not_absorbed_as_rider_when_bike_slows_alongside():
    engine = RuleEngine()
    for f in syn.snatch_from_behind().frames(seed=3, jitter_px=2.0, drop_prob=0.1):
        engine.step(f)
        assert 1 not in engine.scene.rider_of


def test_phone_call_hand_at_neck_is_not_a_reaction():
    sc = syn.snatch_from_behind()
    victim = next(a for a in sc.actors if a.track_id == 1)
    victim.arms = [syn.ArmAction(0.0, 20.0, "neck", side=1)]   # on the phone the whole time
    decisions = run(sc, 0)
    for d in decisions:
        ev = {e.key: e for e in d.verdict.evidence}
        assert not ev["hands_to_neck"].passed
        assert "ignored" in ev["hands_to_neck"].detail


def test_night_profile_is_used():
    engine = RuleEngine()
    frames = list(syn.snatch_from_behind(Lighting.IR, pose=False, reaction="chase").frames(seed=0))
    engine.step(frames[0])
    assert engine.params.name == "ir"
    assert engine.params.neck_radius > engine.rules.for_lighting(Lighting.DAY).neck_radius


def _victim_track(boxes, t0=0.0, frame=(1280, 720)):
    from snatchguard.core.types import Detection, FrameObservation, ObjClass
    from snatchguard.rules.scene import Scene
    sc, P = Scene(), RuleEngine().params
    for i, b in enumerate(boxes):
        sc.update(FrameObservation(t0 + i * 0.125, frame, [Detection(1, ObjClass.PERSON, np.array(b, float), 0.8)]), P)
    return sc.tracks[1], P


@pytest.mark.parametrize("bottom,expect_fall", [(650.0, True), (716.0, False)])
def test_fall_needs_a_box_not_cut_by_the_frame(bottom, expect_fall):
    """A real fall: the box top drops while the feet stay. A person walking out of the bottom of the frame
    also loses box height, but that box is cut by the edge and says nothing about a fall."""
    from snatchguard.rules.evaluate import eval_reaction
    from snatchguard.rules.pair import PairState
    standing = [[600, bottom - 200, 660, bottom]] * 17                 # 2 s before the contact at t=2.0
    after = [[600, bottom - 200 + k * 25, 660 + (k * 6 if bottom < 700 else 0), bottom] for k in range(1, 6)]
    victim, P = _victim_track(standing + after)
    _, _, ev, _ = eval_reaction(PairState(1, 2), victim, 2.0, 2.2, P)
    assert next(e for e in ev if e.key == "fall").passed is expect_fall
