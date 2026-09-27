"""Scene model details that matter on real CCTV: class flips, seated riders, model smoke test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from snatchguard.core.config import REPO_ROOT, RuleConfig
from snatchguard.core.types import Detection, FrameObservation, Lighting, ObjClass
from snatchguard.rules.scene import Scene

P = RuleConfig.load().for_lighting(Lighting.DAY)


def _frame(t, dets):
    return FrameObservation(t, (1280, 720), dets, {}, Lighting.DAY)


def test_track_keeps_majority_class_when_detector_flips():
    sc = Scene()
    box = np.array([100, 100, 180, 260.0])
    for i in range(10):
        cls = ObjClass.MOTORCYCLE if i % 4 == 3 else ObjClass.PERSON   # occasional flip
        sc.update(_frame(i * 0.125, [Detection(7, cls, box, 0.5)]), P)
    assert sc.tracks[7].cls is ObjClass.PERSON
    assert len(sc.tracks[7].obs) == 10                                  # history kept across flips


def test_seated_person_is_not_established_as_pedestrian():
    """A rider whose scooter the detector missed: squat box -> never a possible victim."""
    sc = Scene()
    for i in range(20):
        x = 100 + i * 5
        sc.update(_frame(i * 0.125, [Detection(3, ObjClass.PERSON, np.array([x, 300, x + 70, 400.0]), 0.6)]), P)
    assert 3 not in sc.established


def test_fast_moving_person_is_not_established():
    """Moving at traffic speed (bike not detected) -> not a pedestrian."""
    sc = Scene()
    for i in range(20):
        x = 100 + i * 60                                  # ~480 px/s for a 170 px person = 2.8 H/s
        sc.update(_frame(i * 0.125, [Detection(4, ObjClass.PERSON, np.array([x, 300, x + 60, 470.0]), 0.6)]), P)
    assert 4 not in sc.established


def test_walking_person_is_established():
    sc = Scene()
    for i in range(20):
        x = 100 + i * 12                                  # ~0.6 H/s walk
        sc.update(_frame(i * 0.125, [Detection(5, ObjClass.PERSON, np.array([x, 300, x + 60, 470.0]), 0.6)]), P)
    assert 5 in sc.established


MODEL = REPO_ROOT / "models" / "yolo26n_512_openvino_model"


@pytest.mark.skipif(not MODEL.exists(), reason="run scripts/fetch_models.py first")
def test_frame_pipeline_smoke():
    from snatchguard.edge.pipeline import FramePipeline
    import ultralytics
    import cv2
    img = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    models = {"detector": str(MODEL), "pose": str(REPO_ROOT / "models" / "yolo26n-pose_openvino_model"),
              "imgsz": 512, "pose_imgsz": 384, "conf": {"day": 0.1}}
    pipe = FramePipeline(models, str(REPO_ROOT / "configs" / "tracker.yaml"), RuleConfig.load())
    for i in range(3):
        res = pipe.process(img, i * 0.125)
    assert any(d.cls is ObjClass.PERSON for d in res.detections)
    assert res.mode is Lighting.DAY


def test_established_person_who_sits_down_on_a_scooter_is_no_longer_a_possible_victim():
    """Seen standing earlier, later seated on a scooter at a signal: must drop out of pedestrians()."""
    sc = Scene()
    t = 0.0
    for i in range(12):                                   # standing still, upright box
        t = i * 0.125
        sc.update(_frame(t, [Detection(8, ObjClass.PERSON, np.array([400, 200, 460, 370.0]), 0.6)]), P)
    assert 8 in {tr.track_id for tr in sc.pedestrians(P)}
    for i in range(12, 24):                               # now a squat, seated box
        t = i * 0.125
        sc.update(_frame(t, [Detection(8, ObjClass.PERSON, np.array([395, 260, 470, 370.0]), 0.6)]), P)
    assert 8 not in {tr.track_id for tr in sc.pedestrians(P)}


def test_person_box_coinciding_with_motorcycle_box_is_never_a_contact():
    from snatchguard.rules.pair import CONTACT_NONE, build_sample
    sc = Scene()
    for i in range(12):
        t = i * 0.125
        sc.update(_frame(t, [Detection(1, ObjClass.PERSON, np.array([400, 200, 460, 370.0]), 0.6),
                             Detection(2, ObjClass.MOTORCYCLE, np.array([402, 205, 462, 372.0]), 0.6)]), P)
    s = build_sample(sc, sc.tracks[1], sc.tracks[2], P)
    assert s.contact == CONTACT_NONE
