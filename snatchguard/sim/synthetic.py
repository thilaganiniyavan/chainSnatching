"""Synthetic scenes: scripted people and bikes with COCO-17 skeletons.

They exercise the rule engine without any video or model. Every scenario is a
known situation (real snatch, passing bike, friends chatting, playful tap,
...) with the verdict it should get, so thresholds can be tuned safely.
`render()` draws a frame so the full edge -> central -> dashboard loop can be
demonstrated with the same scenes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from snatchguard.core.types import Detection, FrameObservation, Level, Lighting, ObjClass, Pose

FRAME_W, FRAME_H = 1280, 720
H = 170.0  # standing person height in px

# Standing skeleton: (dx, dy) in units of body height, dx from box centre, dy from head top.
_TEMPLATE = np.array([
    [0.00, 0.06], [-0.02, 0.05], [0.02, 0.05], [-0.045, 0.06], [0.045, 0.06],   # nose eyes ears
    [-0.11, 0.18], [0.11, 0.18], [-0.14, 0.33], [0.14, 0.33], [-0.15, 0.46], [0.15, 0.46],  # sh el wr
    [-0.08, 0.52], [0.08, 0.52], [-0.08, 0.72], [0.08, 0.72], [-0.08, 0.95], [0.08, 0.95],  # hip knee ankle
])


def lerp_path(keys: list[tuple[float, float, float]], t: float) -> np.ndarray:
    """Piecewise-linear foot position from (t, x, y) keyframes."""
    if t <= keys[0][0]:
        return np.array(keys[0][1:], dtype=float)
    for (t0, x0, y0), (t1, x1, y1) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            a = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
            return np.array([x0 + a * (x1 - x0), y0 + a * (y1 - y0)])
    return np.array(keys[-1][1:], dtype=float)


def integrate_path(t_start: float, x0: float, y: float, speed_keys: list[tuple[float, float]], t_end: float,
                   dt: float = 0.05) -> list[tuple[float, float, float]]:
    """Build keyframes from a speed profile [(t, px/s), ...] (linear between keys)."""
    out = [(t_start, x0, y)]
    x, t = x0, t_start
    while t < t_end:
        v = float(np.interp(t, [k[0] for k in speed_keys], [k[1] for k in speed_keys]))
        x += v * dt
        t += dt
        out.append((t, x, y))
    return out


@dataclass
class ArmAction:
    t0: float
    t1: float
    kind: str                    # "reach" (towards another actor's neck) | "neck" (own neck) | "wave"
    target: int | None = None    # actor id for "reach"
    side: int = 1                # 1 = right wrist, -1 = left wrist
    shoulder_level: bool = False  # reach to the shoulder instead of the neck (friendly pat)


@dataclass
class Actor:
    track_id: int
    cls: ObjClass
    path: list[tuple[float, float, float]]
    box_h: float
    box_w: float
    body_h: float = H
    visible: tuple[float, float] = (0.0, 1e9)
    arms: list[ArmAction] = field(default_factory=list)
    has_pose: bool = True
    fall_at: float | None = None
    impulse: tuple[float, float, float] | None = None   # (t, dx, dy) sudden shove, in px

    def foot(self, t: float) -> np.ndarray:
        f = lerp_path(self.path, t)
        if self.impulse and t >= self.impulse[0]:
            k = min(1.0, (t - self.impulse[0]) / 0.3)
            f = f + k * np.array(self.impulse[1:])
        return f

    def box(self, t: float) -> np.ndarray:
        fx, fy = self.foot(t)
        h, w = self.box_h, self.box_w
        if self.fall_at is not None and t >= self.fall_at:
            k = min(1.0, (t - self.fall_at) / 0.5)
            h, w = h * (1 - 0.5 * k), w * (1 + 1.2 * k)
        return np.array([fx - w / 2, fy - h, fx + w / 2, fy])

    def neck(self, t: float) -> np.ndarray:
        b = self.box(t)
        return np.array([(b[0] + b[2]) / 2, b[1] + 0.17 * self.body_h])

    def pose(self, t: float, actors: dict[int, "Actor"]) -> Pose:
        b = self.box(t)
        cx, top = (b[0] + b[2]) / 2, b[1]
        xy = np.stack([cx + _TEMPLATE[:, 0] * self.body_h, top + _TEMPLATE[:, 1] * self.body_h], axis=1)
        if self.box_h < self.body_h * 0.9:           # seated rider: fold the legs into the box
            xy[11:, 1] = np.minimum(xy[11:, 1], b[3] - 2)
        for arm in self.arms:
            if not (arm.t0 <= t <= arm.t1):
                continue
            wi, ei, si = (10, 8, 6) if arm.side > 0 else (9, 7, 5)
            if arm.kind == "reach" and arm.target in actors:
                tgt_actor = actors[arm.target]
                target = tgt_actor.neck(t)
                if arm.shoulder_level:
                    target = target + np.array([0.0, 0.12 * tgt_actor.body_h])
            elif arm.kind == "neck":
                target = self.neck(t) + np.array([arm.side * 0.03 * self.body_h, 0.0])
            elif arm.kind == "wave":
                target = np.array([xy[si][0] + arm.side * 0.05 * self.body_h, top - 0.08 * self.body_h])
            else:
                continue
            xy[wi] = target
            xy[ei] = (xy[si] + target) / 2 + np.array([0.0, 0.04 * self.body_h])
        return Pose(xy=xy, conf=np.full(17, 0.9))


@dataclass
class Scenario:
    name: str
    description: str
    actors: list[Actor]
    duration: float
    expect: set[Level]                 # acceptable outcomes (DISMISSED = no alert)
    lighting: Lighting = Lighting.DAY
    fps: float = 8.0

    def frames(self, seed: int = 0, jitter_px: float = 1.5, drop_prob: float = 0.08):
        """Yield FrameObservations with tracker-like noise and dropped frames."""
        rng = np.random.default_rng(seed)
        actors = {a.track_id: a for a in self.actors}
        t = 0.0
        while t <= self.duration:
            if rng.random() >= drop_prob:
                dets, poses = [], {}
                for a in self.actors:
                    if not (a.visible[0] <= t <= a.visible[1]):
                        continue
                    box = a.box(t) + rng.normal(0, jitter_px, 4)
                    if box[2] < 0 or box[0] > FRAME_W:
                        continue
                    box = np.clip(box, [0, 0, 0, 0], [FRAME_W, FRAME_H, FRAME_W, FRAME_H])
                    dets.append(Detection(a.track_id, a.cls, box, 0.85))
                    if a.cls is ObjClass.PERSON and a.has_pose:
                        pose = a.pose(t, actors)
                        pose.xy = pose.xy + rng.normal(0, jitter_px, pose.xy.shape)
                        poses[a.track_id] = pose
                yield FrameObservation(t=round(t, 4), frame_size=(FRAME_W, FRAME_H), detections=dets,
                                       poses=poses, lighting=self.lighting)
            t += 1.0 / self.fps * rng.uniform(0.85, 1.15)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _pedestrian(tid: int, keys, **kw) -> Actor:
    return Actor(tid, ObjClass.PERSON, keys, box_h=H, box_w=0.36 * H, **kw)


def _bike_with_riders(bike_id: int, rider_ids: list[int], keys, *, pillion_arms=None,
                      riders_have_pose: bool = True, visible=(0.0, 1e9)) -> list[Actor]:
    bike = Actor(bike_id, ObjClass.MOTORCYCLE, keys, box_h=0.55 * H, box_w=0.95 * H, visible=visible)
    actors = [bike]
    for i, rid in enumerate(rider_ids):
        dx = 0.12 * H - i * 0.30 * H          # driver in front, pillion behind (bike moves +x)
        rkeys = [(t, x + dx, y - 0.06 * H) for t, x, y in keys]
        actors.append(Actor(rid, ObjClass.PERSON, rkeys, box_h=0.80 * H, box_w=0.34 * H, visible=visible,
                            arms=(pillion_arms or []) if i == len(rider_ids) - 1 else [],
                            has_pose=riders_have_pose))
    return actors


VY, BY = 520.0, 540.0     # victim foot y, bike foot y (bike slightly nearer the camera)


def _walk(x0: float, speed_hs: float, t_end: float, y: float = VY):
    return integrate_path(0.0, x0, y, [(0.0, speed_hs * H), (t_end, speed_hs * H)], t_end)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

# Scenes start with 1.5 s lead-in (bike still off-screen), like a live camera that was already running.
LEAD = 1.5
LEAD_X = -2.5 * H * LEAD


def _encounter(profile_hs: list[tuple[float, float]], t_contact: float, victim_speed_hs: float, duration: float, *,
               n_riders: int = 2, bike_x0: float = -150.0 + LEAD_X, bike_y: float = BY, victim_y: float = VY,
               riders_pose: bool = True, victim_pose: bool = True) -> tuple[Actor, list[Actor], Actor]:
    """Bike follows `profile_hs` (t, speed in H/s); the victim is placed so the last rider
    is right beside them at `t_contact`."""
    bike_keys = integrate_path(0.0, bike_x0, bike_y, [(t, v * H) for t, v in profile_hs], duration)
    riders = _bike_with_riders(10, list(range(11, 11 + n_riders)), bike_keys, riders_have_pose=riders_pose)
    toucher = riders[-1]
    direction = 1.0 if profile_hs[0][1] >= 0 else -1.0
    x_contact = toucher.foot(t_contact)[0] + direction * 0.05 * H
    x0 = x_contact - victim_speed_hs * H * t_contact
    victim = _pedestrian(1, _walk(x0, victim_speed_hs, duration, y=victim_y), has_pose=victim_pose)
    return victim, riders, toucher


# cruise 2.5 H/s, slow to 0.9 H/s beside the victim, burst away at 4.5 H/s
SNATCH_PROFILE = [(0.0, 2.5), (LEAD + 1.2, 2.5), (LEAD + 2.2, 0.9), (LEAD + 3.2, 0.9), (LEAD + 3.8, 4.5), (14.0, 4.5)]
T_SNATCH = LEAD + 2.7


def snatch_from_behind(lighting: Lighting = Lighting.DAY, pose: bool = True, reaction: str = "neck") -> Scenario:
    """Classic chain snatch: bike comes from behind, slows, pillion grabs the neck, bike speeds off."""
    victim, riders, pillion = _encounter(SNATCH_PROFILE, T_SNATCH, 0.7, 12.0, riders_pose=pose, victim_pose=pose)
    tc = T_SNATCH
    pillion.arms = [ArmAction(tc - 0.1, tc + 0.5, "reach", target=1, side=1)]
    if reaction in ("neck", "chase"):
        victim.impulse = (tc + 0.1, 0.12 * H, 0.0)
    if reaction == "neck":
        victim.arms = [ArmAction(tc + 0.4, tc + 2.5, "neck", side=1), ArmAction(tc + 0.5, tc + 2.5, "neck", side=-1)]
    elif reaction == "chase":
        victim.path = _merge_paths(victim.path, tc + 0.6, 2.2 * H, 12.0)
    elif reaction == "fall":
        victim.fall_at = tc + 0.3
    return Scenario(
        f"snatch_from_behind[{lighting.value},{'pose' if pose else 'nopose'},{reaction}]",
        "Bike approaches from behind, slows beside the victim, pillion grabs the neck, bike accelerates away.",
        [victim] + riders, 9.5,
        expect={Level.CONFIRMED} if (pose or reaction != "none") else {Level.CONFIRMED, Level.SUSPECT},
        lighting=lighting)


def snatch_head_on() -> Scenario:
    """Oncoming bike (opposite direction to the victim) slows, grabs, speeds away."""
    profile = [(t, -v) for t, v in SNATCH_PROFILE]
    victim, riders, pillion = _encounter(profile, T_SNATCH, 0.7, 12.0, bike_x0=1450.0 - LEAD_X)
    victim.path = _walk(victim.path[0][1], 0.7, 12.0)
    pillion.arms = [ArmAction(T_SNATCH - 0.1, T_SNATCH + 0.5, "reach", target=1)]
    victim.arms = [ArmAction(T_SNATCH + 0.4, T_SNATCH + 2.5, "neck", side=1)]
    return Scenario("snatch_head_on", "Oncoming bike slows, pillion grabs, bike accelerates away.",
                    [victim] + riders, 9.5, expect={Level.CONFIRMED})


def passing_bike() -> Scenario:
    """A bike overtakes a pedestrian at constant speed; boxes overlap briefly. Nothing happens."""
    victim, riders, _ = _encounter([(0.0, 3.0), (14.0, 3.0)], LEAD + 2.5, 0.7, 10.0, bike_x0=-150.0 - 3.0 * H * LEAD)
    return Scenario("passing_bike", "Bike passes close by at constant speed.",
                    [victim] + riders, 9.5, expect={Level.DISMISSED})


def passing_bike_other_depth() -> Scenario:
    """Bike passes between the camera and the pedestrian: overlap in 2D, metres apart in the world.
    It even slows and speeds up, so only the depth check keeps it quiet."""
    victim, riders, _ = _encounter(SNATCH_PROFILE, T_SNATCH, 0.7, 12.0, bike_y=660.0, victim_y=470.0)
    return Scenario("passing_bike_other_depth", "Bike passes in front of the pedestrian (different depth).",
                    [victim] + riders, 9.5, expect={Level.DISMISSED})


def friends_chat_then_leave() -> Scenario:
    """Bike pulls up next to a friend, they talk for 8 s, a pat on the shoulder, bike rides off."""
    profile = [(0.0, 2.0), (1.0, 1.0), (1.6, 0.0), (10.2, 0.0), (10.8, 1.5), (12.0, 4.0), (18.0, 4.0)]
    victim, riders, rider = _encounter(profile, 9.5, 0.0, 18.0, n_riders=1, bike_x0=150.0)
    rider.arms = [ArmAction(9.4, 10.1, "reach", target=1, shoulder_level=True)]
    victim.arms = [ArmAction(10.8, 12.5, "wave", side=1)]
    return Scenario("friends_chat_then_leave", "Long stationary chat, friendly pat, bike leaves, victim waves.",
                    [victim] + riders, 16.0, expect={Level.DISMISSED})


def playful_tap_and_go() -> Scenario:
    """A friend on a bike slows, taps the pedestrian's shoulder and speeds off. Victim waves back."""
    victim, riders, pillion = _encounter(SNATCH_PROFILE, T_SNATCH, 0.7, 12.0)
    pillion.arms = [ArmAction(T_SNATCH - 0.1, T_SNATCH + 0.4, "reach", target=1, shoulder_level=True)]
    victim.arms = [ArmAction(T_SNATCH + 0.8, T_SNATCH + 2.5, "wave", side=1)]
    return Scenario("playful_tap_and_go", "Friendly shoulder tap from a passing bike, victim waves, no distress.",
                    [victim] + riders, 9.5, expect={Level.DISMISSED, Level.SUSPECT})


def handover_stationary() -> Scenario:
    """Bike stops, rider hands something over for 3 s, bike leaves slowly (delivery / helping)."""
    profile = [(0.0, 2.0), (1.0, 0.8), (1.6, 0.0), (6.5, 0.0), (7.5, 0.6), (9.0, 1.2), (16.0, 1.5)]
    victim, riders, rider = _encounter(profile, 4.0, 0.0, 16.0, n_riders=1, bike_x0=150.0)
    rider.arms = [ArmAction(2.5, 5.8, "reach", target=1, shoulder_level=True)]
    return Scenario("handover_stationary", "Stopped bike, 3 s handover at chest height, slow departure.",
                    [victim] + riders, 14.0, expect={Level.DISMISSED})


def hug_goodbye() -> Scenario:
    """Pillion hugs the pedestrian around the neck for 2.5 s while stopped, then the bike leaves quickly."""
    profile = [(0.0, 2.0), (1.0, 0.5), (1.6, 0.0), (7.5, 0.0), (8.0, 2.0), (9.0, 4.5), (16.0, 4.5)]
    victim, riders, pillion = _encounter(profile, 6.0, 0.0, 16.0, bike_x0=250.0)
    pillion.arms = [ArmAction(4.6, 7.2, "reach", target=1)]
    return Scenario("hug_goodbye", "Long neck hug while stopped (friends), then a quick departure.",
                    [victim] + riders, 14.0, expect={Level.DISMISSED, Level.SUSPECT})


def walking_alongside() -> Scenario:
    """A pedestrian walks next to a slowly-ridden bike for 8 s, touches the rider, they split up."""
    profile = [(0.0, 0.8), (9.5, 0.8), (10.5, 3.0), (16.0, 4.0)]
    victim, riders, rider = _encounter(profile, 9.0, 0.8, 16.0, n_riders=1, bike_x0=100.0)
    # keep the pedestrian a little ahead/beside so boxes do not overlap while walking together
    victim.path = [(t, x + 0.45 * H, y) for t, x, y in victim.path]
    rider.arms = [ArmAction(9.0, 9.6, "reach", target=1, shoulder_level=True)]
    return Scenario("walking_alongside", "Friends moving together for 9 s, then the bike rides off.",
                    [victim] + riders, 14.0, expect={Level.DISMISSED, Level.SUSPECT})


def phone_call_passing() -> Scenario:
    """Pedestrian holds a phone to the ear (hand at the neck all along) while a bike passes close."""
    victim, riders, _ = _encounter([(0.0, 3.0), (14.0, 3.2)], LEAD + 2.5, 0.7, 10.0, bike_x0=-150.0 - 3.0 * H * LEAD)
    victim.arms = [ArmAction(0.0, 12.0, "neck", side=1)]
    return Scenario("phone_call_passing", "On a phone call while a bike passes close by.",
                    [victim] + riders, 9.5, expect={Level.DISMISSED})


def snatch_with_bystander() -> Scenario:
    """A snatch next to a second pedestrian: exactly one alert should come out."""
    sc = snatch_from_behind()
    victim = next(a for a in sc.actors if a.track_id == 1)
    bystander = _pedestrian(2, [(t, x - 0.9 * H, y - 15) for t, x, y in victim.path])
    sc.actors.append(bystander)
    sc.name = "snatch_with_bystander"
    return sc



def all_scenarios() -> list[Scenario]:
    return [
        snatch_from_behind(),
        snatch_from_behind(reaction="chase"),
        snatch_from_behind(reaction="fall"),
        snatch_from_behind(Lighting.IR, pose=False, reaction="chase"),
        snatch_from_behind(Lighting.LOWLIGHT, pose=True, reaction="neck"),
        snatch_head_on(),
        passing_bike(),
        passing_bike_other_depth(),
        snatch_with_bystander(),
        friends_chat_then_leave(),
        playful_tap_and_go(),
        handover_stationary(),
        hug_goodbye(),
        walking_alongside(),
        phone_call_passing(),
    ]


SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "snatch": snatch_from_behind,
    "snatch_chase": lambda: snatch_from_behind(reaction="chase"),
    "snatch_night_ir": lambda: snatch_from_behind(Lighting.IR, pose=False, reaction="chase"),
    "snatch_head_on": snatch_head_on,
    "passing_bike": passing_bike,
    "friends_chat": friends_chat_then_leave,
    "playful_tap": playful_tap_and_go,
    "handover": handover_stationary,
    "hug": hug_goodbye,
    "walking_alongside": walking_alongside,
    "phone_call": phone_call_passing,
}


# ---------------------------------------------------------------------------

def _first_time(cond, t0: float, t1: float, dt: float = 0.02) -> float:
    t = t0
    while t <= t1:
        if cond(t):
            return t
        t += dt
    raise ValueError("scenario never reaches the requested condition")


def _merge_paths(walk, t_switch: float, run_speed: float, t_end: float):
    x_switch = lerp_path(walk, t_switch)[0]
    keep = [k for k in walk if k[0] < t_switch]
    return keep + integrate_path(t_switch, x_switch, walk[0][2], [(t_switch, run_speed), (t_end, run_speed)], t_end)


_SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
             (12, 14), (14, 16), (0, 5), (0, 6)]


def render(scenario: Scenario, frame: FrameObservation) -> np.ndarray:
    """Draw a synthetic frame (for evidence clips and dashboard demos)."""
    night = scenario.lighting is not Lighting.DAY
    img = np.full((FRAME_H, FRAME_W, 3), (40, 40, 40) if night else (200, 205, 200), np.uint8)
    cv2.rectangle(img, (0, 500), (FRAME_W, FRAME_H), (60, 60, 60) if night else (150, 150, 150), -1)
    cv2.line(img, (0, 500), (FRAME_W, 500), (90, 90, 90), 2)
    for det in frame.detections:
        x1, y1, x2, y2 = det.box.astype(int)
        if det.cls.is_vehicle:
            cv2.rectangle(img, (x1, y1 + 10), (x2, y2 - 12), (30, 30, 120), -1)
            for cx in (x1 + 25, x2 - 25):
                cv2.circle(img, (cx, y2 - 12), 12, (10, 10, 10), -1)
        else:
            pose = frame.poses.get(det.track_id)
            if pose is not None:
                pts = pose.xy.astype(int)
                for a, b in _SKELETON:
                    cv2.line(img, tuple(pts[a]), tuple(pts[b]), (220, 220, 220) if night else (50, 50, 50), 5)
                cv2.circle(img, tuple(pts[0]), 14, (220, 220, 220) if night else (50, 50, 50), -1)
            else:
                cv2.rectangle(img, (x1 + 8, y1), (x2 - 8, y2), (130, 130, 130), -1)
    cv2.putText(img, f"SIMULATED  {scenario.name}  t={frame.t:5.1f}s", (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255) if night else (0, 0, 0), 2)
    return img
