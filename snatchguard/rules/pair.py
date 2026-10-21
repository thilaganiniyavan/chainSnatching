"""Per-frame measurements for one (victim, bike) pair.

A PairSample is the only thing the evaluator looks at, so every rule works on
the same numbers the dashboard later shows as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from snatchguard.core import geometry as g
from snatchguard.core.config import RuleParams
from snatchguard.core.types import L_SHOULDER, NOSE, R_SHOULDER, WRISTS, ObjClass, Pose
from snatchguard.rules.scene import Scene, Track

CONTACT_NONE, CONTACT_BOX, CONTACT_POSE = 0, 1, 2


@dataclass(slots=True)
class PairSample:
    t: float
    H: float                     # victim reference height (px)
    gap: float                   # closest victim <-> attacker box distance (H)
    bike_speed: float            # H/s
    victim_speed: float          # H/s
    bike_vel: np.ndarray         # H/s
    victim_vel: np.ndarray       # H/s
    away_dir: np.ndarray         # unit vector victim -> bike
    depth_ok: bool
    contact: int                 # CONTACT_* level
    wrist_neck: float | None     # closest rider wrist to victim neck (H)
    hands_at_neck: bool | None   # None = victim pose not visible
    waving: bool
    victim_h: float              # raw victim box height (px) this frame
    victim_aspect: float         # width / height
    victim_pose: bool
    bike_lost: bool
    crowd: int = 0               # other people within reach of the victim


@dataclass
class ContactEpisode:
    start: float
    end: float
    level: int
    samples: int = 1
    min_wrist_neck: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


def neck_point(victim_box: np.ndarray, pose: Pose | None, H: float, min_conf: float) -> tuple[np.ndarray, bool]:
    """Neck estimate from shoulders/nose; falls back to a box prior. Returns (point, from_pose)."""
    if pose is not None:
        ls, rs = pose.point(L_SHOULDER, min_conf), pose.point(R_SHOULDER, min_conf)
        nose = pose.point(NOSE, min_conf)
        if ls is not None and rs is not None:
            mid = (ls + rs) / 2.0
            return (mid + 0.25 * (nose - mid) if nose is not None else mid - np.array([0.0, 0.04 * H])), True
        if nose is not None:
            return nose + np.array([0.0, 0.10 * H]), True
    cx = (victim_box[0] + victim_box[2]) / 2.0
    return np.array([cx, victim_box[1] + 0.17 * H]), False


def attacker_boxes(scene: Scene, bike: Track) -> tuple[list[np.ndarray], list[Track]]:
    """Rider boxes if riders are known; otherwise the bike box stretched up to where a rider sits.
    For an attacker on foot, the person themself (their wrists are checked like a rider's)."""
    if bike.cls is ObjClass.PERSON:
        return [bike.last.box], [bike] if bike.lost_at is None else []
    riders = [r for r in scene.riders(bike.track_id) if r.lost_at is None]
    if riders:
        return [r.last.box for r in riders], riders
    b = bike.last.box.copy()
    b[1] -= 0.9 * g.height(b)
    return [b], []


def _people_near(scene: Scene, victim: Track, H: float, radius: float) -> list[int]:
    """Live people within `radius` (H) of the victim this frame; shared by all of the victim's pairs."""
    key = ("near", scene.t, H, radius)
    hit = victim._memo.get(key)
    if hit is None:
        vbox = victim.last.box
        hit = [tr.track_id for tr in scene.tracks.values()
               if tr.cls is ObjClass.PERSON and tr.lost_at is None and tr.track_id != victim.track_id
               and g.box_gap(vbox, tr.last.box) / H <= radius]
        victim._memo[key] = hit
    return hit


def build_sample(scene: Scene, victim: Track, bike: Track, p: RuleParams, seated: bool = False) -> PairSample:
    t = scene.t
    vbox = victim.last.box
    H = max(victim.ref_height(t), 1.0)
    boxes, riders = attacker_boxes(scene, bike)
    bike_lost = bike.lost_at is not None

    gap = min(g.box_gap(vbox, b) for b in boxes) / H
    dy = float(vbox[3] - bike.last.box[3]) / H
    if seated:
        # a seated rider's box ends at the footrest, above the ground: it may sit higher in the image
        depth_ok = -p.same_depth_tol <= dy <= p.same_depth_tol + p.seated_ground_offset
    else:
        depth_ok = abs(dy) <= p.same_depth_tol
    if depth_ok and riders:
        lo, hi = p.scale_ratio_range
        depth_ok = any(lo <= g.height(r.last.box) / H <= hi for r in riders)

    bike_vel = (bike.velocity(bike.last_t) if bike_lost else bike.velocity(t)) / H
    victim_vel = victim.velocity(t) / H
    away = g.center(bike.last.box) - g.center(vbox)
    away_dir = away / (np.linalg.norm(away) + 1e-9)

    # Keypoints of small, distant people are noise: only trust pose cues above a minimum size.
    vpose = victim.latest_pose(t) if H >= p.min_pose_height_px else None
    neck, neck_from_pose = neck_point(vbox, vpose, H, p.kpt_min_conf)

    # ---- CONTACT -----------------------------------------------------------
    contact = CONTACT_NONE
    wrist_neck = None
    # Too close to the camera (cut by the bottom edge) or head cut off: depth, scale and the neck
    # can't be judged. Objects cut at the sides are still usable (entering / leaving the view).
    vc, bc = victim.last.clip, bike.last.clip
    cut_off = vc[1] or vc[3] or bc[3] or any(r.last.clip[3] for r in riders)
    # A person box and a motorcycle box that nearly coincide are one rider on their own bike.
    cut_off = cut_off or g.iou(vbox, bike.last.box) > 0.5
    if depth_ok and not bike_lost and not cut_off:
        for r in riders:
            rpose = r.latest_pose(t) if g.height(r.last.box) >= 0.6 * p.min_pose_height_px else None
            if rpose is None:
                continue
            for w in WRISTS:
                pt = rpose.point(w, p.kpt_min_conf)
                if pt is None:
                    continue
                d = float(np.linalg.norm(pt - neck)) / H
                wrist_neck = d if wrist_neck is None else min(wrist_neck, d)
        if wrist_neck is not None and wrist_neck <= p.neck_radius and neck_from_pose:
            contact = CONTACT_POSE
        else:
            upper = np.array([vbox[0], vbox[1], vbox[2], vbox[1] + p.box_contact_upper_frac * H])
            upper_area = max(g.area(upper), 1.0)
            if any(g.intersection(upper, b) / upper_area >= p.box_contact_min_overlap for b in boxes):
                contact = CONTACT_BOX

    # ---- victim self-observations -------------------------------------------
    hands_at_neck: bool | None = None
    waving = False
    if vpose is not None and neck_from_pose:
        hands_at_neck = False
        nose = vpose.point(NOSE, p.kpt_min_conf)
        for w in WRISTS:
            pt = vpose.point(w, p.kpt_min_conf)
            if pt is None:
                continue
            if float(np.linalg.norm(pt - neck)) / H <= p.hands_to_neck_radius:
                hands_at_neck = True
            if nose is not None and (nose[1] - pt[1]) / H >= p.wave_above_nose:
                waving = True

    crowd = sum(1 for tid in _people_near(scene, victim, H, p.crowd_radius)
                if tid != bike.track_id and scene.rider_of.get(tid) != bike.track_id)

    return PairSample(
        t=t,
        H=H,
        gap=gap,
        bike_speed=float(np.linalg.norm(bike_vel)),
        victim_speed=float(np.linalg.norm(victim_vel)),
        bike_vel=bike_vel,
        victim_vel=victim_vel,
        away_dir=away_dir,
        depth_ok=depth_ok,
        contact=contact,
        wrist_neck=wrist_neck,
        hands_at_neck=hands_at_neck,
        waving=waving,
        victim_h=g.height(vbox),
        victim_aspect=g.width(vbox) / max(g.height(vbox), 1.0),
        victim_pose=vpose is not None,
        bike_lost=bike_lost,
        crowd=crowd,
    )


@dataclass
class PairState:
    victim_id: int
    bike_id: int                    # the attacker: a vehicle, or a person when on_foot / seated
    on_foot: bool = False
    seated: bool = False            # attacker is a seated person whose two-wheeler was not detected
    samples: list[PairSample] = field(default_factory=list)
    episodes: list[ContactEpisode] = field(default_factory=list)
    phase: str = "WATCH"            # WATCH | CONTACT | COOLDOWN
    phase_contact_start: float | None = None
    escape_t: float | None = None
    cooldown_until: float = 0.0
    far_since: float | None = None

    @property
    def key(self) -> tuple[int, int]:
        return (self.victim_id, self.bike_id)

    def add(self, s: PairSample, p: RuleParams, keep_s: float) -> ContactEpisode | None:
        """Store a sample; returns a newly started contact episode, if any."""
        self.samples.append(s)
        while self.samples and s.t - self.samples[0].t > keep_s:
            self.samples.pop(0)
        self.episodes = [e for e in self.episodes if s.t - e.end <= keep_s]
        if s.contact == CONTACT_NONE:
            return None
        last = self.episodes[-1] if self.episodes else None
        if last is not None and s.t - last.end <= p.contact_gap_tolerance_s:
            last.end = s.t
            last.level = max(last.level, s.contact)
            last.samples += 1
            if s.wrist_neck is not None and s.contact == CONTACT_POSE:
                last.min_wrist_neck = s.wrist_neck if last.min_wrist_neck is None else min(last.min_wrist_neck, s.wrist_neck)
            return None
        ep = ContactEpisode(s.t, s.t, s.contact, 1, s.wrist_neck if s.contact == CONTACT_POSE else None)
        self.episodes.append(ep)
        return ep

    def between(self, t0: float, t1: float) -> list[PairSample]:
        return [s for s in self.samples if t0 <= s.t <= t1]
