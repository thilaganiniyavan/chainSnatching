"""Scene model: per-track history plus rider / pedestrian roles.

Everything downstream (features, rules) reads from here, so all timing uses
the observation timestamps. Dropped frames only reduce resolution; they never
change the physics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from snatchguard.core import geometry as g
from snatchguard.core.config import RuleParams
from snatchguard.core.types import Detection, FrameObservation, ObjClass, Pose

TRACK_MEMORY_S = 30.0     # how much history each track keeps
LOST_FORGET_S = 8.0       # lost tracks are forgotten after this long
STITCH_MAX_GAP_S = 1.2    # a new tracker id this soon after a loss, where the lost track was heading,
STITCH_DIST_H = 0.6       # ...within this many body-heights of the predicted spot = the same object


@dataclass(slots=True)
class Obs:
    t: float
    box: np.ndarray
    conf: float
    pose: Pose | None = None
    clip: tuple[bool, bool, bool, bool] = (False, False, False, False)  # box cut by left/top/right/bottom edge


@dataclass
class Track:
    track_id: int
    cls: ObjClass
    obs: deque[Obs] = field(default_factory=deque)
    lost_at: float | None = None
    lost_near_border: bool = False
    cls_votes: deque = field(default_factory=lambda: deque(maxlen=15))
    _memo: dict = field(default_factory=dict, repr=False)   # per-observation cache of derived values

    def vote(self, cls: ObjClass) -> None:
        """Detectors flip a rider+bike blob between 'person' and 'motorcycle'; keep the majority."""
        self.cls_votes.append(cls)
        counts: dict[ObjClass, int] = {}
        for c in self.cls_votes:
            counts[c] = counts.get(c, 0) + 1
        best = max(counts, key=counts.get)
        if counts[best] > counts.get(self.cls, 0):
            self.cls = best

    def append(self, o: Obs) -> None:
        self.obs.append(o)
        self._memo.clear()

    # ---- raw access -------------------------------------------------------
    @property
    def last(self) -> Obs:
        return self.obs[-1]

    @property
    def last_t(self) -> float:
        return self.obs[-1].t

    def window(self, t0: float, t1: float) -> list[Obs]:
        """Observations with t0 <= t <= t1 (scans back from the newest: windows are almost always recent)."""
        out = []
        for o in reversed(self.obs):
            if o.t < t0:
                break
            if o.t <= t1:
                out.append(o)
        out.reverse()
        return out

    def at(self, t: float) -> Obs | None:
        """Latest observation at or before t."""
        for o in reversed(self.obs):
            if o.t <= t:
                return o
        return None

    # ---- derived quantities ----------------------------------------------
    def ref_height(self, t: float, span: float = 1.0) -> float:
        key = ("h", t, span)
        if key in self._memo:
            return self._memo[key]
        hs = sorted(float(o.box[3] - o.box[1]) for o in self.window(t - span, t))
        if not hs:
            val = g.height(self.last.box)
        else:
            n = len(hs)
            val = hs[n // 2] if n % 2 else 0.5 * (hs[n // 2 - 1] + hs[n // 2])
        self._memo[key] = val
        return val

    def velocity(self, t: float, span: float = 0.6, point: str = "foot", min_points: int = 3,
                 max_span: float = 1.5) -> np.ndarray:
        """Least-squares velocity (px/s) of the foot/centre point around time t.

        Uses at least `min_points` observations (reaching back up to `max_span`) so a few
        dropped frames never produce a zero or wild estimate. Boxes cut by the frame edge
        are tracked by their uncut edge, otherwise a bike leaving the frame looks slow.
        """
        key = ("v", t, span, point, min_points, max_span)
        hit = self._memo.get(key)
        if hit is not None:
            return hit.copy()
        pts = self.window(t - span, t)
        if len(pts) < min_points:
            earlier = self.window(t - max_span, t)
            pts = earlier[-min_points:] if len(earlier) >= min_points else earlier
        if len(pts) < 2:
            val = np.zeros(2)
            self._memo[key] = val
            return val.copy()
        left = any(o.clip[0] for o in pts)
        right = any(o.clip[2] for o in pts)
        bottom = any(o.clip[3] for o in pts)
        n = len(pts)
        ts, xs, ys = [], [], []
        for o in pts:
            b = o.box
            if right and not left:
                x = b[0]
            elif left and not right:
                x = b[2]
            else:
                x = (b[0] + b[2]) * 0.5
            if point == "foot":
                y = b[1] if bottom else b[3]
            else:
                y = (b[1] + b[3]) * 0.5
            ts.append(o.t)
            xs.append(float(x))
            ys.append(float(y))
        tm, xm, ym = sum(ts) / n, sum(xs) / n, sum(ys) / n
        denom = sum((ti - tm) ** 2 for ti in ts)
        if denom < 1e-9:
            val = np.zeros(2)
        else:
            vx = sum((ti - tm) * (xi - xm) for ti, xi in zip(ts, xs)) / denom
            vy = sum((ti - tm) * (yi - ym) for ti, yi in zip(ts, ys)) / denom
            val = np.array([vx, vy])
        self._memo[key] = val
        return val.copy()

    def speed(self, t: float, span: float = 0.6) -> float:
        v = self.velocity(t, span)
        return float((v[0] * v[0] + v[1] * v[1]) ** 0.5)

    def latest_pose(self, t: float, max_age: float = 0.4) -> Pose | None:
        for o in reversed(self.obs):
            if o.t > t:
                continue
            if t - o.t > max_age:
                return None
            if o.pose is not None:
                return o.pose
        return None


class Scene:
    """Tracks + role assignment (which people are riding which bike)."""

    def __init__(self) -> None:
        self.tracks: dict[int, Track] = {}
        self.frame_size: tuple[int, int] = (1, 1)
        self.t: float = 0.0
        self._rider_score: dict[tuple[int, int], float] = {}
        self.rider_of: dict[int, int] = {}          # person_id -> bike_id
        self._free_since: dict[int, float] = {}     # person_id -> t since clear of every bike
        self.established: set[int] = set()          # people seen walking on their own
        self._match_since: dict[tuple[int, int], float] = {}  # (person, bike) -> t continuously riding along
        self.locked_pedestrians: set[int] = set()   # set by the rule engine during contacts
        self._alias: dict[int, int] = {}            # tracker id -> stitched (canonical) track id
        self._seen_raw: set[int] = set()

    # ---- update ------------------------------------------------------------
    def update(self, frame: FrameObservation, p: RuleParams) -> None:
        self.t = frame.t
        self.frame_size = frame.frame_size
        seen: set[int] = set()
        for det in frame.detections:
            tid = self._canonical_for(det, frame.t, seen)
            self._add(det, frame.t, frame.poses.get(det.track_id), tid)
            seen.add(tid)
        self._mark_lost(seen, frame.t, p)
        self._update_roles(frame.t, p)

    def canonical(self, raw_id: int) -> int:
        return self._alias.get(raw_id, raw_id)

    def _canonical_for(self, det: Detection, t: float, taken: set[int]) -> int:
        """Track stitching: the tracker often restarts an object under a new id after an occlusion or in
        compressed footage. A brand-new id that appears where a just-lost track of the same kind was heading
        continues that track, so speeds, contacts and getaways stay measurable across the switch."""
        if det.track_id in self._alias or det.track_id in self._seen_raw:
            return self._alias.get(det.track_id, det.track_id)
        self._seen_raw.add(det.track_id)
        box = np.asarray(det.box, dtype=float)
        best, best_d = None, None
        for tr in self.tracks.values():
            if tr.lost_at is None or tr.track_id in taken or tr.cls.is_vehicle != det.cls.is_vehicle:
                continue
            gap = t - tr.last_t
            if gap > STITCH_MAX_GAP_S:
                continue
            h = g.height(tr.last.box)
            if not 0.7 <= g.height(box) / max(h, 1.0) <= 1.4:
                continue
            predicted = g.foot(tr.last.box) + tr.velocity(tr.last_t) * gap
            d = float(np.linalg.norm(g.foot(box) - predicted))
            if d <= max(STITCH_DIST_H * h, 25.0) and (best_d is None or d < best_d):
                best, best_d = tr.track_id, d
        if best is not None:
            self._alias[det.track_id] = best
            return best
        return det.track_id

    def _add(self, det: Detection, t: float, pose: Pose | None, tid: int | None = None) -> None:
        tid = det.track_id if tid is None else tid
        tr = self.tracks.get(tid)
        if tr is None:
            tr = Track(tid, det.cls)
            self.tracks[tid] = tr
        tr.vote(det.cls)
        box = np.asarray(det.box, dtype=float)
        w, h = self.frame_size
        clip = (box[0] <= 2, box[1] <= 2, box[2] >= w - 2, box[3] >= h - 2)
        tr.append(Obs(t, box, det.conf, pose, clip))
        tr.lost_at = None
        while tr.obs and t - tr.obs[0].t > TRACK_MEMORY_S:
            tr.obs.popleft()

    def attach_pose(self, track_id: int, pose: Pose) -> None:
        tr = self.tracks.get(self.canonical(track_id))
        if tr is not None and tr.obs:
            tr.obs[-1].pose = pose
            tr._memo.clear()

    def _mark_lost(self, seen: set[int], t: float, p: RuleParams) -> None:
        w, h = self.frame_size
        for tid in list(self.tracks):
            tr = self.tracks[tid]
            if tid in seen:
                continue
            if tr.lost_at is None:
                tr.lost_at = t
                # At low fps a fast bike can cross the frame edge between two processed frames:
                # project its last box forward to the moment we noticed it missing.
                v = tr.velocity(tr.last_t)
                dt = t - tr.last_t
                projected = tr.last.box + np.array([v[0], v[1], v[0], v[1]]) * dt
                tr.lost_near_border = (g.near_border(tr.last.box, w, h, p.escape_border_px)
                                       or g.near_border(projected, w, h, p.escape_border_px))
            elif t - tr.lost_at > LOST_FORGET_S:
                del self.tracks[tid]
                self.rider_of.pop(tid, None)
                self._free_since.pop(tid, None)
                self.established.discard(tid)
                self._rider_score = {k: v for k, v in self._rider_score.items() if tid not in k}
                self._match_since = {k: v for k, v in self._match_since.items() if tid not in k}
                self._alias = {raw: c for raw, c in self._alias.items() if c != tid}

    # ---- roles -------------------------------------------------------------
    def _rider_match(self, person: Track, bike: Track, t: float, p: RuleParams) -> float:
        """Instantaneous evidence (0..1) that `person` is riding `bike`."""
        pb, bb = person.last.box, bike.last.box
        if person.last_t != t or bike.last_t != t:
            return 0.0
        overlap = g.intersection(pb, bb) / max(g.area(pb), 1.0)
        if overlap < p.rider_min_overlap:
            return 0.0
        foot_y = pb[3]
        bh = g.height(bb)
        if not (bb[1] + 0.1 * bh <= foot_y <= bb[3] + 0.25 * g.height(pb)):
            return 0.0
        cx = (pb[0] + pb[2]) / 2
        bw = g.width(bb)
        if not (bb[0] - 0.15 * bw <= cx <= bb[2] + 0.15 * bw):
            return 0.0
        # Moving together: a pedestrian brushing past a bike moves differently. Box jitter makes
        # velocities noisy, so this only softens the evidence instead of vetoing it.
        vb, vp = bike.velocity(t), person.velocity(t)
        ref = max(float(np.linalg.norm(vb)), 1.0 * g.height(pb))
        if float(np.linalg.norm(vp - vb)) > 0.8 * ref:
            return 0.6
        return 1.0

    def _update_roles(self, t: float, p: RuleParams) -> None:
        persons = [tr for tr in self.tracks.values() if tr.cls is ObjClass.PERSON and tr.lost_at is None]
        bikes = [tr for tr in self.tracks.values() if tr.cls.is_vehicle and tr.lost_at is None]
        for person in persons:
            pid = person.track_id
            best_bike, best_score, touching = None, 0.0, False
            for bike in bikes:
                key = (pid, bike.track_id)
                match = self._rider_match(person, bike, t, p)
                touching |= match > 0
                if match >= 1.0:
                    self._match_since.setdefault(key, t)
                else:
                    self._match_since.pop(key, None)
                score = (1 - p.rider_ema) * self._rider_score.get(key, 0.0) + p.rider_ema * match
                self._rider_score[key] = score
                if score > best_score:
                    best_bike, best_score = bike.track_id, score

            # "Established pedestrian": seen on their own, clear of every bike, for a while.
            # Riders who enter the scene on a bike never qualify, so they never become victims.
            if touching or pid in self.rider_of or not self._looks_on_foot(person, t, p):
                self._free_since.pop(pid, None)
            else:
                since = self._free_since.setdefault(pid, t)
                if t - since >= p.pedestrian_min_free_s:
                    self.established.add(pid)

            current = self.rider_of.get(pid)
            if current is not None:
                if self._rider_score.get((pid, current), 0.0) < p.rider_score_off:
                    del self.rider_of[pid]
            elif best_bike is not None and best_score >= p.rider_score_on:
                # An established pedestrian only "boards" after riding along continuously for a while;
                # a bike slowing beside them for a grab lasts far less. During a contact they never board.
                boarded_for = t - self._match_since.get((pid, best_bike), t)
                if pid not in self.established or (
                        pid not in self.locked_pedestrians and boarded_for >= p.pedestrian_board_s):
                    self.rider_of[pid] = best_bike
                    self.established.discard(pid)

    def looks_on_foot(self, person: Track, t: float, p: RuleParams) -> bool:
        return self._looks_on_foot(person, t, p)

    def _looks_on_foot(self, person: Track, t: float, p: RuleParams) -> bool:
        """Upright standing box and walking/running speed. A rider whose bike the detector
        missed is seated (squat box) and moves at traffic speed, so never becomes a 'pedestrian'."""
        recent = person.window(t - 1.0, t)
        aspects = [g.height(o.box) / max(g.width(o.box), 1.0) for o in recent if not any(o.clip)]
        if aspects and float(np.median(aspects)) < p.pedestrian_min_aspect:
            return False
        H = max(person.ref_height(t), 1.0)
        return person.speed(t, span=1.0) / H <= p.pedestrian_max_speed

    # ---- queries -----------------------------------------------------------
    def pedestrians(self, p: RuleParams | None = None) -> list[Track]:
        """Established pedestrians who still look on foot now: the only people who can be victims."""
        return [
            tr for tr in self.tracks.values()
            if tr.cls is ObjClass.PERSON and tr.lost_at is None
            and tr.track_id in self.established and tr.track_id not in self.rider_of
            and (p is None or self._looks_on_foot(tr, self.t, p))
        ]

    def vehicles(self, include_lost: bool = False) -> list[Track]:
        return [tr for tr in self.tracks.values() if tr.cls.is_vehicle and (include_lost or tr.lost_at is None)]

    def riders(self, bike_id: int) -> list[Track]:
        return [self.tracks[pid] for pid, bid in self.rider_of.items() if bid == bike_id and pid in self.tracks]

    def rider_score(self, person_id: int, bike_id: int) -> float:
        return self._rider_score.get((person_id, bike_id), 0.0)
