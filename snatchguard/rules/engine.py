"""Rule engine: opens (victim, bike) pairs, runs each pair's state machine, emits decisions.

Per frame the edge node calls, in order:
    engine.ingest(frame)            # detections + tracks -> scene, pairs
    engine.pose_requests(max_n)     # which regions deserve a pose pass
    engine.attach_poses({...})      # pose results for those regions
    engine.evaluate_frame()         # -> list[Decision]
`engine.step(frame)` does all of it when poses are already in the frame (simulator, tests).

Pair state machine:
    WATCH --contact--> CONTACT --escape+hold | timeout--> (decide) --> COOLDOWN --> WATCH
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from snatchguard.core import geometry as g
from snatchguard.core.config import RuleConfig, RuleParams
from snatchguard.core.types import FrameObservation, Level, Lighting, ObjClass, Pose
from snatchguard.rules.evaluate import Verdict, eval_escape, evaluate
from snatchguard.rules.pair import PairState, attacker_boxes, build_sample
from snatchguard.rules.scene import Scene

log = logging.getLogger(__name__)

POSE_GAP = 1.5          # pairs closer than this (H) get pose estimation
LOST_BIKE_DROP_S = 1.0  # an idle pair whose bike vanished is dropped after this


@dataclass
class Decision:
    verdict: Verdict
    t: float
    victim_id: int
    bike_id: int
    rider_ids: list[int]
    lighting: Lighting
    boxes: dict[str, list] = field(default_factory=dict)   # role -> [x1, y1, x2, y2] (for snapshots)
    mode: str = "motorcycle"                                # motorcycle | bicycle | on_foot

    @property
    def level(self) -> Level:
        return self.verdict.level

    def to_payload(self) -> dict:
        v = self.verdict
        return {
            "level": v.level.value,
            "mode": self.mode,
            "score": v.score,
            "confidence": v.confidence,
            "lighting": self.lighting.value,
            "contact_start": round(v.contact_start, 3),
            "contact_end": round(v.contact_end, 3),
            "decided_at": round(self.t, 3),
            "latency_s": round(self.t - v.contact_start, 3),
            "victim_id": self.victim_id,
            "vehicle_id": self.bike_id,
            "rider_ids": self.rider_ids,
            "components": {k: round(x, 3) for k, x in v.components.items()},
            "penalties": v.penalties,
            "evidence": [e.as_dict() for e in v.evidence],
            "boxes": self.boxes,
        }


class RuleEngine:
    def __init__(self, rules: RuleConfig | None = None) -> None:
        self.rules = rules or RuleConfig.load()
        self.scene = Scene()
        self.pairs: dict[tuple[int, int], PairState] = {}
        self._bike_cooldown: dict[int, float] = {}
        self._victim_cooldown: dict[int, float] = {}
        self._lighting = Lighting.DAY

    @property
    def params(self) -> RuleParams:
        return self.rules.for_lighting(self._lighting)

    def params_for(self, pair: PairState) -> RuleParams:
        return self.rules.for_lighting(self._lighting, foot=pair.on_foot)

    # ------------------------------------------------------------------
    def step(self, frame: FrameObservation) -> list[Decision]:
        self.ingest(frame)
        return self.evaluate_frame()

    def ingest(self, frame: FrameObservation) -> None:
        self._lighting = frame.lighting
        p = self.params
        self.scene.update(frame, p)
        self._open_pairs(p)

    def attach_poses(self, poses: dict[int, Pose]) -> None:
        for tid, pose in poses.items():
            self.scene.attach_pose(tid, pose)

    def pose_requests(self, max_crops: int = 2) -> list[np.ndarray]:
        """Regions (x1, y1, x2, y2) around the closest pairs, most urgent first."""
        scene = self.scene
        w, h = scene.frame_size
        ranked: list[tuple[float, np.ndarray]] = []
        for pair in self.pairs.values():
            victim, bike = scene.tracks.get(pair.victim_id), scene.tracks.get(pair.bike_id)
            if victim is None or bike is None or victim.lost_at is not None or bike.lost_at is not None:
                continue
            boxes, riders = attacker_boxes(scene, bike)
            H = max(victim.ref_height(scene.t), 1.0)
            gap = min(g.box_gap(victim.last.box, b) for b in boxes) / H
            urgent = pair.phase == "CONTACT"
            if gap > POSE_GAP and not urgent:
                continue
            region = g.union_box([victim.last.box, bike.last.box] + [r.last.box for r in riders])
            ranked.append((-1.0 if urgent else gap, g.pad_box(region, 0.15, w, h)))
        ranked.sort(key=lambda x: x[0])
        crops: list[np.ndarray] = []
        for _, box in ranked:
            if any(g.iou(box, c) > 0.5 for c in crops):
                continue
            crops.append(box)
            if len(crops) >= max_crops:
                break
        return crops

    # ------------------------------------------------------------------
    def _open_pairs(self, p: RuleParams) -> None:
        scene = self.scene
        for victim in scene.pedestrians(p):
            H = victim.ref_height(scene.t)
            if H < p.min_victim_height_px:
                continue
            for bike in scene.vehicles():
                key = (victim.track_id, bike.track_id)
                if key in self.pairs:
                    continue
                # People still being recognised as this bike's riders are not victims.
                if scene.rider_score(victim.track_id, bike.track_id) >= p.rider_score_off:
                    continue
                boxes, _ = attacker_boxes(scene, bike)
                gap = min(g.box_gap(victim.last.box, b) for b in boxes) / H
                if gap <= p.pair_open_gap:
                    self.pairs[key] = PairState(victim.track_id, bike.track_id)
            if p.foot_enabled:
                self._open_foot_pairs(victim, H, p)

    def _open_foot_pairs(self, victim, H: float, p: RuleParams) -> None:
        """Someone on foot closing in on (or already right next to) an established pedestrian."""
        scene = self.scene
        cv = g.center(victim.last.box)
        vv = victim.velocity(scene.t)
        candidates = []
        for other in scene.tracks.values():
            if (other.cls is not ObjClass.PERSON or other.track_id == victim.track_id or other.lost_at is not None
                    or other.track_id in scene.rider_of):
                continue
            key = (victim.track_id, other.track_id)
            if key in self.pairs:
                continue
            gap = g.box_gap(victim.last.box, other.last.box) / H
            if gap > p.foot_pair_open_gap:
                continue
            u = g.center(other.last.box) - cv
            u = u / (np.linalg.norm(u) + 1e-9)
            closing = -float(np.dot(other.velocity(scene.t) - vv, u)) / H
            if closing >= p.foot_min_closing or gap <= p.foot_close_gap:
                candidates.append((gap, key))
        open_now = sum(1 for k, pr in self.pairs.items() if k[0] == victim.track_id and pr.on_foot)
        for _, key in sorted(candidates)[: max(0, p.foot_max_pairs_per_victim - open_now)]:
            self.pairs[key] = PairState(key[0], key[1], on_foot=True)

    def evaluate_frame(self) -> list[Decision]:
        scene = self.scene
        t = scene.t
        decisions: list[Decision] = []

        for key, pair in list(self.pairs.items()):
            p = self.params_for(pair)
            keep_s = p.history_window_s + p.approach_window_s
            victim, bike = scene.tracks.get(pair.victim_id), scene.tracks.get(pair.bike_id)
            if victim is None or bike is None:
                self._drop(key)
                continue
            if pair.on_foot and pair.phase == "WATCH" and (bike.track_id in scene.rider_of or bike.cls is not ObjClass.PERSON):
                self._drop(key)          # the "pedestrian" attacker turned out to be riding / not a person
                continue
            if not pair.on_foot and not bike.cls.is_vehicle:
                self._drop(key)          # the "vehicle" was a rider+bike blob now voted 'person': not a valid pair
                continue
            if scene.rider_of.get(pair.victim_id) == pair.bike_id and pair.phase != "CONTACT":
                self._drop(key)          # turned out to be this bike's rider
                continue

            if pair.phase == "COOLDOWN" and t >= pair.cooldown_until:
                pair.phase, pair.phase_contact_start, pair.escape_t = "WATCH", None, None

            if victim.lost_at is None:
                sample = build_sample(scene, victim, bike, p)
                new_ep = pair.add(sample, p, keep_s)
            else:
                sample, new_ep = None, None

            if pair.phase == "WATCH":
                # the victim must still look like someone on foot just before the contact
                if new_ep is not None and victim.track_id not in scene.rider_of \
                        and scene.looks_on_foot(victim, new_ep.start - 0.1, p):
                    pair.phase = "CONTACT"
                    pair.phase_contact_start = new_ep.start
                    scene.locked_pedestrians.add(victim.track_id)
                    log.debug("pair %s contact at %.2f (level %d)", key, new_ep.start, new_ep.level)
                elif self._idle_expired(pair, sample, bike, t, p):
                    self._drop(key)
                continue

            if pair.phase == "CONTACT":
                ce = pair.episodes[-1].end
                if pair.escape_t is None:
                    escaped, _ = eval_escape(pair, bike, pair.phase_contact_start, ce, t, p)
                    if escaped:
                        pair.escape_t = t
                ready = (pair.escape_t is not None and t - pair.escape_t >= p.decision_hold_s) \
                    or (t - ce >= p.decision_timeout_s)
                if ready:
                    decision = self._decide(pair, victim, bike, t, p)
                    if decision is not None:
                        decisions.append(decision)
        return decisions

    def flush(self) -> list[Decision]:
        """End of stream: judge every pair still in CONTACT with what has been observed so far."""
        out = []
        for pair in list(self.pairs.values()):
            if pair.phase != "CONTACT":
                continue
            p = self.params_for(pair)
            victim, bike = self.scene.tracks.get(pair.victim_id), self.scene.tracks.get(pair.bike_id)
            if victim is None or bike is None:
                continue
            d = self._decide(pair, victim, bike, self.scene.t, p)
            if d is not None:
                out.append(d)
        return out

    def _decide(self, pair, victim, bike, t: float, p: RuleParams) -> Decision | None:
        verdict = evaluate(pair, victim, bike, t, p, self.scene.vehicles(include_lost=True))
        pair.phase = "COOLDOWN"
        pair.cooldown_until = t + p.cooldown_s
        self.scene.locked_pedestrians.discard(pair.victim_id)
        log.info("pair %s -> %s score=%.2f", pair.key, verdict.level.value, verdict.score)

        if verdict.level is not Level.DISMISSED:
            # one alert per attacker and per victim: bystander pairs and the mirrored on-foot pair stay quiet
            if self._bike_cooldown.get(pair.bike_id, -1.0) > t or self._victim_cooldown.get(pair.victim_id, -1.0) > t \
                    or (pair.on_foot and self._bike_cooldown.get(pair.victim_id, -1.0) > t):
                return None
            self._bike_cooldown[pair.bike_id] = t + p.cooldown_s
            self._victim_cooldown[pair.victim_id] = t + p.cooldown_s
        riders = [] if pair.on_foot else self.scene.riders(pair.bike_id)
        boxes = {"victim": [round(float(x), 1) for x in victim.last.box],
                 ("attacker" if pair.on_foot else "vehicle"): [round(float(x), 1) for x in bike.last.box]}
        for i, r in enumerate(riders):
            boxes[f"rider{i + 1}"] = [round(float(x), 1) for x in r.last.box]
        mode = "on_foot" if pair.on_foot else bike.cls.value
        return Decision(verdict, t, pair.victim_id, pair.bike_id, [r.track_id for r in riders], self._lighting, boxes,
                        mode)

    def _idle_expired(self, pair: PairState, sample, bike, t: float, p: RuleParams) -> bool:
        if bike.lost_at is not None:
            return t - bike.lost_at >= LOST_BIKE_DROP_S
        if sample is None:
            return False
        if sample.gap > p.pair_close_gap:
            pair.far_since = pair.far_since if pair.far_since is not None else t
            return t - pair.far_since >= p.pair_idle_timeout
        pair.far_since = None
        return False

    def _drop(self, key: tuple[int, int]) -> None:
        pair = self.pairs.pop(key, None)
        if pair is not None and pair.phase == "CONTACT":
            self.scene.locked_pedestrians.discard(pair.victim_id)
