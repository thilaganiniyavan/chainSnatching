"""Evidence evaluation for one pair once its contact phase is ready to be judged.

Four groups of evidence are scored: CONTACT, ESCAPE, REACTION, APPROACH.
Benign indicators (familiarity, co-travel, long or repeated contact, the
bike lingering, victim waving or calm) are subtracted. Both lists go out with
the alert so an operator can see exactly why it fired.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from snatchguard.core import geometry as g
from snatchguard.core.config import RuleParams
from snatchguard.core.types import Level
from snatchguard.rules.pair import CONTACT_POSE, PairSample, PairState
from snatchguard.rules.scene import Track


@dataclass
class Evidence:
    key: str
    group: str                 # contact | escape | reaction | approach | benign
    passed: bool
    detail: str
    value: float | None = None

    def as_dict(self) -> dict:
        return {"key": self.key, "group": self.group, "passed": self.passed,
                "detail": self.detail, "value": None if self.value is None else round(self.value, 3)}


@dataclass
class Verdict:
    level: Level
    score: float
    contact_start: float
    contact_end: float
    evidence: list[Evidence] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)
    confidence: str = "high"   # high | medium | low (how much the pose model could see)


def _dts(samples: list[PairSample]) -> list[float]:
    """Time each sample represents (gap to the next one, capped)."""
    out = []
    for a, b in zip(samples, samples[1:]):
        out.append(min(b.t - a.t, 0.5))
    if samples:
        out.append(min(out[-1] if out else 0.1, 0.5))
    return out


def _sustained(samples: list[PairSample], cond, min_s: float) -> bool:
    """True if `cond` holds on consecutive samples spanning at least `min_s`."""
    run_start = None
    for s in samples:
        if cond(s):
            run_start = s.t if run_start is None else run_start
            if s.t - run_start >= min_s:
                return True
        else:
            run_start = None
    return False


# ---------------------------------------------------------------------------
# Evidence groups
# ---------------------------------------------------------------------------

def is_crowded(pair: PairState, cs: float, ce: float, p: RuleParams) -> tuple[bool, float]:
    counts = sorted(s.crowd for s in pair.between(cs - 1.0, ce))
    med = counts[len(counts) // 2] if counts else 0
    return med >= p.crowd_people, med


def eval_escape(pair: PairState, bike: Track, cs: float, ce: float, now: float, p: RuleParams) -> tuple[bool, list[Evidence]]:
    during = pair.between(cs - 0.3, ce)
    after = [s for s in pair.between(ce, ce + p.escape_window_s) if s.t > ce and not s.bike_lost]
    ev: list[Evidence] = []
    if not during:
        return False, ev
    # speed at the grab itself (start of contact), not averaged over a long overlap that may already
    # include the getaway (an attacker on foot passing right through the victim)
    at_grab = pair.between(cs - 0.3, min(ce, cs + 0.5)) or during
    sac = float(np.mean([s.bike_speed for s in at_grab]))
    thr = max(p.escape_speedup_ratio * sac, sac + p.escape_speedup_abs)

    def gap_rate_ok(s: PairSample) -> bool:
        ref = during[-1]
        dt = s.t - ref.t
        return dt > 0 and (s.gap - ref.gap) / dt >= p.escape_gap_rate

    speedup = _sustained(after, lambda s: s.bike_speed >= thr and gap_rate_ok(s), 0.3)
    fast_departure = sac < p.escape_slow_contact_speed and _sustained(
        after, lambda s: s.bike_speed >= p.escape_fast_departure and gap_rate_ok(s), 0.3)
    peak = max((s.bike_speed for s in after), default=0.0)
    # Leaving the frame is only a getaway if the bike was fast on the way out; in slow, crowded
    # traffic tracks drop out at the edges all the time.
    exit_speed = float(np.linalg.norm(bike.velocity(bike.last_t))) / during[-1].H
    left_scene = (
        bike.lost_at is not None and bike.lost_near_border and ce < bike.lost_at <= ce + p.escape_window_s
        and (not after or after[-1].gap >= during[-1].gap)
        and max(exit_speed, peak) >= max(thr, p.escape_fast_departure)
    )
    ev.append(Evidence("bike_speedup", "escape", speedup,
                       f"bike speed {sac:.2f} -> {peak:.2f} H/s after contact (needs >= {thr:.2f})", peak))
    ev.append(Evidence("fast_departure", "escape", fast_departure,
                       f"bike slowed to {sac:.2f} H/s at contact, then left fast"))
    # Walking out of the view is not a getaway on foot, nor in a crowd (with a close-up camera everyone leaves
    # the frame within seconds): there the attacker must visibly accelerate away.
    crowded, _ = is_crowded(pair, cs, ce, p)
    if not p.left_scene_is_escape or crowded:
        left_scene = False
    ev.append(Evidence("left_scene", "escape", left_scene, "bike left the frame right after contact"))
    return speedup or fast_departure or left_scene, ev


def eval_reaction(pair: PairState, victim: Track, cs: float, ce: float, p: RuleParams) -> tuple[float, bool, list[Evidence]]:
    window = pair.between(cs, cs + p.reaction_window_s)
    before = pair.between(cs - p.hands_to_neck_baseline_s, cs - 0.05)
    strengths = p.reaction_strength
    ev: list[Evidence] = []
    found: dict[str, bool] = {}

    # hands go to the neck after the contact (and weren't already there, e.g. a phone call)
    already = any(s.hands_at_neck for s in before)
    flagged = [s for s in window if s.hands_at_neck]
    dur = sum(dt for s, dt in zip(window, _dts(window)) if s.hands_at_neck)
    found["hands_to_neck"] = (not already) and len(flagged) >= 2 and dur >= p.hands_to_neck_min_s
    ev.append(Evidence("hands_to_neck", "reaction", found["hands_to_neck"],
                       "victim's hands went to the neck after contact"
                       + (" (ignored: hands were already there before)" if already else ""), dur))

    # fall / stumble: the box collapses from the top while the feet stay on the ground.
    # (A vehicle hiding the legs shrinks the box from the bottom instead: not a fall.)
    base_obs = victim.window(cs - 2.0, cs)
    base_h = float(np.median([g.height(o.box) for o in base_obs])) if base_obs else None
    base_foot = float(np.median([o.box[3] for o in base_obs])) if base_obs else None
    fell = base_h is not None and any(
        g.height(o.box) < p.fall_height_ratio * base_h
        and g.width(o.box) / max(g.height(o.box), 1.0) > p.fall_aspect
        and o.box[3] >= base_foot - 0.15 * base_h
        for o in victim.window(cs, cs + p.reaction_window_s))
    found["fall"] = fell
    ev.append(Evidence("fall", "reaction", fell, "victim fell or crouched after contact"))

    # jerk: sudden change of the victim's velocity across the contact
    v_before = victim.velocity(cs, span=0.8)
    v_after = victim.velocity(min(cs + 0.8, victim.last_t), span=0.6)
    H = window[0].H if window else 1.0
    dv = float(np.linalg.norm(v_after - v_before)) / H
    found["jerk"] = dv >= p.jerk_dv
    ev.append(Evidence("jerk", "reaction", found["jerk"], f"victim velocity changed by {dv:.2f} H/s at contact", dv))

    # chase: victim breaks into a run in the bike's direction of travel
    pre_speed = np.mean([s.victim_speed for s in pair.between(cs - 1.5, cs)] or [0.0])

    def running_after(s: PairSample) -> bool:
        heading = s.bike_vel if s.bike_speed > 0.5 else s.away_dir
        return s.victim_speed >= p.chase_speed and g.cos_sim(s.victim_vel, heading) >= p.chase_cos

    chase = pre_speed < p.pre_contact_walk_speed and _sustained(
        pair.between(cs, cs + p.reaction_window_s + 1.0), running_after, p.chase_min_s)
    found["chase"] = chase
    ev.append(Evidence("chase", "reaction", chase, "victim ran after the bike"))

    strength = max((strengths[k] for k, v in found.items() if v), default=0.0)
    calm = strength == 0.0 and dv < 0.3 and any(s.victim_pose for s in window)
    # strong reactions are specific to being robbed; a 'jolt' alone is too often tracking noise in crowds
    strong = found["hands_to_neck"] or found["fall"] or found["chase"]
    return strength, calm, ev, strong


def _gap_at(victim: Track, bike: Track, t: float) -> float | None:
    vo, bo = victim.at(t), bike.at(t)
    if vo is None or bo is None or t - vo.t > 0.5 or t - bo.t > 0.5:
        return None
    b = bo.box.copy()
    b[1] -= 0.9 * g.height(b)            # include where the riders sit
    return g.box_gap(vo.box, b)


def eval_approach(pair: PairState, victim: Track, bike: Track, cs: float, p: RuleParams) -> tuple[float, bool, list[Evidence]]:
    """Uses the full track histories, so it also sees the approach from before the pair was opened."""
    ev: list[Evidence] = []
    parts = p.approach_strength
    strength = 0.0
    H = max(victim.ref_height(cs), 1.0)

    times = [o.t for o in bike.obs if cs - p.approach_window_s <= o.t <= cs]
    gaps = [(t, _gap_at(victim, bike, t)) for t in times]
    gaps = [(t, x / H) for t, x in gaps if x is not None]
    start = next(((t, x) for t, x in gaps if x >= p.approach_min_start_gap), None)
    closing = False
    if start is not None and gaps:
        end_t, end_gap = gaps[-1]
        dt = end_t - start[0]
        rate = (start[1] - end_gap) / dt if dt > 0.3 else 0.0
        closing = rate >= p.approach_closing_speed
        ev.append(Evidence("closing", "approach", closing, f"gap closed at {rate:.2f} H/s", rate))
    else:
        ev.append(Evidence("closing", "approach", False, "bike was already close when first seen"))
    if closing:
        strength += parts["closing"]

    first_t = bike.obs[0].t if bike.obs else cs

    def speeds(t0: float, t1: float) -> list[float]:
        # velocity is undefined for a track's first few observations: skip them
        return [float(np.linalg.norm(bike.velocity(o.t))) / H for o in bike.obs
                if t0 <= o.t <= t1 and o.t - first_t >= 0.3]

    near = speeds(cs - 1.0, cs + 0.3)
    earlier = speeds(max(cs - 5.0, first_t), cs - 1.0)
    if near and earlier:
        decel = max(earlier) > 0.5 and min(near) < p.decel_ratio * max(earlier)
        detail = f"bike slowed from {max(earlier):.2f} to {min(near):.2f} H/s next to the victim"
    else:
        decel = False
        detail = "bike speed before contact unknown (entered the view just before)"
    ev.append(Evidence("decelerated", "approach", decel, detail))
    if decel:
        strength += parts["decel"]

    vb = bike.velocity(cs) / H
    vv = victim.velocity(cs) / H
    behind = float(np.linalg.norm(vv)) > 0.3 and g.cos_sim(vb, vv) >= p.from_behind_cos
    ev.append(Evidence("from_behind", "approach", behind, "bike came from behind, same direction as victim"))
    if behind:
        strength += parts["from_behind"]
    return min(strength, 1.0), decel, ev


def _in_traffic_fraction(victim: Track, vehicles: list[Track], cs: float, p: RuleParams) -> float:
    """Share of the seconds before contact the 'victim' spent overlapping vehicles.
    Riders queued at a signal (their own scooter missed by the detector) sit among vehicles
    all the time; a pedestrian who gets targeted does not."""
    obs = victim.window(cs - p.in_traffic_window_s, cs - 0.3)
    if len(obs) < 3:
        return 0.0
    hits = 0
    for o in obs:
        area = max(g.area(o.box), 1.0)
        for veh in vehicles:
            vo = veh.at(o.t)
            if vo is not None and o.t - vo.t <= 0.3 and g.intersection(o.box, vo.box) / area >= 0.2:
                hits += 1
                break
    return hits / len(obs)


def eval_benign(pair: PairState, cs: float, ce: float, now: float, p: RuleParams,
                victim: Track | None = None, vehicles: list[Track] | None = None) -> tuple[dict[str, bool], list[Evidence]]:
    hist = pair.between(cs - p.history_window_s, cs)
    dts = _dts(hist)
    dwell = sum(dt for s, dt in zip(hist, dts) if s.gap <= p.dwell_gap and s.bike_speed <= p.dwell_bike_speed)
    cotravel = sum(dt for s, dt in zip(hist, dts)
                   if s.gap <= p.cotravel_gap and s.victim_speed > 0.3
                   and float(np.linalg.norm(s.bike_vel - s.victim_vel)) <= p.cotravel_dv)
    episodes = [e for e in pair.episodes if e.start >= cs - p.history_window_s]
    # Contact flicker (noise, dropped frames) inside one pass is one encounter; only a bike that
    # pulls away and comes back again counts as a repeat.
    encounters = 0
    last_end = None
    for e in episodes:
        if last_end is None or e.start - last_end >= p.repeat_min_separation_s:
            encounters += 1
        last_end = e.end
    # Sustained contact = hand at the neck for long, or boxes overlapping while the bike is stopped.
    # Overlap while riding alongside for a moment (a snatch) is not "long".
    during = pair.between(cs, ce)
    ddt = _dts(during)
    pose_time = sum(dt for s, dt in zip(during, ddt) if s.contact == CONTACT_POSE)
    still_time = sum(dt for s, dt in zip(during, ddt) if s.contact and s.bike_speed <= p.dwell_bike_speed)
    longest = max(pose_time, still_time)
    after = [s for s in pair.between(ce, now) if s.t > ce]
    linger = sum(dt for s, dt in zip(after, _dts(after))
                 if not s.bike_lost and s.gap <= p.linger_gap and s.bike_speed <= p.dwell_bike_speed)
    wave = any(s.waving for s in pair.between(cs, now))
    traffic = _in_traffic_fraction(victim, vehicles, cs, p) if victim is not None and vehicles else 0.0

    flags = {
        "dwell": dwell >= p.dwell_min_s,
        "cotravel": cotravel >= p.cotravel_min_s,
        "long_contact": longest >= p.long_contact_s,
        "repeated_contact": encounters >= p.repeated_contacts,
        "linger": linger >= p.linger_s,
        "wave": wave,
        "in_traffic": traffic >= p.in_traffic_frac,
    }
    ev = [
        Evidence("dwell", "benign", flags["dwell"], f"stood together {dwell:.1f}s before contact (familiar)", dwell),
        Evidence("cotravel", "benign", flags["cotravel"], f"moved together {cotravel:.1f}s before contact", cotravel),
        Evidence("long_contact", "benign", flags["long_contact"], f"contact lasted {longest:.1f}s (hug / handover)", longest),
        Evidence("repeated_contact", "benign", flags["repeated_contact"], f"{encounters} separate contact encounters (play)", encounters),
        Evidence("linger", "benign", flags["linger"], f"bike stayed near victim {linger:.1f}s after contact", linger),
        Evidence("wave", "benign", flags["wave"], "victim waved"),
        Evidence("in_traffic", "benign", flags["in_traffic"],
                 f"'victim' was among vehicles {traffic:.0%} of the time before (likely a rider in traffic)", traffic),
    ]
    return flags, ev


# ---------------------------------------------------------------------------

def evaluate(pair: PairState, victim: Track, bike: Track, now: float, p: RuleParams,
             vehicles: list[Track] | None = None) -> Verdict:
    eps = [e for e in pair.episodes if pair.phase_contact_start is not None and e.start >= pair.phase_contact_start - 0.01]
    cs = eps[0].start
    ce = eps[-1].end
    contact_level = max(e.level for e in eps)
    w = p.weights

    evidence: list[Evidence] = []
    min_wn = min((e.min_wrist_neck for e in eps if e.min_wrist_neck is not None), default=None)
    evidence.append(Evidence(
        "contact", "contact", True,
        "rider's hand reached the victim's neck" if contact_level == CONTACT_POSE
        else "rider/bike overlapped the victim's upper body (box only, no pose)",
        min_wn))

    escaped, ev = eval_escape(pair, bike, cs, ce, now, p)
    evidence += ev
    reaction, calm, ev, strong_reaction = eval_reaction(pair, victim, cs, ce, p)
    evidence += ev
    approach, decel, ev = eval_approach(pair, victim, bike, cs, p)
    closing = any(e.key == "closing" and e.passed for e in ev)
    evidence += ev
    benign, ev = eval_benign(pair, cs, ce, now, p, victim, [v for v in (vehicles or []) if v.track_id != bike.track_id])
    evidence += ev
    evidence.append(Evidence("calm", "benign", calm, "victim carried on calmly, no reaction"))

    components = {
        "contact": w["contact"] * (1.0 if contact_level == CONTACT_POSE else p.contact_box_factor),
        "escape": w["escape"] * (1.0 if escaped else 0.0),
        "reaction": w["reaction"] * reaction,
        "approach": w["approach"] * approach,
    }
    penalties = {k: p.penalties[k] for k, v in benign.items() if v}
    if calm:
        penalties["calm"] = p.penalties["calm"]
    score = max(0.0, min(1.0, sum(components.values()) - sum(penalties.values())))

    # Contact + getaway is not enough on its own: in crowds, boxes overlap and people leave the frame all the
    # time. Every alert needs one strong, independent cue: the hand at the neck seen by pose, a specific victim
    # reaction (hands to neck, fall, chase), or the attacker clearly closing in and slowing beside the victim.
    crowded, crowd_n = is_crowded(pair, cs, ce, p)
    evidence.append(Evidence("crowded", "benign", crowded,
                             f"{crowd_n:.0f} other people within reach of the victim: busy scene, stricter evidence",
                             crowd_n))
    if crowded:
        # slowing down and passing close to people is normal in a crowd: approach no longer corroborates,
        # and the hand at the neck alone isn't enough (people overlap in the image): a reaction is needed too
        strong = strong_reaction
    else:
        strong = contact_level == CONTACT_POSE or strong_reaction or (closing and decel)
    corroborated = strong
    if p.require_pose_contact_for_confirm:
        corroborated = contact_level == CONTACT_POSE and (strong_reaction or approach > 0)
    suspect_ok = escaped and strong
    if p.suspect_requires_pose_or_reaction:
        suspect_ok = escaped and (contact_level == CONTACT_POSE or strong_reaction)
    if calm and p.calm_vetoes:
        suspect_ok = corroborated = False   # on foot: a visibly calm 'victim' means nothing was taken
    if escaped and not benign["linger"] and not benign["in_traffic"] and corroborated and score >= p.confirm_threshold:
        level = Level.CONFIRMED
    elif score >= p.suspect_threshold and suspect_ok:
        level = Level.SUSPECT
    else:
        level = Level.DISMISSED

    if pair.on_foot:                      # same evidence, worded for a snatcher on foot
        for e in evidence:
            e.detail = e.detail.replace("rider's hand", "attacker's hand").replace("rider/bike", "attacker") \
                .replace("bike", "attacker")
    pose_seen = sum(1 for s in pair.between(cs - 1.0, ce + 1.0) if s.victim_pose)
    confidence = "high" if contact_level == CONTACT_POSE else ("medium" if pose_seen else "low")
    return Verdict(level, round(score, 3), cs, ce, evidence, components, penalties, confidence)
