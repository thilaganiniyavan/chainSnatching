"""Stage 1: every new close contact between two tracked objects, merged into clips to verify.

The rule engine decides who is a rider, a pedestrian and a possible victim before it pairs anyone. On real
CCTV that role logic picked the wrong people in most judged events, so this stage uses no roles at all: a
candidate is any pair (at least one a person) whose boxes come within CLOSE body-heights after having been
apart. Riders on their own vehicle are skipped. Candidates that start close together in time and space are
merged into one clip, because in a crowd one grab produces several pair contacts at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from snatchguard.core import geometry as g
from snatchguard.core.types import Detection, ObjClass

CLOSE = 0.25        # gap, in body heights, that counts as contact
OPEN = 0.6          # gap that ends a contact episode (hysteresis against jitter)
RIDER_IOU = 0.3     # a person box on a vehicle box is its rider, not a pair
MIN_SEP_S = 1.5     # a pair must be out of contact this long before it can start a new candidate
ABSENT_S = 0.5      # a pair not seen together this long is out of contact (the detector lost one of them)
MIN_PERSON_PX = 24  # people shorter than this are too small to judge
CROP_SCALE = 2.0    # clip crop side = CROP_SCALE x the larger side of the actors' union box
CROP_MIN = 96.0     # ... but at least this many pixels
MERGE_S = 1.0       # candidates starting within this time ...
MERGE_IOU = 0.3     # ... with crops overlapping this much become one clip
STALE_S = 10.0      # forget pairs not seen for this long


@dataclass
class Contact:
    t: float
    a: int
    b: int


@dataclass
class ClipCandidate:
    """A clip to verify: onset time, square crop (x0, y0, side) in frame pixels, and the pairs it covers."""

    t: float
    crop: tuple[float, float, float]
    pairs: list[tuple[int, int]] = field(default_factory=list)


def crop_for(boxes: list[np.ndarray]) -> tuple[float, float, float]:
    bx = np.asarray(boxes, dtype=float)
    x1, y1, x2, y2 = bx[:, 0].min(), bx[:, 1].min(), bx[:, 2].max(), bx[:, 3].max()
    side = max(max(x2 - x1, y2 - y1) * CROP_SCALE, CROP_MIN)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return (float(cx - side / 2), float(cy - side / 2), float(side))


def square_iou(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[2], b[1] + b[2]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[2] + b[2] * b[2] - inter + 1e-9)


class ContactDetector:
    """Feed detections in time order; returns the contacts that start at this frame."""

    def __init__(self) -> None:
        self._state: dict[tuple[int, int], list] = {}   # pair -> [in_contact, last_contact_t, last_seen_t]

    def step(self, t: float, dets: list[Detection]) -> list[Contact]:
        out: list[Contact] = []
        n = len(dets)
        for i in range(n):
            d1 = dets[i]
            for j in range(i + 1, n):
                d2 = dets[j]
                v1, v2 = d1.cls.is_vehicle, d2.cls.is_vehicle
                if v1 and v2:
                    continue
                if v1 != v2 and g.iou(d1.box, d2.box) > RIDER_IOU:
                    continue
                H = max(g.height(d.box) for d in (d1, d2) if d.cls is ObjClass.PERSON)
                if H < MIN_PERSON_PX:
                    continue
                gap = g.box_gap(d1.box, d2.box) / H
                key = (d1.track_id, d2.track_id) if d1.track_id < d2.track_id else (d2.track_id, d1.track_id)
                st = self._state.get(key)
                if st is None:
                    st = self._state[key] = [False, -1e9, t]
                if t - st[2] > ABSENT_S:
                    st[0] = False   # else a grab after one of them was lost continues the earlier contact
                st[2] = t
                if gap <= CLOSE:
                    if not st[0] and t - st[1] >= MIN_SEP_S:
                        out.append(Contact(t, key[0], key[1]))
                    st[0], st[1] = True, t
                elif gap > OPEN:
                    st[0] = False
        if len(self._state) > 4000:
            self._state = {k: v for k, v in self._state.items() if t - v[2] < STALE_S}
        return out


class CandidateStage:
    """Contacts merged into clip candidates. `step` returns the clips that start at this frame."""

    def __init__(self) -> None:
        self.contacts = ContactDetector()
        self._open: list[ClipCandidate] = []

    def step(self, t: float, dets: list[Detection]) -> list[ClipCandidate]:
        self._open = [c for c in self._open if t - c.t <= MERGE_S]
        boxes = {d.track_id: d.box for d in dets}
        new: list[ClipCandidate] = []
        for c in self.contacts.step(t, dets):
            crop = crop_for([boxes[c.a], boxes[c.b]])
            for clip in self._open:
                if square_iou(clip.crop, crop) > MERGE_IOU:
                    clip.pairs.append((c.a, c.b))
                    break
            else:
                clip = ClipCandidate(t, crop, [(c.a, c.b)])
                self._open.append(clip)
                new.append(clip)
        return new
