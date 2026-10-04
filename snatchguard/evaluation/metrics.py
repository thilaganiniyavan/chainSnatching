"""Event-level scoring: does every real snatch get an alert, and is every alert a real snatch?

Definitions (the ones the >= 90 % targets are measured with):

* An **event** is one labelled snatch (one per snatch clip). Its window is
  [event_start - 1.5 s, event_end + 1.5 s], or the whole clip when no times were labelled.
* An **alert** is a decision at the chosen level ("any" = CONFIRMED or SUSPECT, i.e. anything that
  notifies the dashboard; "confirmed" = CONFIRMED only).
* An alert is a **true alert** when its contact interval overlaps an event window; otherwise it is a
  **false alarm** (this includes alerts at the wrong moment inside a snatch clip and every alert on
  normal footage).
* **Recall** = events with at least one true alert / events.
* **Precision** = true alerts / all alerts. Several alerts about the same event all count as true.
* 95 % confidence intervals use the Wilson score interval (honest for small counts).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

IN_SCOPE = ("snatch_bike", "snatch_foot")
WINDOW_PAD_S = 1.5


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def is_alert(decision: dict, level: str) -> bool:
    return decision["level"] == "CONFIRMED" or (level == "any" and decision["level"] == "SUSPECT")


def event_window(row: dict, duration_s: float) -> tuple[float, float]:
    if row.get("event_start_s"):
        start = float(row["event_start_s"])
        end = float(row["event_end_s"] or start)
        return (start - WINDOW_PAD_S, end + WINDOW_PAD_S)
    return (-1e9, 1e9)


def overlaps(decision: dict, window: tuple[float, float]) -> bool:
    return decision["contact_start"] <= window[1] and decision["contact_end"] >= window[0]


@dataclass
class VideoOutcome:
    video: str
    category: str
    fold: str
    lighting: str
    duration_s: float
    event: bool
    detected: bool = False
    true_alerts: int = 0
    false_alarms: int = 0
    latency_s: float | None = None
    alerts: list[dict] = field(default_factory=list)


@dataclass
class Metrics:
    level: str
    events: int
    events_detected: int
    true_alerts: int
    false_alarms: int
    normal_hours: float
    normal_false_alarms: int
    outcomes: list[VideoOutcome] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.events_detected / self.events if self.events else 0.0

    @property
    def precision(self) -> float:
        n = self.true_alerts + self.false_alarms
        return self.true_alerts / n if n else 1.0

    @property
    def f1(self) -> float:
        r, p = self.recall, self.precision
        return 2 * r * p / (r + p) if r + p else 0.0

    @property
    def recall_ci(self) -> tuple[float, float]:
        return wilson(self.events_detected, self.events)

    @property
    def precision_ci(self) -> tuple[float, float]:
        return wilson(self.true_alerts, self.true_alerts + self.false_alarms)

    @property
    def false_alarms_per_hour(self) -> float:
        return self.normal_false_alarms / self.normal_hours if self.normal_hours else 0.0

    def summary(self) -> dict:
        return {
            "level": self.level, "events": self.events, "events_detected": self.events_detected,
            "recall": round(self.recall, 4), "recall_ci95": [round(x, 3) for x in self.recall_ci],
            "true_alerts": self.true_alerts, "false_alarms": self.false_alarms,
            "precision": round(self.precision, 4), "precision_ci95": [round(x, 3) for x in self.precision_ci],
            "f1": round(self.f1, 4), "normal_hours": round(self.normal_hours, 2),
            "false_alarms_per_hour": round(self.false_alarms_per_hour, 3),
        }


def score(results: dict[str, tuple[dict, float, list[dict]]], level: str = "any",
          categories: tuple[str, ...] = IN_SCOPE) -> Metrics:
    """results: video -> (label row, analysed duration in s, decisions)."""
    m = Metrics(level, 0, 0, 0, 0, 0.0, 0)
    for video, (row, duration, decisions) in results.items():
        cat = row["category"]
        if cat not in categories and cat != "normal":
            continue
        alerts = [d for d in decisions if is_alert(d, level)]
        o = VideoOutcome(video, cat, row.get("fold", ""), row.get("lighting", ""), duration, cat != "normal",
                         alerts=alerts)
        if cat == "normal":
            o.false_alarms = len(alerts)
            m.normal_hours += duration / 3600
            m.normal_false_alarms += len(alerts)
        else:
            window = event_window(row, duration)
            hits = [a for a in alerts if overlaps(a, window)]
            o.true_alerts, o.false_alarms = len(hits), len(alerts) - len(hits)
            o.detected = bool(hits)
            if hits and row.get("event_start_s"):
                o.latency_s = round(min(a["decided_at"] for a in hits) - float(row["event_start_s"]), 2)
            m.events += 1
            m.events_detected += o.detected
        m.true_alerts += o.true_alerts
        m.false_alarms += o.false_alarms
        m.outcomes.append(o)
    return m


def merge(parts: list[Metrics]) -> Metrics:
    """Pool cross-validation folds into one estimate."""
    out = Metrics(parts[0].level, 0, 0, 0, 0, 0.0, 0)
    for p in parts:
        out.events += p.events
        out.events_detected += p.events_detected
        out.true_alerts += p.true_alerts
        out.false_alarms += p.false_alarms
        out.normal_hours += p.normal_hours
        out.normal_false_alarms += p.normal_false_alarms
        out.outcomes += p.outcomes
    return out
