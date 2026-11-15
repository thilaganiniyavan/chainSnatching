"""Second-opinion review of snatch alerts by a video-capable model (Google Gemini).

Enabled when GEMINI_API_KEY (or GOOGLE_API_KEY) is set; SNATCHGUARD_AI_REVIEW=0 turns it off. Each candidate snatch
alert is held back and its trimmed detected video (or, failing that, its snapshot) is sent to the model:
  confirmed    -> the alert is published in the dashboard
  rejected     -> the alert stays hidden (it is kept in the database for audit)
  unavailable  -> the review failed (network, quota, blocked): the alert is published WITHOUT the badge, so an
                  outage never silences an alert.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# tried in order: the next one is used when a model is overloaded, times out or is unavailable
MODELS = [m.strip() for m in os.environ.get("SNATCHGUARD_REVIEW_MODELS", "gemini-3.6-flash,gemini-3.5-flash").split(",") if m.strip()]
MIN_CONFIDENCE = float(os.environ.get("SNATCHGUARD_REVIEW_MIN_CONFIDENCE", "0.6"))
TIMEOUT_S = float(os.environ.get("SNATCHGUARD_REVIEW_TIMEOUT_S", "45"))
VIDEO_FPS = float(os.environ.get("SNATCHGUARD_REVIEW_FPS", "5"))     # a grab lasts 1-2 s; the default 1 fps misses it
MIME = {".mp4": "video/mp4", ".webm": "video/webm", ".avi": "video/x-msvideo", ".mov": "video/quicktime",
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

PROMPT = """This is CCTV footage that an automatic detector flagged as a possible snatch theft{at}.

Decide whether it shows a snatch: one person suddenly grabbing or pulling something (a chain or necklace, a bag, a
phone, jewellery) from another person who does not consent, often followed by the thief running or riding away.

Not a snatch: people walking past or close to each other, handshakes, hugs, handing over an object, playing,
people adjusting their own belongings, crowds, or traffic.

Be strict: answer snatch=true only if the footage shows the grab or its unmistakable aftermath (victim reaching after
the thief, thief fleeing with the item). Give confidence 0-1 and one short sentence of reason."""

SCHEMA = {
    "type": "object",
    "properties": {"snatch": {"type": "boolean"}, "confidence": {"type": "number"}, "reason": {"type": "string"}},
    "required": ["snatch", "confidence", "reason"],
}


def enabled() -> bool:
    if os.environ.get("SNATCHGUARD_AI_REVIEW", "").lower() in ("0", "false", "off"):
        return False
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


class AIReviewer:
    def __init__(self, client=None):
        if client is None:
            from google import genai
            from google.genai import types
            client = genai.Client(http_options=types.HttpOptions(timeout=int(TIMEOUT_S * 1000)))
        self.client = client

    def review(self, path: Path, event_at_s: float | None = None) -> dict:
        """Review a video (.mp4/.webm/...) or a snapshot image. Never raises: failures return state 'unavailable'."""
        try:
            from google.genai import types
            path = Path(path)
            mime = MIME.get(path.suffix.lower())
            if mime is None:
                raise ValueError(f"unsupported file {path.name}")
            data = path.read_bytes()
            if not data:
                raise ValueError(f"empty file {path.name}")
            if mime.startswith("video/"):
                media = types.Part(inline_data=types.Blob(data=data, mime_type=mime),
                                   video_metadata=types.VideoMetadata(fps=VIDEO_FPS))
                at = f", around {event_at_s:.0f} s into this video" if event_at_s is not None else ""
            else:
                media = types.Part.from_bytes(data=data, mime_type=mime)
                at = " (one still image from the camera)"
            config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=SCHEMA,
                                                 temperature=0)
            resp, model, last = None, None, None
            for model in MODELS:
                try:
                    resp = self.client.models.generate_content(model=model, contents=[media, PROMPT.format(at=at)],
                                                               config=config)
                    break
                except Exception as exc:  # noqa: BLE001 - overloaded / timed out / not available: try the next
                    last = exc
                    log.warning("AI review: %s failed (%s), trying the next model", model, str(exc)[:120])
            if resp is None:
                raise last or RuntimeError("no review model configured")
            if not resp.text:
                return {"state": "unavailable", "error": "empty or blocked response", "model": model}
            verdict = json.loads(resp.text)
            ok = bool(verdict["snatch"]) and float(verdict["confidence"]) >= MIN_CONFIDENCE
            return {"state": "confirmed" if ok else "rejected", "snatch": bool(verdict["snatch"]),
                    "confidence": float(verdict["confidence"]), "reason": str(verdict["reason"])[:300],
                    "model": model}
        except Exception as exc:  # noqa: BLE001 - any failure must fall back to showing the alert
            log.warning("AI review failed for %s: %s", path, exc)
            return {"state": "unavailable", "error": type(exc).__name__}
