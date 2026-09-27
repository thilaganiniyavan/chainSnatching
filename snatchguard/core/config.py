"""YAML config loading. Rule thresholds get per-lighting profiles."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from snatchguard.core.types import Lighting

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES = REPO_ROOT / "configs" / "rules.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class RuleParams:
    """Attribute access over one merged rule profile (`p.neck_radius`)."""

    def __init__(self, values: dict[str, Any], name: str) -> None:
        self._values = values
        self.name = name

    def __getattr__(self, key: str) -> Any:
        try:
            return self._values[key]
        except KeyError as exc:
            raise AttributeError(f"unknown rule parameter {key!r}") from exc

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._values)


class RuleConfig:
    """Holds one RuleParams per lighting profile."""

    def __init__(self, raw: dict[str, Any]) -> None:
        base = raw.get("base", {})
        profiles = raw.get("profiles", {})
        self.profiles = {
            light: RuleParams(_deep_merge(base, profiles.get(light.value) or {}), light.value)
            for light in Lighting
        }

    @classmethod
    def load(cls, path: str | Path = DEFAULT_RULES, overrides: dict | None = None) -> "RuleConfig":
        raw = load_yaml(path)
        if overrides:
            raw["base"] = _deep_merge(raw.get("base", {}), overrides)
        return cls(raw)

    def for_lighting(self, lighting: Lighting) -> RuleParams:
        return self.profiles[lighting]
