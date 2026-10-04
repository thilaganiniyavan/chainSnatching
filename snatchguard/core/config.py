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
        self.raw = copy.deepcopy(raw)
        base = raw.get("base", {})
        profiles = raw.get("profiles", {})
        foot = raw.get("foot", {}) or {}
        self.profiles = {}
        self.foot_profiles = {}
        for light in Lighting:
            merged = _deep_merge(base, profiles.get(light.value) or {})
            self.profiles[light] = RuleParams(merged, light.value)
            self.foot_profiles[light] = RuleParams(_deep_merge(merged, foot), f"{light.value}+foot")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_RULES, overrides: dict | None = None) -> "RuleConfig":
        raw = load_yaml(path)
        if overrides:
            raw["base"] = _deep_merge(raw.get("base", {}), overrides)
        return cls(raw)

    def for_lighting(self, lighting: Lighting, foot: bool = False) -> RuleParams:
        return (self.foot_profiles if foot else self.profiles)[lighting]

    def with_overrides(self, base: dict | None = None, foot: dict | None = None) -> "RuleConfig":
        """A copy with some parameters changed (used by the tuner)."""
        raw = copy.deepcopy(self.raw)
        if base:
            raw["base"] = _deep_merge(raw.get("base", {}), base)
        if foot:
            raw["foot"] = _deep_merge(raw.get("foot", {}) or {}, foot)
        return RuleConfig(raw)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.raw, f, sort_keys=False)
