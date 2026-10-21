"""Fusion of rule decisions and learned window scores: streaming logic, offline replay, CV splits, fallback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snatchguard.evaluation import fusion as fz
from snatchguard.evaluation.fusion import OFF, Fuser, FusionParams, fuse_video


def _dec(level, cs, ce=None, at=None, score=0.6):
    ce = cs + 0.5 if ce is None else ce
    return {"level": level, "score": score, "mode": "motorcycle", "contact_start": cs, "contact_end": ce,
            "decided_at": ce + 1.0 if at is None else at}


def _wins(scores, t0=0.0, stride=0.5, length=2.0):
    return [(t0 + i * stride, t0 + i * stride + length, s) for i, s in enumerate(scores)]


def test_rules_only_passes_decisions_through_unchanged():
    decs = [_dec("CONFIRMED", 2), _dec("SUSPECT", 10), _dec("DISMISSED", 20)]
    out = fuse_video(decs, None, FusionParams(suspect_gate=0.9, rescue_threshold=0.1, model_threshold=0.1))
    assert [a["level"] for a in out] == ["CONFIRMED", "SUSPECT"]
    assert all(a["source"] == "rules" and a["model_score"] is None for a in out)
    assert [a["contact_start"] for a in out] == [2, 10]


def test_suspect_gate_keeps_or_drops_by_model_score():
    p = FusionParams(suspect_gate=0.5)
    low = _wins([0.1] * 40)
    high = _wins([0.1] * 15 + [0.8] * 10 + [0.1] * 15)
    assert fuse_video([_dec("SUSPECT", 8)], low, p) == []
    out = fuse_video([_dec("SUSPECT", 8)], high, p)
    assert len(out) == 1 and out[0]["level"] == "SUSPECT" and out[0]["source"] == "fused"
    assert out[0]["model_score"] == pytest.approx(0.8)


def test_upgrade_and_rescue():
    w = _wins([0.1] * 15 + [0.95] * 6 + [0.1] * 20)
    up = fuse_video([_dec("SUSPECT", 8)], w, FusionParams(upgrade_threshold=0.9))
    assert up[0]["level"] == "CONFIRMED" and up[0]["rule_level"] == "SUSPECT"
    rescued = fuse_video([_dec("DISMISSED", 8)], w, FusionParams(rescue_threshold=0.9))
    assert rescued[0]["level"] == "SUSPECT" and rescued[0]["source"] == "fused"
    assert fuse_video([_dec("DISMISSED", 8)], w, FusionParams(rescue_threshold=OFF)) == []


def test_model_only_needs_k_windows_and_respects_cooldown():
    p = FusionParams(model_threshold=0.7, model_k=3, cooldown_s=5.0, use_rules=False)
    blip = _wins([0.1] * 10 + [0.9, 0.9] + [0.1] * 10)
    assert fuse_video([], blip, p, use_rules=False) == []
    two_runs = _wins([0.1] * 10 + [0.8, 0.9, 0.85] + [0.1] * 4 + [0.9] * 4 + [0.1] * 40 + [0.9] * 3)
    out = fuse_video([], two_runs, p, use_rules=False)
    assert len(out) == 2                                  # second run within the cooldown is merged
    first = out[0]
    assert first["source"] == "model" and first["model_score"] == pytest.approx(0.9)
    assert (first["contact_start"], first["contact_end"]) == (5.5, 7.5)   # the peak window
    assert out[1]["contact_start"] > first["contact_end"] + 5.0


def test_rule_upgrade_after_model_alert_but_no_duplicate_suspect():
    w = _wins([0.1] * 10 + [0.9] * 3 + [0.1] * 30)
    p = FusionParams(model_threshold=0.7, model_k=2, cooldown_s=8.0)
    out = fuse_video([_dec("SUSPECT", 6, at=12)], w, p)
    assert [a["source"] for a in out] == ["model"]        # rule SUSPECT merged into the model alert
    out = fuse_video([_dec("CONFIRMED", 6, at=12)], w, p)
    assert [a["level"] for a in out] == ["SUSPECT", "CONFIRMED"]
    assert out[1]["upgrade_of"] == out[0]["fusion_id"]


def test_rule_vs_rule_duplicates_are_left_to_the_rule_engine():
    out = fuse_video([_dec("SUSPECT", 5), _dec("SUSPECT", 7)], None, FusionParams(cooldown_s=30))
    assert len(out) == 2


def test_live_hold_waits_for_model_then_times_out():
    f = Fuser(FusionParams(suspect_gate=0.5, hold_max_s=3.0))
    f.add_windows(_wins([0.2] * 4))                       # covers up to 3.5 s
    f.add_rule(_dec("SUSPECT", 4.0, 4.5, at=5.0))         # needs coverage to 6.0
    assert f.poll(5.0) == []                              # waiting for the model
    f.add_windows([(4.0, 6.0, 0.8)])
    out = f.poll(6.1)
    assert len(out) == 1 and out[0].payload["model_score"] == pytest.approx(0.8)
    f.add_rule(_dec("SUSPECT", 20.0, 20.5, at=21.0))      # the model stalls: no windows near it
    assert f.poll(22.0) == []
    late = f.poll(24.5)                                   # hold_max_s passed -> resolved without model there
    assert len(late) == 1 and late[0].payload["model_score"] is None and late[0].payload["source"] == "rules"


def test_ref_is_carried_and_flush_resolves_pending():
    f = Fuser(FusionParams())
    f.add_windows(_wins([0.2] * 3))
    marker = object()
    f.add_rule(_dec("CONFIRMED", 9), ref=marker)
    out = f.flush()
    assert out[0].ref is marker and out[0].payload["level"] == "CONFIRMED"


def test_params_round_trip(tmp_path):
    p = FusionParams(suspect_gate=0.3, model_k=3.4)
    path = tmp_path / "f.yaml"
    path.write_text("params:\n" + "".join(f"  {k}: {v}\n" for k, v in p.to_dict().items()))
    q = FusionParams.load(path)
    assert q.suspect_gate == pytest.approx(0.3) and q.model_k == 3


# ---- offline tuning: CV splits and the rules-only fallback -------------------------------------------

def _fake_dataset(tmp_path: Path, with_model: bool):
    rows = []
    for fold in ("A", "B"):
        for i in range(3):
            rows.append({"video": f"Snatch Theft/{fold}{i}.mp4", "category": "snatch_bike", "lighting": "day",
                         "event_start_s": "5.0", "event_end_s": "6.0", "fold": fold})
        rows.append({"video": f"Normal/{fold}n.avi", "category": "normal", "lighting": "day",
                     "event_start_s": "", "event_end_s": "", "fold": fold})
    items = [fz.Item(r, str(tmp_path / r["video"]), tmp_path / (r["video"].replace("/", "__") + ".pkl.gz"))
             for r in rows]
    if with_model:
        d = tmp_path / "oof" / "snatch_ab"
        d.mkdir(parents=True)
        for it in items:
            snatch = it.row["category"] != "normal"
            ws = _wins([0.1] * 8 + ([0.95] * 4 if snatch else [0.1] * 4) + [0.1] * 40)
            (d / f"{fz.video_stem(it.video_path)}.json").write_text(json.dumps({"windows": ws}))
    return items


def test_cv_tunes_on_one_fold_and_scores_the_other(tmp_path, monkeypatch):
    items = _fake_dataset(tmp_path, with_model=True)
    seen = []
    monkeypatch.setattr(fz, "replay_decisions", lambda cf, rules: (30.0, []))
    real_search = fz.search

    def spy(prepared, variant, *a, **kw):
        seen.append((variant, sorted({p.item.row["fold"] for p in prepared})))
        return real_search(prepared, variant, *a, **kw)

    monkeypatch.setattr(fz, "search", spy)
    rules = object()
    res = fz.run(items, {"A": rules, "B": rules}, rules, trials=30, target=0.9, level="any", seed=0,
                 oof_dir=tmp_path / "oof")
    assert res["model_available"] and res["deployed_variant"] == "fused"
    for v in ("model", "fused"):
        assert (v, ["A"]) in seen and (v, ["B"]) in seen and (v, ["A", "B"]) in seen
    cv = res["variants"]["model"]["cross_validated"]
    assert cv["events"] == 6 and cv["recall"] == 1.0 and cv["false_alarms"] == 0
    assert res["variants"]["rules"]["cross_validated"]["events_detected"] == 0
    assert {o["fold"] for o in res["variants"]["fused"]["outcomes"]} == {"A", "B"}


def test_rules_only_fallback_without_model_scores(tmp_path, monkeypatch):
    items = _fake_dataset(tmp_path, with_model=False)
    monkeypatch.setattr(fz, "replay_decisions",
                        lambda cf, rules: (30.0, [_dec("SUSPECT", 5.2)] if "Snatch" in cf.name else []))
    rules = object()
    res = fz.run(items, {"A": rules, "B": rules}, rules, trials=10, target=0.9, level="any", seed=0,
                 oof_dir=tmp_path / "oof")
    assert list(res["variants"]) == ["rules"] and res["deployed_variant"] == "rules"
    cv = res["variants"]["rules"]["cross_validated"]
    assert cv["recall"] == 1.0 and cv["precision"] == 1.0
    out = tmp_path / "fusion.tuned.yaml"
    fz.write_deployed(res, out)
    assert FusionParams.load(out).use_model is False


def test_acceptance_report_shows_variants_and_lower_bounds_next_to_pass(tmp_path, monkeypatch):
    from snatchguard.evaluation.acceptance import render
    items = _fake_dataset(tmp_path, with_model=True)
    monkeypatch.setattr(fz, "replay_decisions", lambda cf, rules: (30.0, []))
    rules = object()
    res = fz.run(items, {"A": rules, "B": rules}, rules, trials=20, target=0.9, level="any", seed=0,
                 oof_dir=tmp_path / "oof")
    text = render(None, res)
    assert "**PASS**" in text and "Lower 95% bounds" in text
    head = text.split("## Rules, learned scorer")[0]
    assert "Cross-validated recall **100.0%** (6/6" in head
    assert "lower confidence bounds do not" in head      # 6 events cannot prove >= 90 %
    for name in ("Rules only", "Learned scorer only", "Rules + learned scorer (fused) ← deployed"):
        assert name in text
