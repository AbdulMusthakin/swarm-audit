"""The decisions settled on 7 Nov 2026 (decisions: in the build configs)."""

import shutil
from pathlib import Path

import pytest
import yaml

from swarm_audit import build, decisions

RULES = build.load("core.yaml")["decisions"]


def _g3(pH, sig, thr):
    return {"max_abs_diff_p_H": pH, "max_abs_diff_signal": sig,
            "throughput_prompts_per_s_batch": thr}


def test_g3_exact_default_keeps_defaults():
    d = decisions.decide_g3(_g3(0.0, 0.0, 100), None, RULES["g3"])
    assert d["status"] == "decided" and d["batch_invariant"] is False and d["numerical_floor"] == 0


def test_g3_needs_the_invariant_run_when_default_is_not_exact():
    assert decisions.decide_g3(_g3(0.02, 0.0, 100), None, RULES["g3"])["status"] == \
        "need_batch_invariant_run"


def test_g3_prefers_invariance_only_if_better_and_fast_enough():
    d0 = _g3(0.03, 0.01, 100)
    d = decisions.decide_g3(d0, _g3(0.0, 0.0, 60), RULES["g3"])
    assert d["batch_invariant"] is True and d["numerical_floor"] == 0
    d = decisions.decide_g3(d0, _g3(0.0, 0.0, 40), RULES["g3"])       # too slow
    assert d["batch_invariant"] is False and d["numerical_floor"] == 0.03
    d = decisions.decide_g3(d0, _g3(0.03, 0.0, 100), RULES["g3"])     # not better
    assert d["batch_invariant"] is False


def test_g3_floor_above_the_limit_stops():
    d = decisions.decide_g3(_g3(0.2, 0.0, 100), _g3(0.1, 0.0, 100), RULES["g3"])
    assert d["status"] == "stop"


def test_drift_rule():
    assert decisions.drift_action(0.0, 0.0, 0.035) == "ok"
    assert decisions.drift_action(0.02, 0.0, 0.035) == "report"
    assert decisions.drift_action(0.05, 0.0, 0.035) == "pause"
    assert decisions.drift_action(0.05, 0.03, 0.035) == "report"


def _cfgs():
    return {f: build.load(f) for f in ("core.yaml", "calibration.yaml", "secondary.yaml",
                                       "pilot.yaml")}


def _plan(sec_counts, sec_cheap, spent=0.0):
    return decisions.plan_g4(_cfgs(), {"counts": sec_counts, "cheap_talk": sec_cheap}, spent)


def test_g4_plan_everything_fits_when_fast():
    p = _plan(0.002, 0.006)
    assert p["status"] == "go" and p["steps"] == []          # arm 7 included, nothing cut
    assert {i["arm"] for i in p["run"]} >= {"core_counts", "core_cheap_talk",
                                           "second_topology_counts", "calibration_cheap_talk"}
    assert p["signal_audit"] == {"n_fit": 50, "n_score": 25}


def test_g4_plan_cuts_in_the_settled_order():
    full = _plan(0.002, 0.006)
    hours = full["projected_hours"]
    # Slow enough that the full plan breaks the GPU-hour limit: arm 7 goes, the audit is
    # reduced, then n = 20 is cut; arms 1-4b and n = 10 always stay.
    for scale in (1.5, 3, 6):
        p = _plan(0.002 * scale * 12 / hours / 1.25, 0.006 * scale * 12 / hours / 1.25)
        if p["status"] != "go":
            assert p["reason"].startswith("arms 1-4b")
            continue
        arms = {(i["arm"], i["condition"]) for i in p["run"]}
        assert ("calibration_counts", "counts_n4_cycle_last1") in arms
        assert ("core_cheap_talk", "cheap_talk_n10_cycle_full") in arms
        assert not any(a.startswith("second") for a, _ in arms)
        if any("cut n20" in s for s in p["steps"]):
            assert any("audit reduced" in s for s in p["steps"])
            assert ("core_counts", "counts_n20_cycle_full") not in arms


def test_g4_plan_stops_when_even_the_minimum_does_not_fit():
    p = _plan(1.0, 3.0)
    assert p["status"] == "stop"


def test_spent_money_counts_against_the_cap():
    assert _plan(0.002, 0.006, spent=19.99)["status"] == "stop"


def test_set_config_line_keeps_comments(tmp_path):
    for f in build.BUILD_CONFIGS:
        shutil.copy(Path("configs") / f, tmp_path / f)
    build.set_config_line("numerical_floor", "0.03125", folder=tmp_path, note="G3")
    build.set_config_line("batch_invariant", "true", folder=tmp_path)
    build.set_config_line("signal", "{n_fit: 25, n_score: 15}", folder=tmp_path)
    for f in build.BUILD_CONFIGS:
        c = yaml.safe_load(open(tmp_path / f))
        assert c["serving"]["numerical_floor"] == 0.03125 and c["serving"]["batch_invariant"] is True
        assert c["audit"]["signal"] == {"n_fit": 25, "n_score": 15}
        assert c["decisions"]["g4"]["reduced_signal_audit"] == {"n_fit": 25, "n_score": 15}
        assert "# Decisions settled in advance" in (tmp_path / f).read_text()


def test_floor_must_be_set_before_model_runs():
    cfg = build.load("core.yaml")
    assert build.numerical_floor(cfg) == 0.0          # set by G3 (7 Nov 2026)
    cfg["serving"]["numerical_floor"] = None
    with pytest.raises(SystemExit):
        build.numerical_floor(cfg)
