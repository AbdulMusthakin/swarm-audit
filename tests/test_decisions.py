"""Abdul's decisions of 7 Nov 2026: pilot rule, payoff freeze, Modal workspace guard,
final-prompt label check and the signal-responsiveness re-measure."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from swarm_audit import build, pilot, responsiveness
from swarm_audit.ladder import Labels, render
from swarm_audit.policies import LLMBackend

RULE = yaml.safe_load(open("configs/pilot.yaml"))["rule"]


def _runs(streak_starts, R=5, T=20, n=10, seed=0):
    """Pilot-like actions: run r is all-low for rounds start..start+2 (1-based) if given,
    and mixed (half high) everywhere else."""
    A = np.zeros((R, T, n), np.int8)
    A[:, :, ::2] = 1
    for r, start in enumerate(streak_starts):
        if start is not None:
            A[r, start - 1:start + 2] = 0
    return A


def test_pilot_rule_triggers_on_three_runs():
    d = pilot.decide(_runs([1, 2, 3, None, None]), RULE)
    assert d["apply_change"] and not d["ambiguous"]
    assert d["runs_meeting_condition"]["tau_le_by_round"] == 3


def test_pilot_rule_does_not_trigger_on_two_runs():
    d = pilot.decide(_runs([1, 2, None, None, None]), RULE)
    assert not d["apply_change"] and not d["ambiguous"]


def test_pilot_rule_flags_readings_that_disagree():
    d = pilot.decide(_runs([4, 4, 5, None, None]), RULE)   # tau = 4, 4, 5: not done by round 5
    assert d["triggered"]["tau_le_by_round"]
    assert not d["triggered"]["three_rounds_completed_by_round"]
    assert d["ambiguous"]


def test_fallback_keeps_the_prisoners_dilemma_and_freezes_every_config(tmp_path):
    newp = pilot.new_payoffs({"uLH": 12, "uHH": 10, "uLL": 4, "uHL": 0}, RULE["change"])
    assert newp == {"uLH": 11, "uHH": 10, "uLL": 4, "uHL": 0}
    with pytest.raises(ValueError):
        pilot.new_payoffs(newp, {"uLH": 25})
    for f in pilot.PAYOFF_CONFIGS:
        shutil.copy(Path("configs") / f, tmp_path / f)
    pilot.freeze_payoffs(newp, folder=tmp_path)
    for f in pilot.PAYOFF_CONFIGS:
        c = yaml.safe_load(open(tmp_path / f))
        assert c["payoffs"] == newp, f
        assert "# " in (tmp_path / f).read_text()           # comments kept


def test_inputs_recorded_and_prompts_not_yet_frozen():
    cfg = build.load("core.yaml")
    assert cfg["inputs"]["model_confirmed"] is True
    assert cfg["inputs"]["prompts_approved"] == {"cheap_talk": True, "counts": True}
    assert build.missing_inputs(cfg, ("prompts_approved",)) == []
    cfg["inputs"]["prompts_approved"]["counts"] = False
    miss = build.missing_inputs(cfg, ("prompts_approved",))
    assert len(miss) == 1 and miss[0].endswith("counts")
    assert cfg["inputs"]["pilot_rule"] == RULE["name"]
    assert "abdulmusthakin" in cfg["inputs"]["forbidden_workspaces"]
    assert cfg["model"]["hf_id"] == "meta-llama/Llama-3.1-8B-Instruct"


def test_workspace_guard(monkeypatch):
    cfg = build.load("core.yaml")
    monkeypatch.setattr(build, "current_workspace", lambda: "abdulmusthakin")
    with pytest.raises(SystemExit):
        build.check_workspace(cfg)
    monkeypatch.setattr(build, "current_workspace", lambda: "teammate-ws")
    assert build.check_workspace(cfg) == "teammate-ws"


def test_messages_are_about_the_next_bid():
    lab = Labels(" H", " L", " high", " low", " none")
    r = render("cheap_talk", (), 2, lab)
    assert "about your bid in the next round" in r.system
    assert "messages your neighbours sent last round are about their bids in this round" in r.system
    assert '"I\'ll bid high"' not in r.system + r.user      # only the next-round wording


def _toy_cfg(tok):
    cfg = build.load("core.yaml")
    ids_a = [tok.encode(x, add_special_tokens=False)[-1] for x in ("Bid: H", "Bid: L")]
    ids_s = [tok.encode("Message:" + x, add_special_tokens=False)[-1]
             for x in (" high", " low", " none")]
    cfg["model"]["label_ids"] = {"actions": ids_a, "signals": ids_s}
    cfg["model"]["chat_kwargs"] = {}
    return cfg


def test_final_label_check(toy_tokenizer):
    cfg = _toy_cfg(toy_tokenizer)
    res = build.check_final_labels(toy_tokenizer, cfg)
    assert res["pass"], res["problems"]
    assert res["prompts_checked"] == 148 and res["max_prompt_tokens"]["cheap_talk"] > 0
    cfg["model"]["label_ids"]["actions"] = [1, 2]
    assert not build.check_final_labels(toy_tokenizer, cfg)["pass"]


def test_signal_responsiveness_runs_on_the_llm_path(toy_tokenizer, fake_server):
    cfg = _toy_cfg(toy_tokenizer)
    m = cfg["model"]
    be = LLMBackend(toy_tokenizer, build.labels(cfg), tuple(m["label_ids"]["actions"]),
                    tuple(m["label_ids"]["signals"]), fake_server)
    r = responsiveness.measure(be)
    for d in ("d4", "d2"):
        assert 0 <= r[d]["signal_responsiveness_length1_mean"] <= 1
        assert len(r[d]["prompts"]) == 32 and set(r[d]["by_own_last_length1"]) == {"H", "L"}
