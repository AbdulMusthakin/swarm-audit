"""Tests listed in docs/handoff-build.md section 8, plus the LLM-backend plumbing."""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from swarm_audit import build, calibration, stats
from swarm_audit.audit import (PTest, derive_actions_table, enumerated_table, fit_table,
                               marginal_weights, state_weights, synthetic_history)
from swarm_audit.game import DEFAULT_PARAMS, u
from swarm_audit.ladder import Labels, Record, pi_k, q, render
from swarm_audit.policies import HIDDEN_SIG, LLMBackend, Query, ScriptedBackend
from swarm_audit.prediction import ActionsChain, SignalChain, exact_law
from swarm_audit.states import (START, actions_state, all_actions_states, all_signal_states,
                                key, parse, signal_state)
from swarm_audit.swarm import Condition, build_records, load_runs, run_condition, simulate
from swarm_audit.topology import complete, cycle


LABELS = Labels(" H", " L", " high", " low", " none")


# ------------------------------------------------------------------------ topology

def test_cycle_degrees_and_neighbours():
    g = cycle(10)
    assert g.degrees == (2,) * 10 and g.regular_degree == 2
    assert g.in_neighbours[0] == (9, 1) and g.in_neighbours[5] == (4, 6)
    assert len(g.edges) == 10 and (0, 9) in g.edges
    assert g.distance(0, 2) == 2 and g.distance(0, 5) == 5
    assert (g.adjacency == g.adjacency.T).all()


def test_complete_degrees_and_neighbours():
    g = complete(10)
    assert g.regular_degree == 9 and len(g.edges) == 45
    assert g.in_neighbours[3] == (0, 1, 2, 4, 5, 6, 7, 8, 9)
    with pytest.raises(ValueError):
        cycle(2)


# ------------------------------------------------------------------------- records

def test_round_records_on_a_four_cycle():
    g = cycle(4)  # 0 sees 3 and 1; 1 sees 0 and 2; 2 sees 1 and 3; 3 sees 2 and 0
    recs = build_records(g, ["H", "L", "H", "H"], ["high", "none", "low", "high"])
    assert recs[0] == Record("H", 1, 1, "high", 1, 0, 1)   # 3: H/high, 1: L/none
    assert recs[1] == Record("L", 2, 0, "none", 1, 1, 0)   # 0: H/high, 2: H/low
    assert recs[2] == Record("H", 1, 1, "low", 1, 0, 1)    # 1: L/none, 3: H/high
    assert recs[3] == Record("H", 2, 0, "high", 1, 1, 0)   # 2: H/low,  0: H/high
    assert build_records(g, ["L", "L", "L", "H"], None)[0] == Record("L", 1, 1)


def test_payoffs_derived_from_counts():
    for a, k in [("H", 0), ("H", 1), ("H", 2), ("L", 0), ("L", 1), ("L", 2)]:
        r = render("counts", (Record(a, k, 2 - k),), 2, LABELS)
        val = u(a, k / 2, DEFAULT_PARAMS)
        assert f"Your payoff last round was {val:g} points." in r.user


# ----------------------------------------------------------------------------- RNG

def test_same_seed_reproduces_a_run_and_agents_draw_differently():
    be = ScriptedBackend("copy", render_prompts=False)
    c = Condition("t", 1, 1, "cheap_talk", 4, runs=3)
    a, b = simulate(c, be, [0, 1, 2]), simulate(c, be, [2, 1, 0])
    assert a[1] == b[1] and a[2] == b[2]          # order of runs in the batch is irrelevant
    us = [ln["u_action"] for ln in a[0] if ln["round"] == 1]
    assert len(set(us)) == 4                      # every agent its own draw
    assert [ln["u_action"] for ln in a[0]] != [ln["u_action"] for ln in a[1]]
    c2 = Condition("t", 1, 2, "cheap_talk", 4, runs=1)
    assert simulate(c2, be, [0])[0][0]["u_action"] != a[0][0]["u_action"]


# ------------------------------------------------------------------ q, pi_k, states

def test_q_pi_k_and_state_functions():
    h = (Record("L", 0, 2, "low", 0, 2, 0), Record("H", 1, 1, "none", 1, 0, 1))
    assert q(h[1]) == Record("H", 1, 1)
    assert pi_k(tuple(map(q, h)), 1) == (Record("H", 1, 1),)
    assert actions_state(()) == START and actions_state(h) == ("H", 1)
    assert signal_state(h) == ("H", 1, 1, 0, 1, "none")
    assert signal_state(h, own_signal=False) == ("H", 1, 1, 0, 1)
    assert parse(key(signal_state(h))) == signal_state(h)
    assert len(all_actions_states(2)) == 7 and len(all_actions_states(9)) == 21
    assert len(all_signal_states(2)) == 109 and len(all_signal_states(2, False)) == 37


def test_calibration_prompts_are_byte_identical_and_hide_the_round():
    last = Record("H", 1, 1, "high", 2, 0, 0)
    h1 = (Record("L", 0, 2, "low", 0, 1, 1), last)
    h2 = tuple(Record("H", 2, 0, "none", 0, 0, 2) for _ in range(7)) + (last,)
    for step, a, b in [("cheap_talk", h1, h2),
                       ("counts", tuple(map(q, h1)), tuple(map(q, h2)))]:
        r1, r2 = render(step, a, 2, LABELS, last_k=1), render(step, b, 2, LABELS, last_k=1)
        assert (r1.system, r1.user, r1.assistant) == (r2.system, r2.user, r2.assistant)
        assert "round 2" not in r1.user.lower() and "round 8" not in r1.user.lower()
        assert "rounds ago" not in r1.user
    be = ScriptedBackend("copy")
    o = be.query([Query("cheap_talk", h1, 2, 1), Query("cheap_talk", h2, 2, 1)])
    assert o[0].prompt_sha256 == o[1].prompt_sha256


# ----------------------------------------------------------------------- p_test

@pytest.mark.parametrize("channel", ["counts", "cheap_talk"])
def test_synthetic_generator(channel):
    pt = PTest()
    rng = np.random.default_rng(0)
    states = all_actions_states(2, start=False)
    if channel == "cheap_talk":
        states += all_signal_states(2, start=False)[::7]
    for s in states:
        for _ in range(20):
            h = synthetic_history(rng, channel, 2, s, pt)
            assert 1 <= len(h) <= 19
            st = signal_state(h) if len(s) > 2 else actions_state(h)
            assert st == s
            assert all(r.d == 2 and r.has_signals == (channel == "cheap_talk") for r in h)
    assert synthetic_history(rng, channel, 2, START, pt) == ()


def test_state_weights_and_marginalisation_weights_sum_to_one():
    for d in (2, 3):
        for own in (True, False):
            w = state_weights(all_signal_states(d, own), d)
            assert math.isclose(sum(w.values()), 1.0)
            for scheme in ("primary", "equal"):
                W = marginal_weights(d, own, scheme)
                assert set(W) == {key(s) for s in all_actions_states(d)}
                for s, ws in W.items():
                    assert math.isclose(sum(ws.values()), 1.0)
                    assert all(parse(k)[:2] == parse(s)[:2] for k in ws if s != "START")
    assert math.isclose(state_weights(all_actions_states(2), 2)["START"], 1 / 20)


# ------------------------------------------------------------------- tables, chains

def test_derived_table_marginalises_the_signal_table():
    be = ScriptedBackend("hidden_signal", render_prompts=False)
    sig = fit_table(be, "cheap_talk", 2, "signal", 4, 2)
    assert all(v < 1e-12 for v in sig.eps_state().values())   # depends on last-round signals only
    der = derive_actions_table(sig)
    # P(any of 2 neighbours says "high") is 5/9 under p_test, so kappa(H | s) = 0.2 + 0.6 * 5/9.
    for s in all_actions_states(2, start=False):
        assert math.isclose(der.kappa_H(key(s)), 0.2 + 0.6 * 5 / 9)
    assert der.eps_state()["H-1"] > 0.25
    eq = derive_actions_table(sig, "equal")    # 6 configurations, 3 with a "high": 1/2
    assert math.isclose(eq.kappa_H("H-1"), 0.5)


def _copy_chain():
    lk = {key(s): (0.5 if s == START else 0.2 + 0.3 * s[1]) for s in all_actions_states(2)}
    return ActionsChain(lk, 2)


def test_exact_law_sums_to_one_and_matches_a_long_copy_simulation(tmp_path):
    g = cycle(4)
    ex = exact_law(_copy_chain(), g, 20)
    assert np.allclose(ex["X"].sum(1), 1) and math.isclose(ex["y"].sum(), 1)
    assert ex["n_profiles"] == 16
    c = Condition("t", 1, 1, "counts", 4, runs=3000)
    data = load_runs(run_condition(c, ScriptedBackend("copy", render_prompts=False), tmp_path,
                                   log=lambda *a: None))
    y = stats.high_count(data.actions)
    gap = stats.ks_gap(y, stats.cdf_from_law(ex["y"]))
    assert gap < stats.ks_threshold(ex["y"], 3000, 4000, 0.999)
    X5 = np.bincount(data.actions[:, 4].sum(1), minlength=5) / 3000
    assert np.abs(X5 - ex["X"][4]).max() < 0.03


def test_signal_aware_exact_law_matches_a_hidden_signal_simulation(tmp_path):
    be = ScriptedBackend("hidden_signal", render_prompts=False)
    sig = enumerated_table(be, "cheap_talk", 2, "signal")
    ch = SignalChain(sig.lookup(), 2)
    g = cycle(4)
    ex = exact_law(ch, g, 20)
    assert ex["n_profiles"] == 1296 and math.isclose(ex["y"].sum(), 1)
    c = Condition("t", 1, 2, "cheap_talk", 4, runs=3000)
    data = load_runs(run_condition(c, be, tmp_path, log=lambda *a: None))
    y = stats.high_count(data.actions)
    assert stats.ks_gap(y, stats.cdf_from_law(ex["y"])) < stats.ks_threshold(ex["y"], 3000, 4000, 0.999)
    # The chain's own simulation agrees too.
    A = ch.simulate(g, 20, 20000, np.random.default_rng(0))
    emp = np.bincount(stats.high_count(A), minlength=21) / 20000
    assert np.abs(emp - ex["y"]).max() < 0.01


# ---------------------------------------------------------------------------- CRPS

def test_crps_hand_example_and_point_mass():
    G = np.array([0.25, 0.75, 1.0])           # K = 2 (5n = 2 for this toy)
    # y = 1: (0.25 - 0)^2 + (0.75 - 1)^2 + 0 = 0.125, over K = 2: 0.0625
    assert np.allclose(stats.crps(G, np.array([1])), [0.0625])
    # y = 0: (0.25 - 1)^2 + (0.75 - 1)^2 = 0.625 -> 0.3125
    assert np.allclose(stats.crps(G, np.array([0])), [0.3125])
    point = np.array([0.0, 1.0, 1.0])
    assert np.allclose(stats.crps(point, np.array([1])), [0.0])


# --------------------------------------------------------------------- statistics

def test_consensus_and_edge_disagreement():
    A = np.zeros((3, 6, 4), np.int8)
    A[0, 2:] = 1                    # rounds 1-2 all L, then all H from round 3
    A[1, :, 0] = 1                  # never unanimous
    A[2, 0] = 1
    A[2, 1:3] = 0
    A[2, 3:] = [1, 0, 1, 0]
    c = stats.consensus(A)
    assert c.side[0] == "H" and c.tau[0] == 3
    assert np.isnan(c.tau[1]) and c.side[1] == ""
    assert c.side[2] == "" and np.isnan(c.tau[2])   # L in rounds 2-3 only: two rounds, no consensus
    assert c.survival().shape == (4,)
    ed = stats.edge_disagreement(A, cycle(4))
    assert ed[2, 3] == 1.0 and ed[0, 0] == 0.0 and ed[1, 0] == 0.5


def test_defect_shift_identity():
    rng = np.random.default_rng(0)
    states = np.array(rng.choice(["a", "b", "c"], size=(5, 4, 3)), object)
    defects = rng.random((5, 4, 3))
    e_Q, w_Q = {"a": 0.1, "b": 0.2, "c": 0.05}, {"a": 0.5, "b": 0.3, "c": 0.2}
    r = stats.defect_shift(defects, states, e_Q, w_Q)
    assert abs(r["check"]) < 1e-12
    assert math.isclose(r["state_frequency_term"] + r["within_state_term"], r["difference"])


# ------------------------------------------------------------------ LLM plumbing

def _llm(tok, fake_call):
    ids_a = tuple(tok.encode(x, add_special_tokens=False)[-1] for x in ("Bid: H", "Bid: L"))
    ids_s = tuple(tok.encode("Message:" + x, add_special_tokens=False)[-1]
                  for x in (" high", " low", " none"))
    return LLMBackend(tok, LABELS, ids_a, ids_s, fake_call)


def test_llm_backend_plumbing_and_calibration_check(toy_tokenizer, fake_server, tmp_path):
    be = _llm(toy_tokenizer, fake_server)
    h = (Record("H", 1, 1, "high", 1, 0, 1),)
    o = be.query([Query("cheap_talk", h, 2), Query("counts", (q(h[0]),), 2)])
    assert 0 < o[0].p_H < 1 and o[0].p_valid < 1 and set(o[0].sig) == {"H", "L"}
    assert math.isclose(sum(o[0].sig["H"]), 1) and o[1].sig is None
    assert o[0].n_tokens > 50 and len(o[0].prompt_sha256) == 64
    assert math.isclose(o[0].p_H, math.exp(o[0].logprob_H) / o[0].p_valid)
    # Gate-G5 code path: last-round-only swarm against the enumerated table.
    for ch in ("counts", "cheap_talk"):
        tab = enumerated_table(be, ch, 2, "actions" if ch == "counts" else "signal")
        c = Condition("t", 4, 1, ch, 4, last_k=1, runs=3)
        data = load_runs(run_condition(c, be, tmp_path, log=lambda *a: None))
        res = calibration.per_decision(data, tab, floor=1e-12)
        assert res["pass"], res
    # A wrong label id is caught.
    bad = LLMBackend(toy_tokenizer, LABELS, (1, 2), be.signal_ids, fake_server)
    with pytest.raises(ValueError):
        bad.query([Query("counts", (), 2)])


def test_hidden_signal_policy_is_as_specified():
    be = ScriptedBackend("hidden_signal", render_prompts=False)
    h_yes = (Record("L", 0, 2, "none", 1, 1, 0),)
    h_no = (Record("H", 2, 0, "high", 0, 1, 1),)
    o = be.query([Query("cheap_talk", h_yes, 2), Query("cheap_talk", h_no, 2)])
    assert o[0].p_H == 0.8 and o[1].p_H == 0.2 and o[0].sig == HIDDEN_SIG


# ------------------------------------------------------------------------- configs

SHARED = ("inputs", "model", "payoffs", "T", "p_test", "own_signal_in_state", "audit", "seeds",
          "analysis", "serving")


def test_build_configs_agree_on_shared_sections():
    cfgs = [yaml.safe_load(open(Path("configs") / f)) for f in
            ("core.yaml", "calibration.yaml", "secondary.yaml")]
    for k in SHARED:
        assert all(c[k] == cfgs[0][k] for c in cfgs), k
    h = yaml.safe_load(open("configs/harness.yaml"))
    for k in ("T", "payoffs", "p_test", "audit", "analysis", "own_signal_in_state", "seeds"):
        assert h[k] == cfgs[0][k], k


def test_gpu_scripts_stop_without_abdul_inputs():
    need = ("model_confirmed", "prompts_approved", "modal_account", "pilot_rule")
    cfg = build.load("core.yaml")
    build.require_inputs(cfg, need)                      # all recorded (7 Nov 2026)
    for k, blank in (("model_confirmed", False), ("modal_account", None), ("pilot_rule", None),
                     ("prompts_approved", {"counts": True, "cheap_talk": False})):
        c = build.load("core.yaml")
        c["inputs"][k] = blank
        with pytest.raises(SystemExit):
            build.require_inputs(c, need)
