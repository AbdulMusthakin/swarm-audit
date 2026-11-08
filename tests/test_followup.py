"""Extra-arms overnight follow-up (docs/overnight-extra-plan.md)."""

from __future__ import annotations

from swarm_audit import build


def test_followup_config_differs_from_olmo_config_only_in_arms():
    a, b = build.load("extra-olmo.yaml"), build.load("extra-followup-olmo.yaml")
    for k in set(a) | set(b):
        if k in ("arms", "_path"):
            continue
        assert a.get(k) == b.get(k), k
    assert build.spend_file(a) == build.spend_file(b)       # one ledger, one $25 cap
    assert build.gates_dir(a) == build.gates_dir(b)         # 9a's G5 record is shared
    (c10,) = build.conditions(b, "followup_olmo_n10_last1")
    (c4,) = build.conditions(b, "followup_olmo_n4_full")
    assert (c10.arm_id, c10.n, c10.last_k, c10.runs) == (92, 10, 1, 50)
    assert (c4.arm_id, c4.n, c4.last_k, c4.runs) == (93, 4, None, 50)
    used = {int(v["arm_id"]) for cfg in (a, build.load("extra-scale.yaml"), build.load("core.yaml"),
                                         build.load("calibration.yaml")) for v in cfg["arms"].values()}
    assert not {92, 93, 94, 95, 96, 97} & used               # disjoint seed namespaces
    obs = {k: v["observe"] for k, v in b["arms"].items() if "observe" in v}
    assert obs == {"followup_obs_unmodified": "unmodified", "followup_obs_own_null": "own_null",
                   "followup_obs_neighbour_null": "neighbour_null", "followup_obs_both_null": "both_null"}
    for k in obs:
        (c,) = build.conditions(b, k)
        assert (c.n, c.last_k, c.runs) == (4, 1, 50)


# ------------------------------------------------- exact ablations and interventions

import numpy as np  # noqa: E402

from swarm_audit import followup as fu  # noqa: E402
from swarm_audit.topology import make  # noqa: E402


def _scripted_J(d=2, seed=0):
    """A random valid kernel array with distinct entries everywhere."""
    rng = np.random.default_rng(seed)
    J = rng.random((2, d + 1, d + 1, d + 1, 3, 6)) + 0.1
    return J / J.sum(-1, keepdims=True)


def test_variant_averaging_weights_and_shapes():
    d = 2
    J = _scripted_J(d)
    w = fu.comp_weights(d)
    assert w[(1, 1, 0)] == 2 / 9 and w[(2, 0, 0)] == 1 / 9
    own = fu.variant_array(J, d, "own_only")
    want = sum(wc * J[0, 1, c[0], c[1], 2] for c, wc in w.items())
    assert np.allclose(own[0, 1, 0, 2, 2], want) and np.allclose(own[0, 1, 2, 0, 2], want)
    nb = fu.variant_array(J, d, "neighbours_only")
    assert np.allclose(nb[1, 0, 1, 0, 0], J[1, 0, 1, 0].mean(0))
    nei = fu.variant_array(J, d, "neither")
    assert np.allclose(nei[0, 2, 0, 0, 0], sum(wc * J[0, 2, c[0], c[1]].mean(0) for c, wc in w.items()))
    assert np.allclose(fu.variant_array(J, d, "both"), J)
    eq = fu.variant_array(J, d, "own_only", "equal")
    assert np.allclose(eq[0, 0, 0, 0, 1], np.mean([J[0, 0, c[0], c[1], 1] for c in w], 0))


def test_routing_and_nulling_on_the_cycle():
    """Agent 1's neighbours on the 4-cycle are 0 and 2; its own message is its own."""
    d, g = 2, make("cycle", 4)
    J = _scripted_J(d, 1)
    # profile: agent 0 (H, high), 1 (L, low), 2 (H, none), 3 (L, high); output = a*3 + m
    x = (0 * 3 + 0, 1 * 3 + 1, 0 * 3 + 2, 1 * 3 + 0)
    Pr = fu.profiles(4)
    xi = int(np.nonzero((Pr == x).all(1))[0][0])
    V = fu.agent_vectors(J, g)
    assert np.allclose(V[xi, 1], J[1, 2, 1, 0, 1])          # L, 2 high nbrs, (1 high, 0 low, 1 none), own low
    Vo = fu.agent_vectors(J, g, fu.observation("own_null"))
    assert np.allclose(Vo[xi, 1], J[1, 2, 1, 0, 2])
    Vn = fu.agent_vectors(J, g, fu.observation("neighbour_null"))
    assert np.allclose(Vn[xi, 1], J[1, 2, 0, 0, 1])          # action counts kept, messages (0, 0, 2)
    Vb = fu.agent_vectors(J, g, fu.observation("both_null"))
    assert np.allclose(Vb[xi, 1], J[1, 2, 0, 0, 2])
    # agent 0: neighbours 3 (L, high) and 1 (L, low); own (H, high)
    assert np.allclose(V[xi, 0], J[0, 0, 1, 1, 0])


def test_uniform_replacement_equals_averaged_audit_and_rows_sum_to_one():
    d, g = 2, make("cycle", 4)
    J = _scripted_J(d, 2)
    for own, nbr, var in ((True, False, "neighbours_only"), (False, True, "own_only"), (True, True, "neither")):
        P_u = fu.transition_matrix(fu.agent_vectors(J, g, fu.uniform_observation(d, own, nbr)))
        P_v = fu.transition_matrix(fu.agent_vectors(fu.variant_array(J, d, var), g))
        assert np.abs(P_u - P_v).max() < 1e-12
        assert np.abs(P_u.sum(1) - 1).max() < 1e-12


def test_propagate_matches_prediction_exact_law_and_crps_identity():
    from swarm_audit import stats
    from swarm_audit.prediction import SignalChain, exact_law
    from swarm_audit.states import key, sig_compositions
    from swarm_audit.game import SIGNALS

    d, g = 2, make("cycle", 4)
    J = _scripted_J(d, 3)
    start = np.array([0.3, 0.1, 0.2, 0.1, 0.2, 0.1])
    lk = {"START": start.reshape(2, 3)}
    for ai, a in enumerate("HL"):
        for k in range(d + 1):
            for (sh, sl, sn) in sig_compositions(d):
                for oi, o in enumerate(SIGNALS):
                    lk[key((a, k, sh, sl, sn, o))] = J[ai, k, sh, sl, oi].reshape(2, 3)
    L = fu.propagate(fu.transition_matrix(fu.agent_vectors(J, g)), fu.start_law(start, 4), 4, 20)
    ex = exact_law(SignalChain(lk, d, True), g, 20)
    assert np.abs(L["y"] - ex["y"]).max() < 1e-12
    assert np.abs(L["EX"] - ex["X"] @ np.arange(5) / 4).max() < 1e-12
    # E_{y~G}[CRPS(Q, y) - CRPS(G, y)] = sum (Q - G)^2 / K
    G = np.cumsum(L["y"])
    Q = np.cumsum(np.full(21, 1 / 21))
    lhs = float((L["y"] * (stats.crps(Q, np.arange(21)) - stats.crps(G, np.arange(21)))).sum())
    assert abs(lhs - fu.expected_excess_crps(Q, G)) < 1e-12


# ------------------------------------------------------------ live observation hook


def _cond(arm="t"):
    from swarm_audit.swarm import Condition

    return Condition(arm=arm, arm_id=99, condition_id=4, channel="cheap_talk", n=4, last_k=1,
                     runs=3, T=6)


def test_simulate_without_observer_is_unchanged_and_identity_only_adds_a_field():
    from swarm_audit.policies import ScriptedBackend
    from swarm_audit.swarm import simulate

    be = ScriptedBackend("hidden_signal")
    a = simulate(_cond(), be, [0, 1, 2])
    b = simulate(_cond(), be, [0, 1, 2], fu.record_observer("unmodified"))
    for r in a:
        for x, y in zip(a[r], b[r]):
            assert y.pop("observed_record") == x["record"]
            assert x == y


def test_live_neighbour_null_changes_what_is_shown_not_what_is_sent():
    from swarm_audit.policies import ScriptedBackend
    from swarm_audit.swarm import simulate

    be = ScriptedBackend("hidden_signal")       # P(H) = 0.8 iff a neighbour said "high"
    lines = simulate(_cond(), be, [0, 1, 2], fu.record_observer("neighbour_null"))
    for r, rows in lines.items():
        for row in rows:
            if row["round"] > 1:
                assert row["p_H"] == 0.2                       # never shown a "high"
                assert row["state_sig"].split("-")[2:5] == ["0", "0", "2"]
            o, a = row["observed_record"], row["record"]
            assert (o["sig_high"], o["sig_low"], o["sig_none"]) == (0, 0, 2)
            assert {k: a[k] for k in ("action", "n_high", "n_low", "signal")} == \
                   {k: o[k] for k in ("action", "n_high", "n_low", "signal")}
    sent = [row["record"]["sig_high"] for rows in lines.values() for row in rows]
    assert max(sent) > 0                                       # actual messages still logged


def test_live_own_null_routes_only_the_own_message():
    from swarm_audit.policies import ScriptedBackend
    from swarm_audit.swarm import simulate

    lines = simulate(_cond(), ScriptedBackend("hidden_signal"), [0], fu.record_observer("own_null"))
    for row in lines[0]:
        assert row["observed_record"]["signal"] == "none"
        assert {k: v for k, v in row["observed_record"].items() if k != "signal"} == \
               {k: v for k, v in row["record"].items() if k != "signal"}
        if row["round"] > 1:
            assert row["state_sig"].endswith("-none")


def test_primary_contrast_label_needs_the_core_arm():
    import json

    from swarm_audit.analysis import is_primary_contrast
    from swarm_audit.build import REPO_ROOT

    core = json.loads((REPO_ROOT / "results/runs/core_cheap_talk/cheap_talk_n10_cycle_full/provenance.json")
                      .read_text())["payload"]["design"]
    olmo = json.loads((REPO_ROOT / "results/extra/runs/olmo_core_cheap_talk/cheap_talk_n10_cycle_full/provenance.json")
                      .read_text())["payload"]["design"]
    assert is_primary_contrast("cheap_talk", 10, "cycle", core)
    assert not is_primary_contrast("cheap_talk", 10, "cycle", olmo)
    assert not is_primary_contrast("cheap_talk", 10, "cycle", {**core, "last_k": 1})
    assert not is_primary_contrast("counts", 10, "cycle", core)


def test_first_entry_excludes_reentries():
    """First-entry mass equals the hitting probability computed with AH made absorbing."""
    d, g = 2, make("cycle", 4)
    J = _scripted_J(d, 4)
    P = fu.transition_matrix(fu.agent_vectors(J, g))
    v1 = fu.start_law(np.full(6, 1 / 6), 4)
    AH = fu.all_high(4)
    T = 8
    E = fu.first_entry(P, v1, AH, T)
    Pa = P.copy()
    Pa[AH] = 0.0
    Pa[np.ix_(AH, AH)] = np.eye(int(AH.sum()))             # absorb on entry
    v = v1.copy()
    for _ in range(T - 1):
        v = v @ Pa
    assert abs(E.sum() - v[AH].sum()) < 1e-12
    L = fu.propagate(P, v1, 4, T)
    assert fu.all_entry(L["v"], P, AH).sum() >= E.sum() - 1e-15
