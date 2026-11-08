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
    assert not {92, 93} & used                               # disjoint seed namespaces


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
