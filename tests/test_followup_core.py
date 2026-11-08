"""Exploratory follow-up code (docs/overnight-core-plan.md): identities and indexing."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from swarm_audit import followup_core as fu
from swarm_audit import stats
from swarm_audit.audit import derive_actions_table, fit_table
from swarm_audit.policies import ScriptedBackend
from swarm_audit.prediction import ActionsChain, SignalChain
from swarm_audit.topology import cycle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


@pytest.fixture(scope="module")
def sig():
    return fit_table(ScriptedBackend("two_round", render_prompts=False), "cheap_talk", 2,
                     "signal", 12, 6)


@pytest.fixture(scope="module")
def counts():
    return fit_table(ScriptedBackend("two_round", render_prompts=False), "counts", 2,
                     "actions", 12, 6)


def test_round_crps_matches_the_endpoint_formula_per_round():
    rng = np.random.default_rng(0)
    A_m = (rng.random((500, 6, 5)) < 0.4).astype(np.int8)
    A_s = (rng.random((7, 6, 5)) < 0.5).astype(np.int8)
    got = fu.round_crps(A_m, A_s)
    for t in range(6):
        G = stats.cdf_from_samples(A_m[:, t].sum(1), 5)
        want = stats.crps(G, A_s[:, t].sum(1))     # same (1/K) sum_k normalisation, K = n
        assert np.allclose(got[:, t], want)
    Gx = np.stack([stats.cdf_from_samples(A_m[:, t].sum(1), 5) for t in range(6)])
    assert np.allclose(fu.round_crps_cdf(Gx, A_s), got)


def test_order0_cdfs_are_binomial():
    Gx, Gy = fu.order0_cdfs(0.3, 4, 3, 5)
    assert Gx.shape == (3, 5) and Gy.shape == (21,)
    assert math.isclose(Gx[0, 0], 0.7 ** 4) and math.isclose(Gy[-1], 1.0)


def test_block_means_use_one_based_rounds():
    v = np.arange(20, dtype=float)[None, :] + 1     # value = round number
    b = fu.block_means(v)
    assert b["2-7"][0] == np.mean(range(2, 8)) and b["14-20"][0] == np.mean(range(14, 21))


def test_clock_tables_partition_the_fitting_histories(sig):
    tabs = [fu.clock_table(sig, b) for b in range(3)]
    for k, e in sig.entries.items():
        if k == "START":
            assert all(t.entries[k]["kappa"] == e["kappa"] for t, _ in tabs)
            continue
        ns = [inf["fit_counts"][k] for _, inf in tabs]
        assert sum(ns) == len(e["fit"])
        # The count-weighted mean of the block tables is the pooled table (no fallbacks).
        if all(n > 0 for n in ns):
            pooled = sum(n * np.asarray(t.entries[k]["kappa"]) for n, (t, _) in zip(ns, tabs)) / sum(ns)
            assert np.allclose(pooled, e["kappa"])
        for b, (t, inf) in enumerate(tabs):
            lo, hi = fu.LENGTH_BLOCKS[b]
            if k not in inf["fallback_states"]:
                assert all(lo <= L <= hi for L in t.entries[k]["fit_L"])
            assert all(lo <= L <= hi for L in t.entries[k]["score_L"])


def test_clock_table_falls_back_to_pooled_on_an_empty_cell(sig):
    k = next(k for k in sig.entries if k != "START")
    e = sig.entries[k]
    keep = [i for i, L in enumerate(e["fit_L"]) if L > 6]
    t2 = type(sig)(sig.channel, sig.d, sig.kind, sig.source, sig.own_signal,
                   {**sig.entries, k: {**e, "fit": [e["fit"][i] for i in keep],
                                       "fit_L": [e["fit_L"][i] for i in keep]}}, sig.meta)
    t, inf = fu.clock_table(t2, 0)
    assert k in inf["fallback_states"] and t.entries[k]["kappa"] == e["kappa"]


def test_length_blocks_cover_1_to_19_and_rounds_map_to_them():
    assert [fu.length_block(L) for L in range(1, 20)] == [0] * 6 + [1] * 6 + [2] * 7
    with pytest.raises(ValueError):
        fu.length_block(0)


def test_clock_chain_with_identical_blocks_reproduces_the_chain(sig, counts):
    g = cycle(6)
    for ch in (SignalChain(sig.lookup(), 2, sig.own_signal), ActionsChain(counts.lookup(), 2)):
        a = ch.simulate(g, 20, 300, np.random.default_rng(5))
        b = fu.ClockChain([ch, ch, ch], "x").simulate(g, 20, 300, np.random.default_rng(5))
        assert np.array_equal(a, b)


def test_clock_chain_uses_the_block_of_l_equals_t_minus_1(counts):
    # Block tables with P(H) = 0, 1, 0 after START: rounds 2-7 low, 8-13 high, 14-20 low.
    def table_with(p):
        return ActionsChain({k: (counts.kappa_H(k) if k == "START" else p) for k in counts.entries}, 2)
    cc = fu.ClockChain([table_with(0.0), table_with(1.0), table_with(0.0)], "x")
    A = cc.simulate(cycle(5), 20, 10, np.random.default_rng(0))
    X = A.mean(2)
    assert (X[:, 1:7] == 0).all() and (X[:, 7:13] == 1).all() and (X[:, 13:] == 0).all()


def test_resampled_table_keeps_start_and_vector_integrity(sig):
    r = fu.resample_table(sig, np.random.default_rng(1))
    assert r.entries["START"] is sig.entries["START"]
    for k, e in r.entries.items():
        kap = np.asarray(e["kappa"])
        assert math.isclose(kap.sum(), 1.0, rel_tol=1e-9)       # mean of six-way joints
    der = derive_actions_table(r)
    assert set(der.entries) == set(derive_actions_table(sig).entries)


def test_resample_with_lengths_keeps_pairs(counts):
    from followup_core_clock import resample_with_lengths

    # Make every fitting value encode its own length, then check the pairs survive.
    ent = {k: (e if k == "START" else {**e, "fit": [L / 100 for L in e["fit_L"]]})
           for k, e in counts.entries.items()}
    t = type(counts)(counts.channel, counts.d, counts.kind, counts.source, counts.own_signal,
                     ent, counts.meta)
    r = resample_with_lengths(t, np.random.default_rng(2))
    for k, e in r.entries.items():
        if k != "START":
            assert all(math.isclose(f, L / 100) for f, L in zip(e["fit"], e["fit_L"]))
            assert math.isclose(e["kappa"], np.mean(e["fit"]))


def test_one_step_errors_and_ratio_tables():
    p = np.array([[[0.6, 0.4], [0.3, 0.9]]])        # (R=1, T=2, n=2)
    s = np.array([[["START", "START"], ["H-1", "L-0"]]])
    e = fu.one_step_errors(p, s, {"START": 0.5, "H-1": 0.2, "L-0": 0.5})
    assert np.allclose(e, [[[0.1, -0.1], [0.1, 0.4]]])
    tab = fu.error_tables(np.repeat(e, 3, 0).repeat(10, 1)[:, :20], np.repeat(s, 3, 0).repeat(10, 1)[:, :20],
                          50, np.random.default_rng(0))
    assert np.isclose(tab["by_round"]["signed"]["mean"][0], 0.0)


def test_primary_flag_needs_the_frozen_arm():
    """Independent review M2: same channel, size, graph and memory is not enough."""
    from types import SimpleNamespace

    from swarm_audit.analysis import is_primary

    def mk(arm, arm_id, **kw):
        d = dict(channel="cheap_talk", n=10, graph="cycle")
        d.update(kw)
        return SimpleNamespace(meta={"design": {"arm": arm, "arm_id": arm_id, "last_k": None}}, **d)
    assert is_primary(mk("core_cheap_talk", 6))
    assert not is_primary(mk("olmo_core_cheap_talk", 91))
    assert not is_primary(mk("core_cheap_talk", 6, n=20))
    assert not is_primary(mk("core_cheap_talk", 6, channel="counts"))
