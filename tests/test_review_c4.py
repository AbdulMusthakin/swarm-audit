"""Review C4: the null reference matches the observed comparison (R runs vs a fixed law)."""

import numpy as np

from swarm_audit import stats
from swarm_audit.prediction import ActionsChain
from swarm_audit.states import START, all_actions_states, key
from swarm_audit.topology import cycle


def test_deterministic_matching_laws_give_zero():
    X = np.ones((30, 20))
    law = stats.binned_law(X)
    assert np.all(stats.delta_t(X, law) == 0)
    ref = stats.null_reference(np.ones((500, 20)), law, R=30, draws=50)
    assert ref["max_over_rounds"] == 0 and np.all(ref["per_round"] == 0)


def test_simultaneous_threshold_has_about_five_percent_rejection_under_the_null():
    lk = {key(s): (0.5 if s == START else 0.2 + 0.3 * s[1]) for s in all_actions_states(2)}
    ch, g, R = ActionsChain(lk, 2), cycle(10), 50
    rng = np.random.default_rng(0)
    law = stats.binned_law(stats.frac_high(ch.simulate(g, 20, 20000, rng)))       # reference
    ref = stats.null_reference(stats.frac_high(ch.simulate(g, 20, 20000, rng)), law, R,
                               draws=1000, rng=rng)                               # null pool
    fresh = stats.frac_high(ch.simulate(g, 20, R * 400, rng)).reshape(400, R, 20)  # "swarms"
    rej = np.mean([stats.delta_t(x, law).max() > ref["max_over_rounds"] for x in fresh])
    assert 0.01 <= rej <= 0.11          # nominal 0.05; Monte Carlo error over 400 swarms
    assert ref["R"] == R and ref["null_pool_runs"] == 20000 and "definition" in ref


def test_two_sample_floor_was_wider_than_the_matched_null():
    """The superseded two-sample reference is wider, i.e. it was too lenient."""
    lk = {key(s): (0.5 if s == START else 0.2 + 0.3 * s[1]) for s in all_actions_states(2)}
    ch, g = ActionsChain(lk, 2), cycle(10)
    rng = np.random.default_rng(1)
    X = stats.frac_high(ch.simulate(g, 20, 20000, rng))
    law = stats.binned_law(X)
    old = stats.noise_floor(X, 50, draws=400, rng=rng)["max_over_rounds"]
    new = stats.null_reference(stats.frac_high(ch.simulate(g, 20, 20000, rng)), law, 50,
                               draws=400, rng=rng)["max_over_rounds"]
    assert new < old
