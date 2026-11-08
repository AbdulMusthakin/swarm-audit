"""The analysis of one condition (hand-off section 7), backend-agnostic.

Inputs: the logged swarm runs (swarm.RunData), the graph, and the audit tables. The
predictors are simulated for `n_model` runs each (exact laws as well when n is small);
everything is computed from logged probabilities, never from re-queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import stats
from .audit import AuditTable, PTest, state_weights
from .baselines import fit_from_swarm, order0_chain, split_runs
from scipy.stats import binom

from .prediction import WINDOW, ActionsChain, Order0, SignalChain, exact_law
from .states import parse
from .swarm import RunData
from .topology import Graph

ROUND_BLOCKS = ((2, 7), (8, 13), (14, 20))

# Review C6: what each number is, saved with every analysis.
NOTES = {
    "primary_contrast": "mean paired CRPS(actions-only) - CRPS(signal-aware) of y (high bids "
                        "in rounds 16-20, normalised by 5n) on the cheap-talk n = 10 cycle "
                        "runs with full history; positive favours the signal-aware forecast. "
                        "Absolute CRPS of both is reported; n = 20 and other arms are secondary.",
    "intervals": "95% percentile intervals from 2,000 resamples of whole swarm runs; they are "
                 "conditional on the fitted audit tables (audit-estimation uncertainty is "
                 "not included) and treat the model side as fixed.",
    "held_out": "the fitted-from-swarm baseline is fitted on runs 1-25 and every predictor is "
                "scored on the same runs 26-50; it uses swarm data, so it is not an audit.",
    "delta_and_null": "Delta_t is the binned TV between the swarm's R runs and the predictor's "
                      "fixed reference law (exact at n = 4, else 20,000 simulated runs). Its "
                      "null reference draws R whole runs from a separate simulation pool "
                      "against the same law; per_round is pointwise, max_over_rounds "
                      "simultaneous. Empirical TV has sampling error.",
    "bound": "B_t is an estimate of the Theorem 1 bound from logged defects with a whole-run "
             "bootstrap interval; it is not a certified bound.",
    "defect_shift": "decomposes the local expected one-step error eps(p_swarm) - eps(p_test), "
                    "not the observed group TV gap; both terms can be negative.",
    "defects": "actions_only is a binary-action TV; signal_aware is a six-outcome joint TV; "
               "compare them only through one_step (action marginal).",
}
EXACT_MAX_N = 4


@dataclass
class Tables:
    """Audit tables of one channel. On cheap talk `actions` is the derived table (or a
    directly fitted one, e.g. on K_10) and `signal` the signal-aware table."""

    actions: AuditTable
    signal: AuditTable | None = None
    actions_equal: AuditTable | None = None   # derived with equal weights (sensitivity)


@dataclass
class ModelRuns:
    name: str
    A: np.ndarray                      # the pool that estimates the reference law
    exact: dict | None = None
    extra: dict = field(default_factory=dict)
    null_pool: np.ndarray | None = None  # a separate pool for the null draws (review C4)


def build_predictors(tables: Tables, g: Graph, T: int, n_model: int, seed: int,
                     exact: bool = True, pt: PTest = PTest()) -> dict[str, ModelRuns]:
    rng = np.random.default_rng(seed)
    d = g.regular_degree
    chains = {"order0": order0_chain(tables.actions, pt),
              "actions_only": ActionsChain(tables.actions.lookup(), d)}
    if tables.actions_equal is not None:
        chains["actions_only_equal_w"] = ActionsChain(tables.actions_equal.lookup(), d)
    if tables.signal is not None:
        chains["signal_aware"] = SignalChain(tables.signal.lookup(), d, tables.signal.own_signal)
    out = {}
    for name, ch in chains.items():
        ex = None
        if isinstance(ch, Order0):        # independent agents: binomial laws, any n
            ex = {"X": np.tile(binom.pmf(np.arange(g.n + 1), g.n, ch.p), (T, 1)),
                  "y": binom.pmf(np.arange(WINDOW * g.n + 1), WINDOW * g.n, ch.p)}
        elif exact and g.n <= EXACT_MAX_N:
            ex = exact_law(ch, g, T)
        out[name] = ModelRuns(name, ch.simulate(g, T, n_model, rng), ex,
                              null_pool=ch.simulate(g, T, n_model, rng))
    return out


def _x_band(A: np.ndarray, B: int, rng) -> dict:
    X = stats.frac_high(A)
    idx = rng.integers(len(X), size=(B, len(X)))
    bs = X[idx].mean(axis=1)
    return {"mean": X.mean(0), "lo": np.quantile(bs, 0.025, 0), "hi": np.quantile(bs, 0.975, 0)}


def score_predictors(data: RunData, preds: dict[str, ModelRuns], B: int, rng) -> dict:
    """CRPS per run for each predictor, mean errors in F, Delta_t with its noise floor."""
    n, T = data.n, data.T
    K = stats.WINDOW * n
    y = stats.high_count(data.actions)
    F = stats.final_split(data.actions)
    Xs = stats.frac_high(data.actions)
    res: dict = {"crps": {}, "crps_per_run": {}, "F_error": {}, "delta": {}, "floor": {},
                 "G": {}}
    for name, m in preds.items():
        if m.exact is not None:
            G = stats.cdf_from_law(m.exact["y"])
            law = stats.binned_law_exact(m.exact["X"])
            mean_F = float((np.arange(K + 1) * m.exact["y"]).sum() / K)
            ref = "exact law"
        else:
            G = stats.cdf_from_samples(stats.high_count(m.A), K)
            law = stats.binned_law(stats.frac_high(m.A))
            mean_F = float(stats.final_split(m.A).mean())
            ref = f"simulation of {len(m.A)} runs"
        pool = m.null_pool if m.null_pool is not None else m.A
        c = stats.crps(G, y)
        res["G"][name] = G
        res["crps_per_run"][name] = c
        res["crps"][name] = stats.bootstrap_mean(c, B, rng)
        res["F_error"][name] = stats.bootstrap_diff_of_means(F, mean_F, B, rng)
        res["delta"][name] = stats.delta_t(Xs, law)
        res["floor"][name] = stats.null_reference(stats.frac_high(pool), law, data.R,
                                                  rng=rng, reference=ref)
    if "actions_only" in preds and "signal_aware" in preds:
        D = res["crps_per_run"]["actions_only"] - res["crps_per_run"]["signal_aware"]
        res["paired_D"] = stats.bootstrap_mean(D, B, rng)
    return res


def is_primary_contrast(channel: str, n: int, graph: str, design: dict) -> bool:
    """The primary contrast is the core's Llama arm 6 (cheap talk, n = 10, cycle, full
    history) only. Independent review M2 (8 Nov): other arms with the same channel, n,
    graph and memory (e.g. OLMo arm 91, exploratory) must not inherit the label, so the
    frozen arm identity from the run provenance is part of the test."""
    return (channel == "cheap_talk" and n == 10 and graph == "cycle"
            and design.get("last_k", "missing") is None
            and design.get("arm") == "core_cheap_talk" and design.get("arm_id") == 6)


def analyse_condition(data: RunData, g: Graph, tables: Tables, n_model: int = 20000,
                      seed: int = 0, B: int = 2000, pt: PTest = PTest(),
                      swarm_baseline: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    T = data.T
    preds = build_predictors(tables, g, T, n_model, seed + 1, pt=pt)
    design = data.meta.get("design", {})
    primary = is_primary_contrast(data.channel, data.n, data.graph, design)
    out: dict = {"condition": {"channel": data.channel, "n": data.n, "graph": data.graph,
                               "T": T, "runs": data.R, "last_k": design.get("last_k"),
                               "run_fingerprint": data.meta.get("fingerprint"),
                               "kernel_fingerprint": data.meta.get("kernel_fingerprint")},
                 "is_primary_contrast": primary,
                 "notes": NOTES}
    out["scores"] = score_predictors(data, preds, B, rng)
    out["X"] = {"swarm": _x_band(data.actions, B, rng),
                **{k: {"mean": stats.frac_high(m.A).mean(0)} for k, m in preds.items()}}
    out["exact"] = {k: m.exact for k, m in preds.items() if m.exact is not None}

    # Edge disagreement and consensus.
    out["edge_disagreement"] = {"swarm": stats.edge_disagreement(data.actions, g).mean(0),
                                **{k: stats.edge_disagreement(m.A, g).mean(0)
                                   for k, m in preds.items()}}
    cons = {"swarm": stats.consensus(data.actions),
            **{k: stats.consensus(m.A) for k, m in preds.items()}}
    out["consensus"] = {k: {"prob": c.prob(), "survival": c.survival()} for k, c in cons.items()}

    # Defects on the swarm's own histories (logged probabilities), and the bound.
    kap = tables.actions.lookup()
    dfx = stats.swarm_defects(data.p_H, data.state, kap)
    out["defects"] = {"actions_only": {
        "measure": "binary-action TV, |p_H - kappa(H | s)|",
        "eps_swarm_round_agent": dfx.mean(0), "eps_swarm": float(dfx.mean()),
        "eps_test_pooled": tables.actions.eps_pooled(pt),
        "eps_test_state": tables.actions.eps_state(),
        "bound": stats.bound(dfx, B, rng)}}
    out["defect_shift"] = {"actions_only": _shift(dfx, data.state, tables.actions, pt)}
    if tables.signal is not None:
        kj = tables.signal.lookup()
        dfj = stats.swarm_defects_joint(data.joint(), data.state_sig, kj)
        out["defects"]["signal_aware"] = {
            "measure": "six-outcome joint TV over (bid, message); not comparable with the "
                       "binary measure: use one_step for a like-for-like comparison",
            "eps_swarm_round_agent": dfj.mean(0), "eps_swarm": float(dfj.mean()),
            "eps_test_pooled": tables.signal.eps_pooled(pt),
            "eps_test_state": tables.signal.eps_state(), "bound": stats.bound(dfj, B, rng)}
        out["defect_shift"]["signal_aware"] = _shift(dfj, data.state_sig, tables.signal, pt)
        # One-step (teacher-forced) comparison of P(H): signal-aware vs actions-only.
        kH = {k: float(np.sum(np.asarray(v)[0])) for k, v in kj.items()}
        one_sig = stats.swarm_defects(data.p_H, data.state_sig, kH)
        out["one_step"] = {"label": "one-step (teacher-forced), not a group prediction; "
                                    "both use the action marginal, so they are like for like",
                           "mean_abs_pH_signal_aware": float(one_sig.mean()),
                           "mean_abs_pH_actions_only": float(dfx.mean())}

    # Shared-neighbour check.
    classes = stats.pair_classes(g)
    sn: dict = {}
    for cls, pairs in classes.items():
        ag = stats.agreement(data.actions, pairs)
        sn[cls] = {"swarm": _x_band_values(ag, B, rng),
                   **{k: stats.agreement(m.A, pairs).mean(0) for k, m in preds.items()}}
    out["shared_neighbour"] = sn

    # Fitted-from-swarm baseline: fit on runs 1-25, score everything on runs 26-50.
    if swarm_baseline and data.R >= 4:
        fit_idx, test_idx = split_runs(data)
        sf = fit_from_swarm(data.subset(fit_idx), g.regular_degree)
        held = data.subset(test_idx)
        p2 = dict(preds)
        ch = sf.chain(g.regular_degree)
        rng2 = np.random.default_rng(seed + 2)
        p2["fitted_from_swarm"] = ModelRuns(
            "fitted_from_swarm", ch.simulate(g, T, n_model, rng2),
            exact_law(ch, g, T) if g.n <= EXACT_MAX_N else None,
            null_pool=ch.simulate(g, T, n_model, rng2))
        sc = score_predictors(held, p2, B, rng)
        out["held_out"] = {"note": "runs 26-50 only; fitted_from_swarm uses swarm data "
                                   "(runs 1-25), so it is not an audit",
                           "runs": held.R, "crps": sc["crps"], "F_error": sc["F_error"],
                           "paired_D": sc.get("paired_D"), "missing_states": sf.missing,
                           "state_counts": sf.counts}
    return out


def _x_band_values(v: np.ndarray, B: int, rng) -> dict:
    idx = rng.integers(len(v), size=(B, len(v)))
    bs = v[idx].mean(axis=1)
    return {"mean": v.mean(0), "lo": np.quantile(bs, 0.025, 0), "hi": np.quantile(bs, 0.975, 0)}


def _shift(defects: np.ndarray, states: np.ndarray, table: AuditTable, pt: PTest) -> dict:
    sts = [parse(k) for k in table.entries]
    e_Q = table.eps_state()
    res = {"pooled": stats.defect_shift(defects, states, e_Q, state_weights(sts, table.d, pt))}
    w_later = state_weights(sts, table.d, pt, include_start=False)
    for a, b in ROUND_BLOCKS:
        res[f"rounds_{a}-{b}"] = stats.defect_shift(defects, states, e_Q, w_later, (a, b))
    return res


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x
