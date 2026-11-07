"""Trajectory statistics, scores, the gap Delta_t, defects and the Theorem 1 bound.

Action arrays are (R, T, n) with 1 = H (rounds 1..T at index 0..T-1), for the swarm
(swarm.RunData.actions) and for every predictor (prediction.*.simulate) alike.
Every interval resamples whole runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .topology import Graph

WINDOW = 5
BINS = np.array([0.0, 0.2, 0.4, 0.6, 0.8])  # [0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1]


# ------------------------------------------------------------------- per-run stats


def frac_high(A: np.ndarray) -> np.ndarray:
    """X_t per run: (R, T)."""
    return A.mean(axis=2)


def high_count(A: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """y: number of high bids in the last `window` rounds, out of window * n. (R,)"""
    return A[:, -window:].sum(axis=(1, 2)).astype(int)


def final_split(A: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """F = y / (window n): the mean of X over the last `window` rounds. (R,)"""
    return high_count(A, window) / (window * A.shape[2])


def edge_disagreement(A: np.ndarray, g: Graph) -> np.ndarray:
    """Fraction of undirected edges whose two ends bid differently, per round: (R, T)."""
    e = np.array(g.edges)
    return (A[:, :, e[:, 0]] != A[:, :, e[:, 1]]).mean(axis=2)


@dataclass
class Consensus:
    tau: np.ndarray          # first round t <= T-2 of consensus, or nan (censored at T)
    side: np.ndarray         # "H", "L" or "" (no consensus)
    T: int

    def prob(self) -> dict[str, float]:
        return {s: float(np.mean(self.side == s)) for s in ("H", "L", "")}

    def survival(self) -> np.ndarray:
        """P(tau > t) for t = 1..T-2; runs without consensus stay in the risk set."""
        t = np.arange(1, self.T - 1)
        tau = np.where(np.isnan(self.tau), np.inf, self.tau)
        return (tau[:, None] > t[None, :]).mean(axis=0)


def consensus(A: np.ndarray) -> Consensus:
    """First t <= T-2 with X_t = X_{t+1} = X_{t+2} in {0, 1}, with its side."""
    X = frac_high(A)
    R, T = X.shape
    tau = np.full(R, np.nan)
    side = np.full(R, "", dtype=object)
    for side_name, v in (("H", 1.0), ("L", 0.0)):
        run3 = (X[:, :-2] == v) & (X[:, 1:-1] == v) & (X[:, 2:] == v)   # (R, T-2)
        has = run3.any(axis=1)
        first = np.argmax(run3, axis=1) + 1.0
        better = has & (np.isnan(tau) | (first < tau))
        tau[better] = first[better]
        side[better] = side_name
    return Consensus(tau, side.astype(str), T)


# -------------------------------------------------------------------------- scores


def cdf_from_samples(y: np.ndarray, kmax: int) -> np.ndarray:
    """G(k) = P(y <= k) for k = 0..kmax."""
    return np.cumsum(np.bincount(y, minlength=kmax + 1)[: kmax + 1]) / len(y)


def cdf_from_law(p: np.ndarray) -> np.ndarray:
    return np.cumsum(p) / p.sum()


def crps(G: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
    """CRPS_r = (1 / K) sum_{k=0}^{K} (G(k) - 1[y_r <= k])^2 with K = 5n = len(G) - 1."""
    K = len(G) - 1
    k = np.arange(K + 1)
    ind = (np.asarray(y_obs)[:, None] <= k[None, :]).astype(float)
    return ((G[None, :] - ind) ** 2).sum(axis=1) / K


def bootstrap_mean(v: np.ndarray, B: int = 2000, rng: np.random.Generator | None = None,
                   level: float = 0.95) -> dict:
    """Mean of per-run values with a percentile interval from resampling runs."""
    rng = rng or np.random.default_rng(0)
    v = np.asarray(v, float)
    idx = rng.integers(len(v), size=(B, len(v)))
    bs = v[idx].mean(axis=1)
    a = (1 - level) / 2
    return {"mean": float(v.mean()), "lo": float(np.quantile(bs, a)),
            "hi": float(np.quantile(bs, 1 - a)), "n": len(v), "B": B}


def bootstrap_diff_of_means(v_swarm: np.ndarray, model_mean: float, B: int = 2000,
                            rng: np.random.Generator | None = None) -> dict:
    """E_swarm F - E_model F with a whole-run bootstrap of the swarm side (the model side
    is fixed: its 20,000 runs make its sampling noise negligible)."""
    r = bootstrap_mean(v_swarm, B, rng)
    return {k: (r[k] - model_mean if k in ("mean", "lo", "hi") else r[k]) for k in r}


# ------------------------------------------------------------------------ the gap


def bin_index(X: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(BINS, X, side="right") - 1, 0, len(BINS) - 1)


def binned_law(X: np.ndarray) -> np.ndarray:
    """Per-round law of X_t over the five bins: (T, 5)."""
    b = bin_index(X)
    return np.stack([np.bincount(b[:, t], minlength=len(BINS)) / len(b)
                     for t in range(X.shape[1])])


def binned_law_exact(PX: np.ndarray) -> np.ndarray:
    """Binned law from an exact law of the number of high bidders (T, n+1)."""
    n = PX.shape[1] - 1
    b = bin_index(np.arange(n + 1) / n)
    out = np.zeros((PX.shape[0], len(BINS)))
    for j in range(n + 1):
        out[:, b[j]] += PX[:, j]
    return out


def tv(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(p - q).sum(axis=-1)


def delta_t(X_swarm: np.ndarray, law_model: np.ndarray) -> np.ndarray:
    """Delta_t: TV between swarm and model laws of X_t, in five bins. (T,)"""
    return tv(binned_law(X_swarm), law_model)


def noise_floor(X_model: np.ndarray, R: int, draws: int = 1000, q: float = 0.95,
                rng: np.random.Generator | None = None) -> dict:
    """95th percentile of the binned TV between two independent samples of R runs from
    the model, per round, and of its maximum over rounds (a simultaneous floor)."""
    rng = rng or np.random.default_rng(1)
    M = len(X_model)
    b = bin_index(X_model)                          # (M, T)
    T = b.shape[1]
    d = np.empty((draws, T))
    for j in range(draws):
        i1, i2 = rng.integers(M, size=R), rng.integers(M, size=R)
        for t in range(T):
            p1 = np.bincount(b[i1, t], minlength=5) / R
            p2 = np.bincount(b[i2, t], minlength=5) / R
            d[j, t] = 0.5 * np.abs(p1 - p2).sum()
    return {"per_round": np.quantile(d, q, axis=0), "max_over_rounds": float(
        np.quantile(d.max(axis=1), q)), "q": q, "draws": draws, "R": R}


def ks_gap(y_obs: np.ndarray, G: np.ndarray) -> float:
    """Largest gap between the empirical CDF of y and G, over k = 0..K."""
    return float(np.abs(cdf_from_samples(np.asarray(y_obs), len(G) - 1) - G).max())


def ks_null_gaps(p_y: np.ndarray, R: int, sims: int = 20000,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """ks_gap for `sims` simulated sets of R runs drawn from the law p_y."""
    rng = rng or np.random.default_rng(2)
    G = cdf_from_law(p_y)
    counts = rng.multinomial(R, p_y / p_y.sum(), size=sims)            # (sims, K+1)
    emp = np.cumsum(counts, axis=1) / R
    return np.abs(emp - G[None, :]).max(axis=1)


def ks_threshold(p_y: np.ndarray, R: int, sims: int = 20000, q: float = 0.99,
                 rng: np.random.Generator | None = None) -> float:
    """q-quantile of ks_gap over `sims` simulated sets of R runs drawn from the law p_y."""
    return float(np.quantile(ks_null_gaps(p_y, R, sims, rng), q))


TIE_EPS = 1e-12


def mc_pvalue(observed: float, null: np.ndarray) -> float:
    """Discrete Monte Carlo p-value: (1 + #{null >= observed}) / (1 + len(null)).
    Ties count as at least as extreme, so a perfectly matched point mass (observed 0,
    every null value 0) gives p = 1 (review C5)."""
    return float((1 + np.sum(null >= observed - TIE_EPS)) / (1 + len(null)))


# ------------------------------------------------------------------------- defects


def swarm_defects(p_H: np.ndarray, states: np.ndarray, kappa: Mapping[str, float]) -> np.ndarray:
    """|p_H - kappa(H | s)| for every logged decision: (R, T, n)."""
    kap = np.vectorize(lambda k: kappa[k], otypes=[float])(states)
    return np.abs(p_H - kap)


def swarm_defects_joint(joint: np.ndarray, states: np.ndarray,
                        kappa: Mapping[str, np.ndarray]) -> np.ndarray:
    """Six-way TV between K(a, m | h) and kappa'(a, m | s') per decision: (R, T, n)."""
    R, T, n = states.shape
    kap = np.stack([np.asarray(kappa[k]).reshape(6) for k in states.reshape(-1)])
    return 0.5 * np.abs(joint.reshape(-1, 6) - kap).sum(axis=1).reshape(R, T, n)


def bound(eps_rtn: np.ndarray, B: int = 2000, rng: np.random.Generator | None = None) -> dict:
    """B_t = sum over rounds s <= t and agents i of eps_i(p_swarm(i, s)), where eps is the
    mean over runs; uncapped and capped at 1, its first saturation round, and a
    whole-run bootstrap interval."""
    rng = rng or np.random.default_rng(3)
    eps = eps_rtn.mean(axis=0)                     # (T, n)
    Bt = np.cumsum(eps.sum(axis=1))
    R = eps_rtn.shape[0]
    per_run = eps_rtn.sum(axis=2)                  # (R, T)
    idx = rng.integers(R, size=(B, R))
    bs = np.cumsum(per_run[idx].mean(axis=1), axis=1)
    sat = np.nonzero(Bt >= 1)[0]
    return {"B": Bt, "B_capped": np.minimum(Bt, 1.0), "lo": np.quantile(bs, 0.025, axis=0),
            "hi": np.quantile(bs, 0.975, axis=0),
            "first_saturation_round": int(sat[0] + 1) if len(sat) else None,
            "eps_round_agent": eps}


def defect_shift(defects: np.ndarray, states: np.ndarray, e_Q: Mapping[str, float],
                 w_Q: Mapping[str, float], rounds: tuple[int, int] | None = None) -> dict:
    """eps(p_swarm) - eps(p_test) = sum_s [w_P(s) - w_Q(s)] e_Q(s)
                                  + sum_s w_P(s) [e_P(s) - e_Q(s)].
    `rounds` = (first, last), 1-based inclusive; w_Q is then p_test given L >= 1 (pass
    the matching weights)."""
    if rounds is not None:
        sl = slice(rounds[0] - 1, rounds[1])
        defects, states = defects[:, sl], states[:, sl]
    st, df = states.reshape(-1), defects.reshape(-1)
    keys_ = sorted(set(w_Q) | set(st))
    w_P = {k: float(np.mean(st == k)) for k in keys_}
    e_P = {k: (float(df[st == k].mean()) if w_P[k] > 0 else 0.0) for k in keys_}
    frequency = sum((w_P[k] - w_Q.get(k, 0.0)) * e_Q[k] for k in keys_)
    within = sum(w_P[k] * (e_P[k] - e_Q[k]) for k in keys_ if w_P[k] > 0)
    eps_P = float(df.mean())
    eps_Q = float(sum(w_Q[k] * e_Q[k] for k in w_Q))
    return {"eps_swarm": eps_P, "eps_test": eps_Q, "difference": eps_P - eps_Q,
            "state_frequency_term": float(frequency), "within_state_term": float(within),
            "check": float(frequency + within - (eps_P - eps_Q)),
            "states_not_in_p_test": [k for k in keys_ if k not in w_Q and w_P[k] > 0]}


# ------------------------------------------------------------------ shared neighbours


def pair_classes(g: Graph) -> dict[str, list[tuple[int, int]]]:
    """Agent pairs by graph distance: 1 (adjacent), 2 (sharing a neighbour), >= 3."""
    out: dict[str, list] = {"1": [], "2": [], ">=3": []}
    for i in range(g.n):
        for j in range(i + 1, g.n):
            dist = g.distance(i, j)
            out["1" if dist == 1 else "2" if dist == 2 else ">=3"].append((i, j))
    return {k: v for k, v in out.items() if v}


def agreement(A: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Per run and round, the fraction of the pairs whose two agents bid alike: (R, T)."""
    p = np.array(pairs)
    return (A[:, :, p[:, 0]] == A[:, :, p[:, 1]]).mean(axis=2)
