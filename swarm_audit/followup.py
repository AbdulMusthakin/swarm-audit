"""Overnight follow-up (docs/overnight-extra-plan.md): exact information ablations of a
last-round cheap-talk kernel, and counterfactual observation interventions.

Notation. An agent's last-round state is (a, k, c, o): own last action a (index 0 = H,
1 = L), k high neighbours, neighbour message counts c = (high, low, none), own last message
o (0 = high, 1 = low, 2 = none). The kernel K(. | a, k, c, o) is a 6-vector over the
agent's output (bid, message), bid-major: index = bid * 3 + message.

The kernel array J has shape (2, d+1, d+1, d+1, 3, 6) indexed [a, k, s_high, s_low, o];
s_none = d - s_high - s_low (entries with s_high + s_low > d are unused).

- Audit variants (E1) average the joint 6-vector over omitted parts of the state with
  p_test's weights (Multinomial(d; 1/3) for c, uniform for o) or, as a sensitivity, equal
  weights on the compositions of c. Every variant keeps the same 6-outcome output space
  and START, and evolves autonomously.
- Observation interventions (E2) keep J and change what the agent is shown: a list of
  (weight, transform) pairs maps the actual (c, o) to the observed one. A deterministic
  null is one pair; independent uniform replacement is a mixture.

Profiles are tuples of per-agent outputs in itertools.product order (agent 0 most
significant), as in prediction.exact_law. Nothing here queries a model.
"""

from __future__ import annotations

from itertools import product
from math import factorial
from typing import Callable, Sequence

import numpy as np

from .game import SIGNALS
from .states import START, key, sig_compositions
from .topology import Graph

VARIANTS = ("both", "own_only", "neighbours_only", "neither")
PROCESSES = ("unmodified", "own_null", "neighbour_null", "both_null")
NONE = 2                                   # index of the "none" message


def comp_weights(d: int, scheme: str = "primary") -> dict[tuple[int, int, int], float]:
    """w(c) over neighbour message compositions: p_test's Multinomial(d; 1/3) or equal."""
    cs = sig_compositions(d)
    if scheme == "primary":
        w = {c: factorial(d) / np.prod([factorial(x) for x in c]) / 3 ** d for c in cs}
    elif scheme == "equal":
        w = {c: 1 / len(cs) for c in cs}
    else:
        raise ValueError(scheme)
    assert abs(sum(w.values()) - 1) < 1e-12
    return w


def kernel_array(lookup, d: int) -> tuple[np.ndarray, np.ndarray]:
    """J[a, k, sh, sl, o] -> 6-vector, and the START 6-vector, from a signal-aware table
    lookup (state key -> 2 x 3 joint) with the own signal in the state."""
    J = np.full((2, d + 1, d + 1, d + 1, 3, 6), np.nan)
    for ai, a in enumerate("HL"):
        for k in range(d + 1):
            for (sh, sl, sn) in sig_compositions(d):
                for oi, o in enumerate(SIGNALS):
                    J[ai, k, sh, sl, oi] = np.asarray(lookup[key((a, k, sh, sl, sn, o))]).reshape(6)
    return J, np.asarray(lookup[key(START)], float).reshape(6)


def variant_array(J: np.ndarray, d: int, variant: str, scheme: str = "primary") -> np.ndarray:
    """The kernel array of an audit variant: the joint 6-vector averaged over the omitted
    parts of the state, broadcast back to the full index so every variant shares one
    profile space."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    out = J.copy()
    w = comp_weights(d, scheme)
    if variant in ("own_only", "neither"):            # average over c
        avg = sum(wc * J[:, :, c[0], c[1]] for c, wc in w.items())   # (2, d+1, 3, 6)
        for c in w:
            out[:, :, c[0], c[1]] = avg
    if variant in ("neighbours_only", "neither"):     # average over o, uniformly
        out = np.broadcast_to(out.mean(axis=4, keepdims=True), out.shape).copy()
    return out


Transform = Callable[[tuple[int, int, int], int], tuple[tuple[int, int, int], int]]


def own_null(c, o):
    return c, NONE


def neighbour_null(c, o):
    return (0, 0, sum(c)), o


def both_null(c, o):
    return (0, 0, sum(c)), NONE


def identity(c, o):
    return c, o


def observation(process: str) -> list[tuple[float, Transform]]:
    """The deterministic observation interventions of E2 as one-element mixtures."""
    return {"unmodified": [(1.0, identity)], "own_null": [(1.0, own_null)],
            "neighbour_null": [(1.0, neighbour_null)], "both_null": [(1.0, both_null)]}[process]


def uniform_observation(d: int, own: bool, neighbours: bool) -> list[tuple[float, Transform]]:
    """E2 sensitivity: each hidden message replaced independently by a uniform signal at
    each observation (own: 3 values; neighbours: each of the d messages uniform, so the
    counts are Multinomial(d; 1/3))."""
    owns = [(1 / 3, m) for m in range(3)] if own else [(1.0, None)]
    comps = list(comp_weights(d, "primary").items()) if neighbours else [(None, 1.0)]
    out = []
    for wo, m in owns:
        for c, wc in comps:
            def f(cc, oo, m=m, c=c):
                return (c if c is not None else cc), (m if m is not None else oo)
            out.append((wo * wc, f))
    return out


def profiles(n: int) -> np.ndarray:
    return np.array(list(product(range(6), repeat=n)), dtype=np.int64)


def agent_vectors(J: np.ndarray, g: Graph, obs: Sequence[tuple[float, Transform]] | None = None
                  ) -> np.ndarray:
    """V[x, i] = the 6-vector of agent i's next output given profile x (observation
    transforms applied to the agent's view of messages only; actions untouched)."""
    n, d = g.n, g.regular_degree
    obs = obs or [(1.0, identity)]
    Pr = profiles(n)
    a, m = Pr // 3, Pr % 3
    V = np.zeros((len(Pr), n, 6))
    for i, nb in enumerate(g.in_neighbours):
        nb = list(nb)
        k = (a[:, nb] == 0).sum(1)
        sh = (m[:, nb] == 0).sum(1)
        sl = (m[:, nb] == 1).sum(1)
        for x in range(len(Pr)):
            c = (int(sh[x]), int(sl[x]), d - int(sh[x]) - int(sl[x]))
            for w, f in obs:
                c2, o2 = f(c, int(m[x, i]))
                V[x, i] += w * J[a[x, i], k[x], c2[0], c2[1], o2]
    return V


def transition_matrix(V: np.ndarray) -> np.ndarray:
    """P[x, y] = prod_i V[x, i, y_i] (agents draw independently given the profile)."""
    S, n, _ = V.shape
    Pr = profiles(n)
    P = np.ones((S, S))
    for i in range(n):
        P *= V[:, i, :][:, Pr[:, i]]
    return P


def start_law(start: np.ndarray, n: int) -> np.ndarray:
    Pr = profiles(n)
    return np.prod(start[Pr], axis=1)


def propagate(P: np.ndarray, v1: np.ndarray, n: int, T: int, window: int = 5) -> dict:
    """Exact profile laws v_t (t = 1..T), E[X_t] and the law of y (high bids, last window)."""
    Pr = profiles(n)
    h = (Pr // 3 == 0).sum(1)
    K = window * n
    vs = [v1]
    for _ in range(1, T):
        vs.append(vs[-1] @ P)
    vs = np.stack(vs)                                     # (T, S)
    yd = np.zeros((len(Pr), K + 1))
    yd[np.arange(len(Pr)), h] = vs[T - window]
    for _ in range(T - window + 1, T):
        nxt = P.T @ yd
        yd = np.zeros_like(nxt)
        for c in range(n + 1):
            sel = h == c
            yd[sel, c:] = nxt[sel, :K + 1 - c]
    y = yd.sum(0)
    return {"v": vs, "y": y, "EX": vs @ h / n}


def law_summary(y: np.ndarray, EX: np.ndarray, n: int, window: int = 5) -> dict:
    K = window * n
    return {"P_F1": float(y[K]), "P_F0": float(y[0]), "E_F": float((np.arange(K + 1) * y).sum() / K),
            "y_law": y.tolist(), "EX_t": EX.tolist()}


def expected_excess_crps(Q: np.ndarray, G: np.ndarray) -> float:
    """E_{y ~ G}[CRPS(Q, y) - CRPS(G, y)] = sum_k (Q(k) - G(k))^2 / K for CDFs on 0..K."""
    Q, G = np.asarray(Q), np.asarray(G)
    return float(((Q - G) ** 2).sum() / (len(G) - 1))


def all_high(n: int) -> np.ndarray:
    return (profiles(n) // 3 == 0).all(1)
