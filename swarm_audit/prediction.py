"""Composed predictions: the audit tables wired along G into Markov chains.

- `ActionsChain`: the order-1 chain on the action profile, every agent drawing H with
  kappa(H | own last action, in-neighbours bidding high last round); round 1 uses START.
- `SignalChain` (cheap talk): the chain on the action-and-signal profile, every agent
  drawing (a, m) from kappa'(a, m | s'). This is an autonomous prediction.
- `Order0`: every agent bids H with probability kappa_0(H), independently.

Each `simulate` returns actions as an (R, T, n) int8 array (1 = H), the same layout as
swarm.RunData.actions, so the statistics code treats swarm and model alike.

`exact_law` propagates a chain exactly on all profiles (2^n states for actions, 6^n for
action-and-signal), with START at round 1: used at n = 4 (16 or 1,296 states).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Mapping

import numpy as np

from .game import SIGNALS
from .states import START, key
from .topology import Graph

WINDOW = 5  # y counts high bids in the last 5 rounds (16-20 for T = 20)


def _actions_array(lookup: Mapping[str, float], d: int) -> tuple[np.ndarray, float]:
    """K[own, n_high] with own 0 = H, 1 = L; and P(H | START)."""
    K = np.full((2, d + 1), np.nan)
    for oi, a in enumerate("HL"):
        for k in range(d + 1):
            K[oi, k] = lookup[key((a, k))]
    return K, float(lookup[key(START)])


def _signal_array(lookup: Mapping[str, np.ndarray], d: int, own_signal: bool
                  ) -> tuple[np.ndarray, np.ndarray]:
    """J[own, n_high, s_high, s_low, own_m] -> 6-vector of K(a, m) (a-major), and START."""
    J = np.full((2, d + 1, d + 1, d + 1, 3, 6), np.nan)
    for oi, a in enumerate("HL"):
        for k in range(d + 1):
            for sh in range(d + 1):
                for sl in range(d + 1 - sh):
                    sn = d - sh - sl
                    for mi, m in enumerate(SIGNALS):
                        s = (a, k, sh, sl, sn) + ((m,) if own_signal else ())
                        J[oi, k, sh, sl, mi] = np.asarray(lookup[key(s)]).reshape(6)
    return J, np.asarray(lookup[key(START)]).reshape(6)


def _draw(probs: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Categorical draws: probs (..., m), u (...) uniform -> index (...)."""
    c = np.cumsum(probs, axis=-1)
    c = c / c[..., -1:]
    return (u[..., None] >= c).sum(-1).clip(max=probs.shape[-1] - 1)


@dataclass
class ActionsChain:
    lookup: Mapping[str, float]   # state key -> kappa(H | s)
    d: int
    name: str = "actions_only"

    def __post_init__(self) -> None:
        self.K, self.p_start = _actions_array(self.lookup, self.d)

    def simulate(self, g: Graph, T: int, R: int, rng: np.random.Generator) -> np.ndarray:
        if g.regular_degree != self.d:
            raise ValueError(f"table has d = {self.d}, graph has {g.regular_degree}")
        A = np.zeros((R, T, g.n), np.int8)
        A[:, 0] = rng.random((R, g.n)) < self.p_start
        adj = g.adjacency.T
        for t in range(1, T):
            prev = A[:, t - 1].astype(np.int64)
            p = self.K[1 - prev, prev @ adj]
            A[:, t] = rng.random((R, g.n)) < p
        return A

    def kernel(self, own: np.ndarray | None, n_high: np.ndarray | None):
        """P(H) per agent given own last actions and counts (None: START)."""
        if own is None:
            return np.full(1, self.p_start)
        return self.K[1 - own, n_high]


@dataclass
class SignalChain:
    lookup: Mapping[str, np.ndarray]   # state key -> 2 x 3 joint kappa'(a, m | s')
    d: int
    own_signal: bool = True
    name: str = "signal_aware"

    def __post_init__(self) -> None:
        self.J, self.p_start = _signal_array(self.lookup, self.d, self.own_signal)

    def simulate(self, g: Graph, T: int, R: int, rng: np.random.Generator,
                 return_signals: bool = False):
        if g.regular_degree != self.d:
            raise ValueError(f"table has d = {self.d}, graph has {g.regular_degree}")
        O = np.zeros((R, T, g.n), np.int8)   # output index a * 3 + m, a: 0 = H
        O[:, 0] = _draw(np.broadcast_to(self.p_start, (R, g.n, 6)), rng.random((R, g.n)))
        adj = g.adjacency.T
        for t in range(1, T):
            prev = O[:, t - 1].astype(np.int64)
            a, m = prev // 3, prev % 3
            k = (a == 0).astype(np.int64) @ adj
            sh = (m == 0).astype(np.int64) @ adj
            sl = (m == 1).astype(np.int64) @ adj
            O[:, t] = _draw(self.J[a, k, sh, sl, m], rng.random((R, g.n)))
        A = (O // 3 == 0).astype(np.int8)
        return (A, (O % 3).astype(np.int8)) if return_signals else A


@dataclass
class Order0:
    p: float
    name: str = "order0"

    def simulate(self, g: Graph, T: int, R: int, rng: np.random.Generator) -> np.ndarray:
        return (rng.random((R, T, g.n)) < self.p).astype(np.int8)


# ------------------------------------------------------------------------ exact law


def _actions_kernel_fn(chain: ActionsChain, g: Graph) -> tuple[int, Callable, np.ndarray]:
    adj = g.adjacency

    def per_agent(x: tuple) -> np.ndarray:
        own = np.array([1 - v for v in x])            # 0 = H, so own index = 1 - (a == H)
        k = adj @ np.array(x)
        p = chain.K[own, k]
        return np.stack([p, 1 - p], -1)               # outputs: 0 = H, 1 = L
    start = np.tile([chain.p_start, 1 - chain.p_start], (g.n, 1))
    return 2, per_agent, start


def _signal_kernel_fn(chain: SignalChain, g: Graph) -> tuple[int, Callable, np.ndarray]:
    adj = g.adjacency

    def per_agent(x: tuple) -> np.ndarray:
        o = np.array(x)
        a, m = o // 3, o % 3
        k, sh, sl = adj @ (a == 0), adj @ (m == 0), adj @ (m == 1)
        return chain.J[a, k, sh, sl, m]
    return 6, per_agent, np.tile(chain.p_start, (g.n, 1))


def exact_law(chain, g: Graph, T: int, window: int = WINDOW) -> dict:
    """Exact laws of X_t (number of agents bidding H in round t, t = 1..T) and of y (high
    bids in the last `window` rounds), by propagating the chain on all profiles.

    Profiles are tuples of per-agent outputs: 1 = H / 0 = L for the actions chain (so the
    tuple is the action indicator), or a * 3 + m for the signal chain (a = 0 is H)."""
    if isinstance(chain, ActionsChain):
        _, per_agent, start = _actions_kernel_fn(chain, g)
        profiles = list(product([1, 0], repeat=g.n))
        is_high = np.array([sum(x) for x in profiles])

        def agent_vec(x):
            return per_agent(x)                        # columns: [P(H), P(L)]

        def col(o):                                    # profile entry -> column of agent_vec
            return 0 if o == 1 else 1
    elif isinstance(chain, SignalChain):
        _, per_agent, start = _signal_kernel_fn(chain, g)
        profiles = list(product(range(6), repeat=g.n))
        is_high = np.array([sum(o // 3 == 0 for o in x) for x in profiles])

        def agent_vec(x):
            return per_agent(x)

        def col(o):
            return o
    else:
        raise TypeError("exact_law needs an ActionsChain or a SignalChain")
    S = len(profiles)
    cols = np.array([[col(o) for o in x] for x in profiles])     # (S, n)

    def prob_of_all(vecs: np.ndarray) -> np.ndarray:
        """P(next profile = y) for every y, given per-agent output vectors (n, m)."""
        return np.prod(vecs[np.arange(g.n), cols], axis=1)

    P = np.empty((S, S))
    for xi, x in enumerate(profiles):
        P[xi] = prob_of_all(agent_vec(x))
    v = prob_of_all(start)
    X = np.zeros((T, g.n + 1))
    t0 = T - window + 1
    yd = None
    for t in range(1, T + 1):
        if t > 1:
            if yd is not None:
                yd = P.T @ yd
                shifted = np.zeros_like(yd)
                C = yd.shape[1]
                for h in range(g.n + 1):          # add this round's high bids to the count
                    sel = is_high == h
                    shifted[sel, h:] = yd[sel, :C - h]
                yd = shifted
            v = v @ P
        X[t - 1] = np.bincount(is_high, weights=v, minlength=g.n + 1)
        if t == t0:
            yd = np.zeros((S, window * g.n + 1))
            yd[np.arange(S), is_high] = v
    y = yd.sum(0)
    return {"X": X, "y": y, "n_profiles": S, "window": window}


def sample_from_law(p: np.ndarray, size, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(len(p), size=size, p=p / p.sum())
