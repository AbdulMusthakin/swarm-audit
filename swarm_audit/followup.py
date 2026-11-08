"""Exploratory follow-up analyses (docs/overnight-core-plan.md, 8 Nov 2026).

Everything here was specified after the original results were seen: temporal (round-block)
CRPS, signed one-step errors, Monte Carlo and audit-table sensitivity, and a clock-indexed
audit whose table depends on the history-length block as well as the state. None of it
changes the original analysis (swarm_audit/analysis.py), whose outputs stay as published.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audit import AuditTable, PTest, derive_actions_table
from .baselines import order0_chain
from .prediction import ActionsChain, Order0, SignalChain, _actions_array, _draw, _signal_array
from .topology import Graph

ROUND_BLOCKS = ((2, 7), (8, 13), (14, 20))
LENGTH_BLOCKS = ((1, 6), (7, 12), (13, 19))
FOLLOWUP_SEED = 20261108


def seeded(*key: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([FOLLOWUP_SEED, *key])))


# ------------------------------------------------------------------ temporal scores


def round_crps(A_model: np.ndarray, A_swarm: np.ndarray) -> np.ndarray:
    """CRPS of the high count c_t = n X_t per (swarm run, round): (R, T).

    CRPS_{r,t} = (1/n) sum_{k=0}^{n} (G_t(k) - 1[c_{r,t} <= k])^2, G_t the model's CDF of c_t
    over {0..n} from its simulated runs (the endpoint score's normalisation, with K = n)."""
    n = A_swarm.shape[2]
    cm = A_model.sum(2)                                   # (R_m, T)
    cs = A_swarm.sum(2).astype(np.int64)                  # (R, T)
    k = np.arange(n + 1)
    G = (cm[:, :, None] <= k).mean(0)                     # (T, n+1)
    ind = cs[:, :, None] <= k                              # (R, T, n+1)
    return ((G[None] - ind) ** 2).sum(-1) / n


def round_crps_cdf(G: np.ndarray, A_swarm: np.ndarray) -> np.ndarray:
    """As round_crps, from a given CDF G (T, n+1) of the high count per round."""
    n = A_swarm.shape[2]
    cs = A_swarm.sum(2).astype(np.int64)
    ind = cs[:, :, None] <= np.arange(n + 1)
    return ((G[None] - ind) ** 2).sum(-1) / n


def order0_cdfs(p: float, n: int, T: int, window: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Exact CDFs for independent agents: per-round count (T, n+1) and y over {0..window n}."""
    from scipy.stats import binom

    Gx = np.tile(binom.cdf(np.arange(n + 1), n, p), (T, 1))
    Gy = binom.cdf(np.arange(window * n + 1), window * n, p)
    return Gx, Gy


def block_means(v: np.ndarray, blocks=ROUND_BLOCKS) -> dict[str, np.ndarray]:
    """Per-run average over the rounds of each block (rounds are 1-based)."""
    return {f"{a}-{b}": v[:, a - 1:b].mean(1) for a, b in blocks}


def boot_mean(v: np.ndarray, B: int, rng) -> dict:
    v = np.asarray(v, float)
    idx = rng.integers(len(v), size=(B, len(v)))
    bs = v[idx].mean(1)
    return {"mean": float(v.mean()), "lo": float(np.quantile(bs, 0.025)),
            "hi": float(np.quantile(bs, 0.975)), "n": int(len(v)), "B": B}


def boot_ratio(num: np.ndarray, den: np.ndarray, B: int, rng) -> dict:
    """sum(num) / sum(den) over runs, with whole-run resampling. num, den: (R, ...)."""
    idx = rng.integers(len(num), size=(B, len(num)))
    with np.errstate(invalid="ignore", divide="ignore"):
        est = num.sum(0) / den.sum(0)
        bs = num[idx].sum(1) / den[idx].sum(1)
    return {"mean": est, "lo": np.nanquantile(bs, 0.025, axis=0),
            "hi": np.nanquantile(bs, 0.975, axis=0), "decisions": den.sum(0)}


# ---------------------------------------------------------------- one-step errors


def kappa_H_of(table: AuditTable) -> dict[str, float]:
    return {k: table.kappa_H(k) for k in table.entries}


def one_step_errors(p_H: np.ndarray, states: np.ndarray, kH: dict[str, float]) -> np.ndarray:
    """Signed teacher-forced error p_H - kappa(H | s) per decision (R, T, n)."""
    lut = np.vectorize(kH.__getitem__, otypes=[float])
    return p_H - lut(states)


def error_tables(e: np.ndarray, coarse: np.ndarray, B: int, rng) -> dict:
    """Means of e and |e| by round, and by coarse state within each round block, with
    whole-run intervals (ratio estimators: decisions are pooled within a resampled run)."""
    R, T, n = e.shape
    out: dict = {"by_round": {}, "by_state_block": {}}
    ones = np.ones_like(e)
    for name, v in (("signed", e), ("absolute", np.abs(e))):
        out["by_round"][name] = boot_ratio(v.sum(2), ones.sum(2), B, rng)
    keys = sorted(k for k in np.unique(coarse) if k != "START")
    for a, b in ROUND_BLOCKS:
        blk = {}
        sl = slice(a - 1, b)
        for k in keys:
            m = (coarse[:, sl] == k)
            num = {nm: (v[:, sl] * m).sum((1, 2)) for nm, v in (("signed", e), ("absolute", np.abs(e)))}
            den = m.sum((1, 2)).astype(float)
            if den.sum() == 0:
                continue
            blk[k] = {nm: {kk: (float(vv) if np.ndim(vv) == 0 else vv)
                           for kk, vv in boot_ratio(x, den, B, rng).items()}
                      for nm, x in num.items()}
        out["by_state_block"][f"{a}-{b}"] = blk
    return out


# ------------------------------------------------------------- table resampling


def resample_table(t: AuditTable, rng: np.random.Generator) -> AuditTable:
    """Resample each state's fitting histories with replacement (each probability vector
    intact); kappa is their mean. Single-history states (START) are kept fixed."""
    entries = {}
    for k, e in t.entries.items():
        f = np.asarray(e["fit"], float)
        if len(f) <= 1:
            entries[k] = e
            continue
        m = f[rng.integers(len(f), size=len(f))].mean(0)
        entries[k] = {**e, "kappa": float(m) if t.kind == "actions" else m.tolist()}
    return AuditTable(t.channel, t.d, t.kind, t.source + "+resampled", t.own_signal, entries,
                      t.meta)


@dataclass
class ChainSet:
    """The original predictors of a channel built from (possibly resampled) tables."""

    chains: dict

    @classmethod
    def from_tables(cls, signal: AuditTable | None, actions: AuditTable | None, d: int,
                    pt: PTest) -> "ChainSet":
        """On cheap talk the actions-only tables are derived from `signal` (primary and
        equal weights), as in the original pipeline; on counts `actions` is the fitted
        table."""
        ch: dict = {}
        if signal is not None:
            act = derive_actions_table(signal, "primary")
            eq = derive_actions_table(signal, "equal")
            ch["order0"] = order0_chain(act, pt)
            ch["actions_only"] = ActionsChain(act.lookup(), d)
            ch["actions_only_equal_w"] = ActionsChain(eq.lookup(), d)
            ch["signal_aware"] = SignalChain(signal.lookup(), d, signal.own_signal)
        else:
            ch["order0"] = order0_chain(actions, pt)
            ch["actions_only"] = ActionsChain(actions.lookup(), d)
        return cls(ch)

    def simulate(self, g: Graph, T: int, R: int, rng) -> dict:
        """Simulated runs per predictor; order-zero is returned as its chain, because its
        laws are exact (binomial), as in the original analysis."""
        return {k: (c if isinstance(c, Order0) else c.simulate(g, T, R, rng))
                for k, c in self.chains.items()}


# --------------------------------------------------------------- clock-indexed audit


def length_block(L: int) -> int:
    for i, (a, b) in enumerate(LENGTH_BLOCKS):
        if a <= L <= b:
            return i
    raise ValueError(f"length {L} outside the blocks")


def clock_table(t: AuditTable, block: int) -> tuple[AuditTable, dict]:
    """The table fitted only on histories whose length is in `block`. Empty cells fall back
    to the pooled entry. Returns the table and per-state fitting counts / fallbacks."""
    lo, hi = LENGTH_BLOCKS[block]
    entries, counts, fallback = {}, {}, []
    for k, e in t.entries.items():
        if len(e["fit"]) != len(e["fit_L"]):
            raise ValueError(f"{k}: fit and fit_L do not align")
        if k == "START":
            entries[k] = e
            continue
        f = np.asarray(e["fit"], float)
        L = np.asarray(e["fit_L"])
        sel = (L >= lo) & (L <= hi)
        counts[k] = int(sel.sum())
        sc = np.asarray(e["score"], float)
        sL = np.asarray(e["score_L"])
        ssel = (sL >= lo) & (sL <= hi)
        if sel.sum() == 0:
            fallback.append(k)
            new = dict(e)
        else:
            m = f[sel].mean(0)
            new = {**e, "kappa": float(m) if t.kind == "actions" else m.tolist(),
                   "fit": f[sel].tolist(), "fit_L": L[sel].tolist(), "n": int(sel.sum())}
        new["score"], new["score_L"] = sc[ssel].tolist(), sL[ssel].tolist()
        new.pop("eps", None)
        entries[k] = new
    out = AuditTable(t.channel, t.d, t.kind, f"{t.source}|L{lo}-{hi}", t.own_signal, entries,
                     {**t.meta, "length_block": [lo, hi]})
    return out, {"fit_counts": counts, "fallback_states": fallback}


def restrict_scoring(t: AuditTable, block: int) -> AuditTable:
    """The pooled table, with its scoring set restricted to lengths in `block` (to compare
    eps(p_test) like for like with the clock table)."""
    lo, hi = LENGTH_BLOCKS[block]
    entries = {}
    for k, e in t.entries.items():
        if k == "START":
            entries[k] = e
            continue
        sc, sL = np.asarray(e["score"], float), np.asarray(e["score_L"])
        sel = (sL >= lo) & (sL <= hi)
        new = {**e, "score": sc[sel].tolist(), "score_L": sL[sel].tolist()}
        new.pop("eps", None)
        entries[k] = new
    return AuditTable(t.channel, t.d, t.kind, t.source, t.own_signal, entries, t.meta)


def eps_with_scores(t: AuditTable) -> dict[str, float]:
    """eps per state on whatever scoring set the entries carry; states with no scoring
    history are left out."""
    out = {}
    for k, e in t.entries.items():
        sc = np.asarray(e.get("score", []), float)
        if len(sc) == 0:
            continue
        if t.kind == "actions":
            out[k] = float(np.mean(np.abs(sc - e["kappa"])))
        else:
            kap = np.asarray(e["kappa"]).reshape(-1)
            out[k] = float(np.mean(0.5 * np.abs(sc.reshape(len(sc), -1) - kap).sum(1)))
    return out


class ClockChain:
    """A time-inhomogeneous composed chain: round t >= 2 uses the chain of the length block
    containing L = t - 1; round 1 uses START (identical in every block's table)."""

    def __init__(self, chains: list, name: str):
        self.chains, self.name = chains, name
        self.kind = "signal" if isinstance(chains[0], SignalChain) else "actions"
        starts = [np.asarray(c.p_start) for c in chains]
        if not all(np.allclose(s, starts[0]) for s in starts):
            raise ValueError("START differs between blocks")

    def chain_at(self, t_round: int):
        return self.chains[length_block(t_round - 1)]

    def simulate(self, g: Graph, T: int, R: int, rng: np.random.Generator) -> np.ndarray:
        adj = g.adjacency.T
        c0 = self.chains[0]
        if self.kind == "actions":
            A = np.zeros((R, T, g.n), np.int8)
            A[:, 0] = rng.random((R, g.n)) < c0.p_start
            for t in range(1, T):
                K = self.chain_at(t + 1).K
                prev = A[:, t - 1].astype(np.int64)
                A[:, t] = rng.random((R, g.n)) < K[1 - prev, prev @ adj]
            return A
        O = np.zeros((R, T, g.n), np.int8)
        O[:, 0] = _draw(np.broadcast_to(c0.p_start, (R, g.n, 6)), rng.random((R, g.n)))
        for t in range(1, T):
            J = self.chain_at(t + 1).J
            prev = O[:, t - 1].astype(np.int64)
            a, m = prev // 3, prev % 3
            k = (a == 0).astype(np.int64) @ adj
            sh = (m == 0).astype(np.int64) @ adj
            sl = (m == 1).astype(np.int64) @ adj
            O[:, t] = _draw(J[a, k, sh, sl, m], rng.random((R, g.n)))
        return (O // 3 == 0).astype(np.int8)


_ = (_actions_array, _signal_array)
