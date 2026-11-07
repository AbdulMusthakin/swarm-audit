"""Per-agent unit tests: synthetic histories from p_test, tables kappa, defects eps(p_test).

Tables are fitted on synthetic histories shown to one agent alone; swarm data never
fits them (the fitted-from-swarm baseline lives in baselines.py).

p_test (hand-off section 5; parameters in the config, described in results/audit/README.md)
-----------------------------------------------------------------------------------------
A history has L past rounds, L uniform on {1, ..., T-1}; START (L = 0) is the single
empty-history prompt. In every round the agent's own action is uniform on {H, L} and each
in-neighbour's action is independently uniform, so n_high ~ Binomial(d, 1/2). On cheap
talk every message (the agent's own included) is uniform over the three signals, so the
neighbours' signal counts are Multinomial(d; 1/3, 1/3, 1/3).

The fitting and scoring sets for state s are draws from p_test conditioned on
pi_1(q(h)) = s: the last round is set to s and everything else is drawn as above (on
cheap talk, the last round's messages too when s is actions-only). As a distribution
over states, p_test gives START weight 1/T and every other state
(T-1)/T x 1/2 x Binom(d, 1/2)(n_high) [x Multinom(d; 1/3)(signal counts) x 1/3 own signal].

Tables
------
- actions-only kappa(H | s): mean of the logged p_H over the fitting histories of s.
- signal-aware kappa'(a, m | s'): mean of P(a | h) P(m | h, a).
- derived actions-only table on cheap talk: kappa(H | s) = sum_{s' -> s} w(s' | s)
  kappa'(H | s'), with w the probability that p_test gives the signal part of s' given s
  ("primary"), or equal weight on each signal configuration ("equal", the teammates'
  choice) as a sensitivity check.
- enumerated table (calibration arm, last round only): one query per state, because the
  prompt is a function of the state.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from math import comb, factorial
from pathlib import Path
from typing import Sequence

import numpy as np

from .game import ACTIONS, SIGNALS
from .ladder import History, Record
from .policies import Backend, Query
from .states import (START, State, all_actions_states, all_signal_states, key, parse,
                     record_for_state, sig_compositions, to_actions)


@dataclass(frozen=True)
class PTest:
    """Parameters of the synthetic history distribution."""

    T: int = 20
    L_min: int = 1
    L_max: int = 19
    p_own_H: float = 0.5
    p_neighbour_H: float = 0.5
    seed: int = 20261108

    @classmethod
    def from_config(cls, cfg: dict) -> "PTest":
        return cls(**{k: cfg[k] for k in ("T", "L_min", "L_max", "p_own_H", "p_neighbour_H",
                                          "seed") if k in cfg})


# ----------------------------------------------------------- synthetic histories


def random_record(rng: np.random.Generator, channel: str, d: int, pt: PTest) -> Record:
    a = "H" if rng.random() < pt.p_own_H else "L"
    n_high = int((rng.random(d) < pt.p_neighbour_H).sum())
    if channel == "counts":
        return Record(a, n_high, d - n_high)
    own = SIGNALS[rng.integers(3)]
    ms = rng.integers(3, size=d)
    return Record(a, n_high, d - n_high, signal=own, sig_high=int((ms == 0).sum()),
                  sig_low=int((ms == 1).sum()), sig_none=int((ms == 2).sum()))


def last_record(rng: np.random.Generator, channel: str, d: int, s: State) -> Record:
    """A last-round record in state s; parts of the record s leaves open are uniform."""
    a, k = s[0], s[1]
    if channel == "counts":
        if len(s) != 2:
            raise ValueError("counts states are actions-only")
        return Record(a, k, d - k)
    if len(s) == 2:
        ms = rng.integers(3, size=d)
        c = (int((ms == 0).sum()), int((ms == 1).sum()), int((ms == 2).sum()))
    else:
        c = tuple(s[2:5])
    own = s[5] if len(s) > 5 else SIGNALS[rng.integers(3)]
    return Record(a, k, d - k, signal=own, sig_high=c[0], sig_low=c[1], sig_none=c[2])


def synthetic_history(rng: np.random.Generator, channel: str, d: int, s: State,
                      pt: PTest) -> History:
    """A draw from p_test conditioned on the coarse state s."""
    if s == START:
        return ()
    L = int(rng.integers(pt.L_min, pt.L_max + 1))
    earlier = tuple(random_record(rng, channel, d, pt) for _ in range(L - 1))
    return earlier + (last_record(rng, channel, d, s),)


def _rng(pt: PTest, tag: str, set_id: int, idx: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [pt.seed, zlib.crc32(tag.encode()), set_id, idx])))


# ------------------------------------------------------------------------ weights


def _binom(d: int, k: int, p: float = 0.5) -> float:
    return comb(d, k) * p ** k * (1 - p) ** (d - k)


def _multinom(c: Sequence[int]) -> float:
    d = sum(c)
    return factorial(d) / np.prod([factorial(x) for x in c]) / 3 ** d


def state_weights(states: Sequence[State], d: int, pt: PTest = PTest(),
                  include_start: bool = True) -> dict[str, float]:
    """p_test's distribution over the given states (actions-only or signal-aware).
    With include_start=False, the distribution given L >= 1 (START excluded)."""
    w: dict[str, float] = {}
    for s in states:
        if s == START:
            if include_start:
                w[key(s)] = 1 / pt.T
            continue
        x = (pt.p_own_H if s[0] == "H" else 1 - pt.p_own_H) * _binom(d, s[1], pt.p_neighbour_H)
        if len(s) > 2:
            x *= _multinom(s[2:5])
        if len(s) > 5:
            x /= 3
        w[key(s)] = x * ((pt.T - 1) / pt.T if include_start else 1.0)
    z = sum(w.values())
    if abs(z - 1) > 1e-9:
        raise ValueError(f"state weights sum to {z}: incomplete state list")
    return w


def marginal_weights(d: int, own_signal: bool, scheme: str = "primary"
                     ) -> dict[str, dict[str, float]]:
    """w(s' | s) for every actions-only s: the signal-aware states s' that map to s.
    "primary": p_test's weights (multinomial counts, uniform own signal).
    "equal": equal weight on each signal configuration."""
    if scheme not in ("primary", "equal"):
        raise ValueError(scheme)
    out: dict[str, dict[str, float]] = {key(START): {key(START): 1.0}}
    for s in all_actions_states(d, start=False):
        ws = {}
        for c in sig_compositions(d):
            for m in (SIGNALS if own_signal else (None,)):
                sp = (*s, *c) + ((m,) if m else ())
                ws[key(sp)] = (_multinom(c) / (3 if m else 1)) if scheme == "primary" else 1.0
        z = sum(ws.values())
        out[key(s)] = {k: v / z for k, v in ws.items()}
    return out


# ------------------------------------------------------------------------- tables


@dataclass
class AuditTable:
    """kappa per state. entries[key] holds:
    kappa  : P(H) (actions) or the 2 x 3 joint K(a, m) as nested lists (signal)
    n, se  : number of fitting histories and the standard error of kappa
    fit    : per-history probabilities (p_H, or the 6-way joint) on the fitting set
    score  : the same on the scoring set
    fit_L, score_L : history lengths
    """

    channel: str
    d: int
    kind: str                     # "actions" or "signal"
    source: str                   # "p_test", "enumerated" or "derived:<scheme>"
    own_signal: bool = True
    entries: dict[str, dict] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def kappa_H(self, k: str) -> float:
        kap = self.entries[k]["kappa"]
        return float(kap) if self.kind == "actions" else float(np.sum(kap[0]))

    def kappa_joint(self, k: str) -> np.ndarray:
        if self.kind != "signal":
            raise ValueError("actions-only table has no joint")
        return np.asarray(self.entries[k]["kappa"])

    def eps_state(self) -> dict[str, float]:
        """eps(p_test) per state on the scoring set (TV distance; six-way on signal)."""
        out = {}
        for k, e in self.entries.items():
            if "eps" in e:
                out[k] = e["eps"]
                continue
            sc = np.asarray(e["score"])
            if self.kind == "actions":
                out[k] = float(np.mean(np.abs(sc - e["kappa"])))
            else:
                kap = np.asarray(e["kappa"]).reshape(-1)
                out[k] = float(np.mean(0.5 * np.abs(sc.reshape(len(sc), -1) - kap).sum(1)))
        return out

    def eps_pooled(self, pt: PTest = PTest()) -> float:
        w = state_weights([parse(k) for k in self.entries], self.d, pt)
        e = self.eps_state()
        return float(sum(w[k] * e[k] for k in self.entries))

    def lookup(self) -> dict:
        """Arrays for the chains: see prediction.py."""
        return {k: (self.kappa_H(k) if self.kind == "actions" else self.kappa_joint(k))
                for k in self.entries}

    def to_json(self) -> dict:
        return {"channel": self.channel, "d": self.d, "kind": self.kind, "source": self.source,
                "own_signal": self.own_signal, "meta": self.meta, "entries": self.entries}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json()) + "\n")

    @classmethod
    def load(cls, path: Path) -> "AuditTable":
        j = json.loads(Path(path).read_text())
        return cls(j["channel"], j["d"], j["kind"], j["source"], j["own_signal"],
                   j["entries"], j["meta"])


def _probs(outs, kind: str) -> list:
    if kind == "actions":
        return [o.p_H for o in outs]
    return [o.joint().tolist() for o in outs]


def _summ(x: list, kind: str) -> tuple:
    a = np.asarray(x, float)
    m = a.mean(0)
    se = (a.std(0, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else np.zeros_like(m)
    return (float(m), float(se)) if kind == "actions" else (m.tolist(), se.tolist())


def table_states(channel: str, d: int, kind: str, own_signal: bool) -> list[State]:
    if kind == "actions":
        return all_actions_states(d)
    if channel != "cheap_talk":
        raise ValueError("signal-aware tables are for the cheap-talk channel")
    return all_signal_states(d, own_signal)


def fit_table(backend: Backend, channel: str, d: int, kind: str, n_fit: int, n_score: int,
              pt: PTest = PTest(), own_signal: bool = True, batch: int = 4096,
              keep_histories: bool = False) -> AuditTable:
    """Fit kappa on p_test histories and keep a disjoint scoring set (different seeds)."""
    states = table_states(channel, d, kind, own_signal)
    tag = f"{channel}|{kind}|d{d}|own{int(own_signal)}"
    plan: list[tuple[str, str, History]] = []   # (state key, set, history)
    for idx, s in enumerate(states):
        for set_id, set_name, size in ((0, "fit", n_fit), (1, "score", n_score)):
            rng = _rng(pt, tag, set_id, idx)
            m = 1 if s == START else size
            plan.extend((key(s), set_name, synthetic_history(rng, channel, d, s, pt))
                        for _ in range(m))
    outs = []
    for c in range(0, len(plan), batch):
        outs.extend(backend.query([Query(channel, h, d) for _, _, h in plan[c:c + batch]]))
    entries: dict[str, dict] = {key(s): {"fit": [], "score": [], "fit_L": [], "score_L": [],
                                         "fit_sha": [], "score_sha": []} for s in states}
    for (k, set_name, h), o in zip(plan, outs):
        e = entries[k]
        e[set_name].append(_probs([o], kind)[0])
        e[f"{set_name}_L"].append(len(h))
        e[f"{set_name}_sha"].append(o.prompt_sha256)
    for k, e in entries.items():
        e["kappa"], e["se"] = _summ(e["fit"], kind)
        e["n"] = len(e["fit"])
        if not keep_histories:
            e.pop("fit_sha"), e.pop("score_sha")
    t = AuditTable(channel, d, kind, "p_test", own_signal, entries,
                   {"n_fit": n_fit, "n_score": n_score, "p_test": pt.__dict__,
                    "backend": backend.describe(), "queries": len(plan)})
    for k, ek in t.eps_state().items():
        entries[k]["eps"] = ek
    return t


def enumerated_table(backend: Backend, channel: str, d: int, kind: str,
                     own_signal: bool = True, last_k: int = 1) -> AuditTable:
    """Calibration arm: with last-round-only prompts, one query per state gives kappa
    exactly (the prompt is a function of the state). Each state's history is the single
    record that realises it."""
    states = table_states(channel, d, kind, own_signal)
    hists = [() if s == START else (record_for_state(s, d),) for s in states]
    outs = backend.query([Query(channel, h, d, last_k) for h in hists])
    entries = {}
    for s, o in zip(states, outs):
        p = _probs([o], kind)[0]
        entries[key(s)] = {"kappa": p, "se": 0.0 if kind == "actions" else
                           np.zeros((2, 3)).tolist(), "n": 1, "fit": [p], "score": [p],
                           "eps": 0.0, "prompt_sha256": o.prompt_sha256,
                           "n_tokens": o.n_tokens}
    return AuditTable(channel, d, kind, "enumerated", own_signal, entries,
                      {"last_k": last_k, "backend": backend.describe(), "queries": len(states)})


def derive_actions_table(sig: AuditTable, scheme: str = "primary") -> AuditTable:
    """Marginalise a signal-aware table over the signal part of the state, and score the
    derived kappa on the signal-aware scoring set with the same weights."""
    if sig.kind != "signal":
        raise ValueError("need a signal-aware table")
    W = marginal_weights(sig.d, sig.own_signal, scheme)
    entries = {}
    for s, ws in W.items():
        kap = sum(w * sig.kappa_H(sp) for sp, w in ws.items())
        eps = 0.0
        for sp, w in ws.items():
            sc = np.asarray(sig.entries[sp]["score"])
            p_h = sc.reshape(len(sc), 2, 3)[:, 0, :].sum(1)
            eps += w * float(np.mean(np.abs(p_h - kap)))
        var = 0.0  # SE of a weighted sum of independent state means of P(H)
        for sp, w in ws.items():
            f = np.asarray(sig.entries[sp]["fit"]).reshape(-1, 2, 3)[:, 0, :].sum(1)
            if len(f) > 1:
                var += w ** 2 * f.var(ddof=1) / len(f)
        entries[s] = {"kappa": float(kap), "n": sum(sig.entries[sp]["n"] for sp in ws),
                      "se": float(np.sqrt(var)), "eps": eps, "fit": [], "score": []}
    return AuditTable(sig.channel, sig.d, "actions", f"derived:{scheme}", sig.own_signal,
                      entries, {"from": sig.source, "scheme": scheme})


def order0(table: AuditTable, pt: PTest = PTest()) -> float:
    """kappa_0(H): the table's P(H) averaged over states with p_test's weights."""
    w = state_weights([parse(k) for k in table.entries], table.d, pt)
    return float(sum(w[k] * table.kappa_H(k) for k in table.entries))


assert ACTIONS == ("H", "L") and SIGNALS == ("high", "low", "none")
_ = to_actions  # re-exported for callers
