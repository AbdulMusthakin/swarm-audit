"""Coarse states of a history: actions-only s and signal-aware s'.

- Actions-only: s = pi_1(q(h)) = (own last action, number of in-neighbours who bid high
  last round). On the cycle (d = 2) that is 2 x 3 = 6 states; on K_10 (d = 9), 2 x 10.
- Signal-aware: s' = s plus the in-neighbours' signal counts last round (high, low, none),
  and, if agents see their own past messages, the agent's own last signal. On the cycle
  that is 6 x 6 = 36 states, or 108 with the own signal.
- START is the empty history of round 1.

States are tuples in code and strings in logs (`key`), e.g. "H-2" or "L-1-1-0-1-none".
"""

from __future__ import annotations

from itertools import product
from typing import Iterable

from .game import ACTIONS, SIGNALS
from .ladder import History, Record, pi_k, q

START: tuple = ("START",)
State = tuple


def actions_state(history: History) -> State:
    """s = pi_1(q(h)): (own last action, n_high last round), or START."""
    last = pi_k(tuple(q(r) for r in history), 1)
    return START if not last else (last[0].action, last[0].n_high)


def signal_state(history: History, own_signal: bool = True) -> State:
    """s': the actions-only state plus last round's signal counts (and own signal)."""
    if not history:
        return START
    r = history[-1]
    if not r.has_signals:
        raise ValueError("signal-aware state needs a cheap-talk record")
    s = (r.action, r.n_high, r.sig_high, r.sig_low, r.sig_none)
    return s + (r.signal,) if own_signal else s


def to_actions(s: State) -> State:
    """The actions-only part of a signal-aware state."""
    return START if s == START else s[:2]


def key(s: State) -> str:
    return "START" if s == START else "-".join(map(str, s))


def parse(k: str) -> State:
    if k == "START":
        return START
    parts = k.split("-")
    return (parts[0], *(int(x) for x in parts[1:5]), *parts[5:])


def sig_compositions(d: int) -> list[tuple[int, int, int]]:
    """All (high, low, none) counts over d neighbours, in a fixed order."""
    return [(h, l, d - h - l) for h in range(d + 1) for l in range(d + 1 - h)]


def all_actions_states(d: int, start: bool = True) -> list[State]:
    out = [(a, k) for a in ACTIONS for k in range(d + 1)]
    return ([START] if start else []) + out


def all_signal_states(d: int, own_signal: bool = True, start: bool = True) -> list[State]:
    out = []
    for a, k in product(ACTIONS, range(d + 1)):
        for c in sig_compositions(d):
            if own_signal:
                out.extend((a, k, *c, m) for m in SIGNALS)
            else:
                out.append((a, k, *c))
    return ([START] if start else []) + out


def record_for_state(s: State, d: int) -> Record:
    """The last-round record that realises state s (signal-aware states fix the signal
    part; an actions-only state gives a counts record)."""
    if s == START:
        raise ValueError("START has no record")
    a, k = s[0], s[1]
    if len(s) == 2:
        return Record(a, k, d - k)
    sh, sl, sn = s[2:5]
    own = s[5] if len(s) > 5 else None
    if own is None:
        raise ValueError("a signal state without the own signal does not fix the record")
    return Record(a, k, d - k, signal=own, sig_high=sh, sig_low=sl, sig_none=sn)


def keys(states: Iterable[State]) -> list[str]:
    return [key(s) for s in states]
