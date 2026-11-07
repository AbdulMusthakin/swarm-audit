"""Baselines: the order-0 audit, and a chain fitted to swarm data.

- Order-0: kappa_0(H) from the audit table (audit.order0), every agent independent.
- Fitted from the swarm: kappa_swarm(s) is the mean of the *logged* p_H over agent-rounds
  in actions-only state s in the fitting runs (runs 1-25 of the condition, i.e. indices
  0-24). It is scored on the other runs (26-50). This baseline uses swarm data, so it is
  not an audit. A state never visited in the fitting runs falls back to the mean logged
  p_H over all fitting agent-rounds with at least one past round (recorded in `missing`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audit import AuditTable, PTest, order0
from .prediction import ActionsChain, Order0
from .states import START, all_actions_states, key
from .swarm import RunData


def order0_chain(table: AuditTable, pt: PTest = PTest()) -> Order0:
    return Order0(order0(table, pt))


@dataclass
class SwarmFit:
    lookup: dict[str, float]
    counts: dict[str, int]
    missing: list[str] = field(default_factory=list)

    def chain(self, d: int) -> ActionsChain:
        return ActionsChain(self.lookup, d, name="fitted_from_swarm")


def fit_from_swarm(data: RunData, d: int) -> SwarmFit:
    lookup, counts, missing = {}, {}, []
    later = data.state[:, 1:].reshape(-1)
    fallback = float(data.p_H[:, 1:].mean())
    for s in all_actions_states(d):
        k = key(s)
        sel = data.state == k
        counts[k] = int(sel.sum())
        if counts[k]:
            lookup[k] = float(data.p_H[sel].mean())
        else:
            lookup[k] = fallback if s != START else float(data.p_H[:, 0].mean())
            missing.append(k)
    assert len(later) > 0
    return SwarmFit(lookup, counts, missing)


def split_runs(data: RunData, n_fit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the fitting runs (the first half by run index) and the scoring runs."""
    order = np.argsort(data.runs)
    n_fit = data.R // 2 if n_fit is None else n_fit
    return order[:n_fit], order[n_fit:]
