#!/usr/bin/env python3
"""How small per-step errors compound over persistence (four-agent OLMo setting, §4.3).

CPU only; reads the cached OLMo tables (results/extra/audit/olmo-3-7b,
cheap_talk_d2_enumerated). It decomposes each audit's probability of the final-five-round
all-high event into an all-high start and survival in the all-high state:

- start: every agent bids high in round 1, the same under both audits;
- actions-only survival: the all-high state is one table entry (H-2), so survival over
  rounds 2-20 is kappa(H | H-2) to the power 4 x 19 = 76 agent-rounds;
- signal-aware survival: exact, by propagating the 81 message profiles of four agents
  that stay all-high (the table conditions on each agent's own and its neighbours' last
  messages);

and compares start x survival with the exact event probabilities. It also counts the
signal-aware audit's situations by number of neighbours, for the report's note on why
audits compress the state.

    python scripts/report_persistence.py

Output: results/extra/followup/persistence.json
"""

from __future__ import annotations

import itertools
import json
import sys
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from swarm_audit import build  # noqa: E402
from swarm_audit.prediction import ActionsChain, SignalChain, exact_law  # noqa: E402
from swarm_audit.topology import cycle  # noqa: E402

OUT = REPO / "results/extra/followup/persistence.json"
TABLES = REPO / "results/extra/audit/olmo-3-7b"
N, T = 4, 20
STEPS = T - 1                     # transitions from round 1 to round 20


def signal_survival(chain: SignalChain, g) -> tuple[float, float, np.ndarray]:
    """P(all high in round 1), and P(all high in rounds 2..T | all high in round 1), exact.

    Output index a * 3 + m (a = 0 is high; m = 0 high, 1 low, 2 none). The state while all
    agents bid high is the profile of their last messages."""
    adj = g.adjacency            # adj[j, i] = 1 if j is an in-neighbour of i
    profiles = list(itertools.product(range(3), repeat=N))
    index = {p: i for i, p in enumerate(profiles)}
    start = np.array([np.prod([chain.p_start[m] for m in p]) for p in profiles])
    p_round1 = start.sum()
    M = np.zeros((len(profiles), len(profiles)))
    for p in profiles:
        per_agent = []
        for i in range(N):
            nb = [p[j] for j in range(N) if adj[j, i]]
            k = len(nb)                                   # every neighbour bid high
            sh, sl = sum(m == 0 for m in nb), sum(m == 1 for m in nb)
            per_agent.append(chain.J[0, k, sh, sl, p[i]][:3])   # P(high, message m')
        for q in profiles:
            M[index[p], index[q]] = np.prod([per_agent[i][q[i]] for i in range(N)])
    v = start / p_round1
    for _ in range(STEPS):
        v = v @ M
    return float(p_round1), float(v.sum()), M


def situations(d: int) -> dict:
    """Audit situations for an agent with d neighbours (plus the empty first-round history)."""
    actions = 2 * (d + 1)
    signal = 2 * (d + 1) * comb(d + 2, 2) * 3
    return {"actions_only": actions, "signal_aware": signal}


def main() -> None:
    tables = build.load_tables(TABLES, "cheap_talk_d2_enumerated")
    g = cycle(N)
    ao = ActionsChain(tables.actions.lookup(), g.regular_degree)
    sa = SignalChain(tables.signal.lookup(), g.regular_degree, tables.signal.own_signal)
    exact = {"actions_only": float(exact_law(ao, g, T)["y"][-1]),
             "signal_aware": float(exact_law(sa, g, T)["y"][-1])}

    p_high_start = float(ao.p_start)
    start_ao = p_high_start ** N
    keep_ao = float(ao.K[0, 2])                          # kappa(H | own H, 2 neighbours H)
    surv_ao = keep_ao ** (N * STEPS)
    start_sa, surv_sa, _ = signal_survival(sa, g)
    keep_sa = surv_sa ** (1 / (N * STEPS))                # effective per agent-round
    out = {
        "setting": "OLMo-3-7B, n = 4 cycle, cheap talk, last-round history, T = 20",
        "event": "every agent bids high in every round 16-20 (F = 1)",
        "agent_rounds": N * STEPS,
        "p_high_round1": p_high_start,
        "p_all_high_round1": {"actions_only": start_ao, "signal_aware": start_sa},
        "keep_high_per_agent_round": {"actions_only": keep_ao, "signal_aware_effective": keep_sa},
        "survival_rounds_2_20": {"actions_only": surv_ao, "signal_aware": surv_sa},
        "start_times_survival": {"actions_only": start_ao * surv_ao, "signal_aware": start_sa * surv_sa},
        "exact_P_F1": exact,
        "survival_ratio_signal_over_actions": surv_sa / surv_ao,
        "per_agent_round_gap": keep_sa - keep_ao,
        "situations_by_neighbours": {d: situations(d) for d in (2, 4, 9)},
        "note": "start x survival omits re-entry into all-high after an exit; the exact event "
                "probabilities include it",
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("p_high_round1", "p_all_high_round1",
                                          "keep_high_per_agent_round", "survival_rounds_2_20",
                                          "start_times_survival", "exact_P_F1",
                                          "survival_ratio_signal_over_actions",
                                          "situations_by_neighbours")}, indent=1))


if __name__ == "__main__":
    main()
