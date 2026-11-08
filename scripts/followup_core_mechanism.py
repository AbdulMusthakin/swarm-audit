#!/usr/bin/env python3
"""Diagnostics behind the clock-indexed result (exploratory; docs/overnight-core-plan.md C2).

1. Teacher-forced signed one-step error by round, p_H - kappa(H | s), for the pooled and the
   clock-indexed tables (signal-aware action marginal and derived actions-only), on the
   swarm's own histories. If matching history length removes the early bias, the clock
   error is near 0 in rounds 2-7.
2. History content: for each round t, the fraction of high bids in the agent's earlier rounds
   (own and neighbours', rounds 1..t-1) in the swarm, against p_test's 1/2 for every earlier
   round except the last. A content shift within a length block is what clock conditioning
   cannot remove.

    python scripts/followup_core_mechanism.py --runs results/runs/core_cheap_talk/cheap_talk_n10_cycle_full \
        --tables results/audit/llama-3.1-8b --tag cheap_talk_d2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from swarm_audit import build  # noqa: E402
from swarm_audit import followup as fu  # noqa: E402
from swarm_audit.analysis import to_jsonable  # noqa: E402
from swarm_audit.audit import derive_actions_table  # noqa: E402
from swarm_audit.swarm import git_commit, graph_of, load_runs  # noqa: E402

OUT = REPO / "results" / "followup" / "core" / "mechanism"


def per_round_errors(data, tables_by_round, B, rng, sig: bool) -> dict:
    """tables_by_round(t) -> table for round t (1-based)."""
    R, T, n = data.p_H.shape
    e = np.zeros((R, T, n))
    for t in range(1, T + 1):
        tab = tables_by_round(t)
        kH = fu.kappa_H_of(tab)
        st = data.state_sig if (sig and tab.kind == "signal") else data.state
        e[:, t - 1] = fu.one_step_errors(data.p_H[:, t - 1], st[:, t - 1], kH)
    res = fu.boot_ratio(e.sum(2), np.ones_like(e).sum(2), B, rng)
    blocks = {f"{a}-{b}": fu.boot_mean(e[:, a - 1:b].mean((1, 2)), B, rng) for a, b in fu.ROUND_BLOCKS}
    return {"by_round": res, "by_block": blocks}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="core.yaml")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    cfg = build.load(args.config)
    B = cfg["analysis"]["bootstrap"]
    data = load_runs(Path(args.runs))
    g = graph_of(data)
    tables = build.load_tables(Path(args.tables), args.tag)
    base = tables.signal if tables.signal is not None else tables.actions
    blocks = [fu.clock_table(base, b)[0] for b in range(3)]

    def pick(t):  # the clock table for round t (START: any block, identical)
        return blocks[fu.length_block(max(t - 1, 1))]

    rng = fu.seeded(401)
    out = {"label": "exploratory diagnostic (teacher-forced, swarm histories)", "errors": {}}
    if tables.signal is not None:
        out["errors"]["signal_aware_pooled"] = per_round_errors(data, lambda t: base, B, rng, True)
        out["errors"]["signal_aware_clock"] = per_round_errors(data, pick, B, rng, True)
        der = [derive_actions_table(b, "primary") for b in blocks]
        out["errors"]["actions_only_pooled"] = per_round_errors(data, lambda t: tables.actions, B, rng, False)
        out["errors"]["actions_only_clock"] = per_round_errors(
            data, lambda t: der[fu.length_block(max(t - 1, 1))], B, rng, False)
    else:
        out["errors"]["actions_only_pooled"] = per_round_errors(data, lambda t: base, B, rng, False)
        out["errors"]["actions_only_clock"] = per_round_errors(data, pick, B, rng, False)

    # History content: fraction of high bids among own + neighbours' actions in rounds 1..t-1.
    A = data.actions.astype(float)                     # (R, T, n)
    adj = g.adjacency.astype(float)                    # A[i, j] = 1 if j -> i
    d = g.regular_degree
    neigh = np.einsum("rtj,ij->rti", A, adj)           # high neighbours per agent and round
    seen = A + neigh                                   # own + neighbours, out of d + 1
    T = A.shape[1]
    content, last = [], []
    for t in range(2, T + 1):
        content.append(float(seen[:, : t - 1].sum(1).mean() / ((t - 1) * (d + 1))))
        last.append(float(seen[:, t - 2].mean() / (d + 1)))
    earlier = [None] + [float(seen[:, : t - 2].sum(1).mean() / ((t - 2) * (d + 1))) for t in range(3, T + 1)]
    out["history_content"] = {
        "round": list(range(2, T + 1)),
        "swarm_high_fraction_all_earlier_rounds": content,
        "swarm_high_fraction_last_round": last,
        "swarm_high_fraction_before_last_round": earlier,
        "p_test_high_fraction_before_last_round": 0.5,
        "note": "p_test draws every round before the last uniformly (own and neighbours high with "
                "probability 1/2); the last round is set by the state, so the comparison that "
                "matters within a state is the rounds before the last"}
    out["git_commit"] = git_commit()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{Path(args.runs).parent.name}__{Path(args.runs).name}.json"
    path.write_text(json.dumps(to_jsonable(out)) + "\n")
    print(f"wrote {path}")
    for k, v in out["errors"].items():
        print(f"  {k:22s} " + "; ".join(f"{b} {x['mean']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]"
                                         for b, x in v["by_block"].items()))
    hc = out["history_content"]
    print("  swarm high fraction before the last round, rounds 3,5,8,12,16,20:",
          [round(hc["swarm_high_fraction_before_last_round"][t - 2], 3) for t in (3, 5, 8, 12, 16, 20)],
          "(p_test 0.5)")


if __name__ == "__main__":
    main()
