"""Gate G5: the calibration arms (n = 4, cycle, last round only) against their tables.

Per decision: every logged probability must equal the enumerated table entry for its
state, up to the numerical floor from G3, and prompts with the same state must be
byte-identical (one prompt sha256 per state, equal to the table's).

Per run (the teammates' test): y = high bids in rounds 16-20 (out of 20 decisions).
The gap between the empirical CDF of y over the runs and the exact CDF from the chain
is compared with the 99th percentile of the same gap over 20,000 simulated sets of
runs from that chain. Arm 4b is gated on the signal-aware chain; the actions-only
chain's result there is a finding, not a gate.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import stats
from .audit import AuditTable
from .prediction import ActionsChain, SignalChain, exact_law
from .swarm import RunData
from .topology import Graph


def per_decision(data: RunData, table: AuditTable, floor: float = 0.0) -> dict:
    """Max difference between logged and table probabilities, and prompt identity."""
    sig = table.kind == "signal"
    st = data.state_sig if sig else data.state
    max_pH, max_sig, n_dec = 0.0, 0.0, 0
    worst = None
    for idx in np.ndindex(st.shape):
        k = st[idx]
        n_dec += 1
        raw = table.entries[k].get("raw")   # enumerated tables keep the raw kernel output
        dH = abs(data.p_H[idx] - (raw["p_H"] if raw else table.kappa_H(k)))
        if dH > max_pH:
            max_pH, worst = dH, {"state": k, "index": list(map(int, idx))}
        if sig:
            if raw:
                conds = [np.asarray(raw["sig"]["H"]), np.asarray(raw["sig"]["L"])]
            else:
                J = table.kappa_joint(k)
                conds = [J[ai] / J[ai].sum() if J[ai].sum() > 0 else None for ai in range(2)]
            for ai, cond in enumerate(conds):
                if cond is not None:
                    max_sig = max(max_sig, float(np.abs(data.sig[idx][ai] - cond).max()))
    shas: dict[str, set] = defaultdict(set)
    for idx in np.ndindex(st.shape):
        shas[st[idx]].add(data.sha[idx])
    multi = {k: len(v) for k, v in shas.items() if len(v) > 1}
    table_mismatch = [k for k, v in shas.items()
                      if table.entries[k].get("prompt_sha256") is not None
                      and v != {table.entries[k]["prompt_sha256"]}]
    ok = max_pH <= floor and max_sig <= floor and not multi and not table_mismatch
    return {"pass": bool(ok), "decisions": n_dec, "floor": floor,
            "max_abs_diff_p_H": max_pH, "max_abs_diff_signal": max_sig, "worst": worst,
            "states_seen": len(shas), "states_with_several_prompts": multi,
            "states_whose_prompt_differs_from_table": table_mismatch}


def per_run(data: RunData, chain, g: Graph, sims: int = 20000, q: float = 0.99,
            seed: int = 5) -> dict:
    ex = exact_law(chain, g, data.T)
    y = stats.high_count(data.actions)
    G = stats.cdf_from_law(ex["y"])
    gap = stats.ks_gap(y, G)
    thr = stats.ks_threshold(ex["y"], data.R, sims, q, np.random.default_rng(seed))
    K = len(G) - 1
    return {"pass": bool(gap < thr), "gap": gap, "threshold_q99": thr, "runs": data.R,
            "chain": getattr(chain, "name", type(chain).__name__),
            "exact_mean_y": float((np.arange(K + 1) * ex["y"]).sum()),
            "swarm_mean_y": float(y.mean()), "n_profiles": ex["n_profiles"]}


def check_arm(data: RunData, tables: dict[str, AuditTable], g: Graph, floor: float = 0.0,
              sims: int = 20000) -> dict:
    d = g.regular_degree
    if data.channel == "counts":
        t = tables["actions"]
        return {"per_decision": per_decision(data, t, floor),
                "per_run": per_run(data, ActionsChain(t.lookup(), d), g, sims)}
    t = tables["signal"]
    res = {"per_decision": per_decision(data, t, floor),
           "per_run": per_run(data, SignalChain(t.lookup(), d, t.own_signal), g, sims)}
    res["finding_actions_only_chain"] = per_run(
        data, ActionsChain(tables["actions"].lookup(), d), g, sims)
    return res
