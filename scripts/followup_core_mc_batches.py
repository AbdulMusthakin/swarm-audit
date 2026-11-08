#!/usr/bin/env python3
"""Post-review sensitivity (disclosed; not in the frozen plan): the simulation SD of the
primary D from 30 independent batches of 20,000 runs, to estimate the table-only SD more
precisely than the plan's 10 batches allowed. Prompted by the independent Codex review,
whose 20 batches gave a smaller SD (5.4e-5) than our 10 (7.8e-5).

    python scripts/followup_core_mc_batches.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

from swarm_audit import build  # noqa: E402
from swarm_audit import followup_core as fu  # noqa: E402
from swarm_audit.swarm import graph_of, load_runs  # noqa: E402
from followup_core_temporal import SCORE_KEYS, scores_from_pools  # noqa: E402

FU = REPO / "results" / "followup" / "core"
BATCHES = 30


def main() -> None:
    cfg = build.load("core.yaml")
    an, pt = cfg["analysis"], build.ptest(cfg)
    data = load_runs(REPO / "results/runs/core_cheap_talk/cheap_talk_n10_cycle_full")
    g = graph_of(data)
    t = build.load_tables(REPO / "results/audit/llama-3.1-8b", "cheap_talk_d2")
    cs = fu.ChainSet.from_tables(t.signal, None, g.regular_degree, pt)
    D = []
    for b in range(BATCHES):
        sc = scores_from_pools(cs.simulate(g, data.T, an["n_model"], fu.seeded(501, b)), data)
        D.append({k: float((sc["actions_only"][k] - sc["signal_aware"][k]).mean()) for k in SCORE_KEYS})
    tr = json.loads((FU / "temporal/core_cheap_talk__cheap_talk_n10_cycle_full.json").read_text())
    key = "actions_only - signal_aware"
    out = {"label": "post-review sensitivity (disclosed addition, not in the frozen plan)",
           "batches": BATCHES, "n_model": an["n_model"], "seed": "SeedSequence([20261108, 501, b])",
           "per_batch_D": D, "summary": {}}
    for k in SCORE_KEYS:
        v = np.array([x[k] for x in D])
        sd = float(v.std(ddof=1))
        res_sd = tr["table_resampling"]["summary"]["paired"][key][k]["sd"]
        out["summary"][k] = {
            "mc_sd_30_batches": sd,
            "mc_sd_10_batches_plan": tr["monte_carlo"]["summary"][key][k]["sd"],
            "table_resample_sd_incl_mc": res_sd,
            "table_only_sd_approx_30": float(np.sqrt(max(res_sd ** 2 - sd ** 2, 0.0))),
            "note": "variance subtraction is approximate: MC variance can change with the table, "
                    "and 200 resamples / 30 batches give imprecise SDs (relative SE about 5% / 13%)"}
    path = FU / "uncertainty_mc30.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
