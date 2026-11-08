#!/usr/bin/env python3
"""E2 live validation (plan.json E2_observation_interventions.live_validation).

For each of the four live cells (n = 4 cycle, last-1, 50 runs, arm_ids 94-97):
- implementation: every logged observed record equals the intervention applied to the
  actual record, and the next decision's logged state is that observed record;
- per decision: logged probabilities equal the enumerated table entry of the observed
  state at the floor (0), with one prompt per state equal to the table's;
- per run: the G5 discrete Monte Carlo p-value of the gap between the empirical CDF of y
  and the cell's exact law (observation_interventions.json), familywise alpha 0.01
  Bonferroni over the four cells (0.0025 each). Failures are reported, never loosened.

    python scripts/extra_followup_live.py

Output: results/extra/followup/observation_live.json. These runs check the counterfactual
implementation against the served LLM; every prompt is an enumerated one, so they add no
new kernel information.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from scipy.stats import beta  # noqa: E402

from swarm_audit import build, calibration, stats  # noqa: E402
from swarm_audit.followup import record_observer  # noqa: E402
from swarm_audit.ladder import Record  # noqa: E402
from swarm_audit.swarm import _rec_dict, load_runs, read_run  # noqa: E402

F = REPO / "results" / "extra" / "followup"
CELLS = {"unmodified": "followup_obs_unmodified", "own_null": "followup_obs_own_null",
         "neighbour_null": "followup_obs_neighbour_null", "both_null": "followup_obs_both_null"}
ALPHA_FW, SIMS, SEED = 0.01, 20000, 20261108


def rec(d: dict) -> Record:
    return Record(d["action"], d["n_high"], d["n_low"], d.get("signal"), d.get("sig_high"),
                  d.get("sig_low"), d.get("sig_none"))


def key_of(r: Record) -> str:
    return f"{r.action}-{r.n_high}-{r.sig_high}-{r.sig_low}-{r.sig_none}-{r.signal}"


def main() -> None:
    obs = json.loads((F / "observation_interventions.json").read_text())["models"]["olmo-3-7b"]["processes"]
    enum = build.load_tables(REPO / "results/extra/audit/olmo-3-7b", "cheap_talk_d2_enumerated")
    alpha = ALPHA_FW / len(CELLS)
    out = {"alpha_per_cell": alpha, "familywise_alpha": ALPHA_FW, "sims": SIMS, "cells": {}}
    for proc, arm in CELLS.items():
        folder = F / "runs" / arm / "cheap_talk_n4_cycle_last1"
        files = sorted(folder.glob("run_*.jsonl.gz"))
        if len(files) < 50:
            out["cells"][proc] = {"status": f"not run or incomplete ({len(files)} runs)"}
            continue
        fn = record_observer(proc)
        bad_obs, bad_state, n_rows = 0, 0, 0
        for f in files:
            rows = read_run(f)
            by = {(r["round"], r["agent"]): r for r in rows}
            for r in rows:
                n_rows += 1
                if r.get("observed_record") != _rec_dict(fn(rec(r["record"]))):
                    bad_obs += 1
                nxt = by.get((r["round"] + 1, r["agent"]))
                if nxt is not None and nxt["state_sig"] != key_of(rec(r["observed_record"])):
                    bad_state += 1
        data = load_runs(folder)
        pdec = calibration.per_decision(data, enum.signal, 0.0)
        y = stats.high_count(data.actions)
        law = np.asarray(obs[proc]["y_law"])
        G = stats.cdf_from_law(law)
        gap = stats.ks_gap(y, G)
        null = stats.ks_null_gaps(law, data.R, SIMS, np.random.default_rng(SEED))
        p = stats.mc_pvalue(gap, null)
        k = int((y == 20).sum())
        lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, data.R - k + 1))
        hi = 1.0 if k == data.R else float(beta.ppf(0.975, k + 1, data.R - k))
        man = json.loads((folder.parent / "manifest.json").read_text())["conditions"][folder.name]
        dr = json.loads((folder / "drift.json").read_text())
        out["cells"][proc] = {
            "arm": arm, "runs": int(data.R), "decisions": n_rows,
            "implementation": {"observed_record_mismatches": bad_obs, "next_state_mismatches": bad_state},
            "per_decision": pdec,
            "per_run": {"gap": gap, "p_value": p, "pass": bool(p > alpha), "alpha": alpha},
            "live_P_F1": k / data.R, "live_F1_count": k, "live_P_F1_ci": [lo, hi],
            "exact_P_F1": obs[proc]["P_F1"],
            "live_mean_F": float((y / 20).mean()), "exact_E_F": obs[proc]["E_F"],
            "round1_all_H_runs": int(data.actions[:, 0, :].all(1).sum()),
            "drift": {kk: dr.get(kk) for kk in ("start_vs_end", "start_vs_mixed", "floor", "action")},
            "p_valid_below_0.9": man["summary"]["p_valid_below_0.9"],
            "evidence": "live runs (implementation check against the served LLM)"}
        c = out["cells"][proc]
        print(proc, f"F1 {k}/{data.R}", "exact", round(c["exact_P_F1"], 4), "p", round(p, 4),
              "per-dec", pdec["pass"], pdec["max_abs_diff_p_H"], "impl", bad_obs, bad_state)
    done = [c for c in out["cells"].values() if "per_run" in c]
    out["all_pass"] = bool(done) and all(c["per_run"]["pass"] and c["per_decision"]["pass"]
                                         and not c["implementation"]["observed_record_mismatches"]
                                         and not c["implementation"]["next_state_mismatches"] for c in done)
    out["cells_complete"] = len(done)
    (F / "observation_live.json").write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
