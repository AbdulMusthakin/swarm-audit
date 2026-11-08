#!/usr/bin/env python3
"""Reapply the original four-agent OLMo roundwise check to the fresh, unmodified cohort.

Exploratory and retrospective. The original analysis (extra arm 9a,
results/extra/analysis/olmo_calibration_cheap_talk/cheap_talk_n4_cycle_last1.json)
compared 50 live runs with the exact round laws of the cached audit chains and a
simultaneous cutoff for the largest binned TV over rounds. This script applies the same
check, unchanged, to the 50 separately seeded runs of the E2 "unmodified" cell
(results/extra/followup/runs/followup_obs_unmodified), which used the same kernel
(fingerprint checked below) and showed agents their messages as logged.

Inputs, all committed: the enumerated OLMo tables, both cohorts' per-decision logs and
the original analysis (for its cutoffs). CPU only; no model queries; deterministic.

    python scripts/report_fresh_cohort.py

Output: results/extra/followup/fresh_cohort_roundwise.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from scipy.stats import beta  # noqa: E402

from swarm_audit import build, stats  # noqa: E402
from swarm_audit.prediction import ActionsChain, SignalChain, exact_law  # noqa: E402
from swarm_audit.swarm import load_runs  # noqa: E402
from swarm_audit.topology import cycle  # noqa: E402

ORIGINAL = REPO / "results/extra/analysis/olmo_calibration_cheap_talk/cheap_talk_n4_cycle_last1.json"
COHORTS = {
    "original": REPO / "results/extra/runs/olmo_calibration_cheap_talk/cheap_talk_n4_cycle_last1",
    "fresh": REPO / "results/extra/followup/runs/followup_obs_unmodified/cheap_talk_n4_cycle_last1",
}
TABLES = REPO / "results/extra/audit/olmo-3-7b"
OUT = REPO / "results/extra/followup/fresh_cohort_roundwise.json"
N, T = 4, 20


def clopper_pearson(k: int, n: int) -> list[float]:
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def main() -> None:
    orig = json.loads(ORIGINAL.read_text())
    kernel = orig["condition"]["kernel_fingerprint"]
    tables = build.load_tables(TABLES, "cheap_talk_d2_enumerated")
    g = cycle(N)
    chains = {"actions_only": ActionsChain(tables.actions.lookup(), g.regular_degree),
              "signal_aware": SignalChain(tables.signal.lookup(), g.regular_degree,
                                          tables.signal.own_signal)}
    laws = {k: exact_law(ch, g, T) for k, ch in chains.items()}
    out: dict = {
        "status": "exploratory, retrospective: the original check reapplied without change",
        "reference": "exact round laws of the cached audit chains (n = 4 cycle, last-round)",
        "cutoff_source": str(ORIGINAL.relative_to(REPO)),
        "event": "F = 1: every agent bids high in every round 16-20",
        "exact_P_F1": {k: float(v["y"][-1]) for k, v in laws.items()},
        "cohorts": {},
    }
    for name, folder in COHORTS.items():
        manifest = json.loads((folder.parent / "manifest.json").read_text())
        fp = manifest["conditions"][folder.name]["provenance"]["kernel_fingerprint"]
        if fp != kernel:
            raise SystemExit(f"{name}: kernel fingerprint {fp} differs from the original {kernel}")
        data = load_runs(folder)
        X = stats.frac_high(data.actions)
        y = stats.high_count(data.actions)
        k = int((y == stats.WINDOW * N).sum())
        res = {"runs": int(data.R), "kernel_fingerprint": fp,
               "live_F1": k, "live_F1_ci_clopper_pearson": clopper_pearson(k, int(data.R)),
               "round1_all_high_runs": int(data.actions[:, 0, :].all(1).sum()),
               "F1_runs_started_all_high": int(((y == stats.WINDOW * N)
                                                & data.actions[:, 0, :].all(1)).sum()),
               "roundwise": {}}
        for chain, law in laws.items():
            delta = stats.delta_t(X, stats.binned_law_exact(law["X"]))
            i = int(np.argmax(delta))
            cut = float(orig["scores"]["floor"][chain]["max_over_rounds"])
            res["roundwise"][chain] = {"max_delta": float(delta[i]), "round_of_max": i + 1,
                                       "original_cutoff": cut, "flagged": bool(delta[i] > cut),
                                       "delta_t": [float(v) for v in delta]}
        out["cohorts"][name] = res
        ao = res["roundwise"]["actions_only"]
        print(f"{name}: F=1 in {k}/{data.R}; actions-only max TV {ao['max_delta']:.3f} "
              f"(round {ao['round_of_max']}) vs cutoff {ao['original_cutoff']:.3f}")
    # The original cohort must reproduce the committed analysis exactly.
    for chain in laws:
        ref = orig["scores"]["delta"][chain]
        got = out["cohorts"]["original"]["roundwise"][chain]["delta_t"]
        if not np.allclose(ref, got, atol=1e-12):
            raise SystemExit(f"original cohort does not reproduce the committed Delta_t ({chain})")
    out["original_reproduces_committed_analysis"] = True
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
