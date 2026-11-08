#!/usr/bin/env python3
"""E3: OLMo size by memory, two by two (plan.json E3_size_by_memory).

Cells (cycle, degree 2, cheap talk, T = 20, the same prompts and payoffs):
  n4_last1  = arm 9a (existing)          n10_full  = arm 9b (existing)
  n10_last1 = followup_olmo_n10_last1    n4_full   = followup_olmo_n4_full (new tonight)

Reads logged runs and tables only. Also runs the core analysis (scripts/analyse.py's
analyse_condition) on the two new cells and the per-decision table check on n10_last1.

    python scripts/extra_followup_size_memory.py

Outputs (results/extra/followup/): size_memory.csv, size_memory.json,
analysis/<arm>/<condition>.json.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from scipy.stats import beta, norm  # noqa: E402

from swarm_audit import build, calibration, stats  # noqa: E402
from swarm_audit.analysis import analyse_condition, to_jsonable  # noqa: E402
from swarm_audit.provenance import check_tables_match_runs  # noqa: E402
from swarm_audit.swarm import graph_of, load_runs  # noqa: E402

CFG = build.load("extra-followup-olmo.yaml")
X = REPO / "results" / "extra"
OUT = X / "followup"
TABLES = X / "audit" / "olmo-3-7b"
B, SEED = 2000, 20261108
CELLS = {
    "n4_last1": {"runs": X / "runs/olmo_calibration_cheap_talk/cheap_talk_n4_cycle_last1", "n": 4,
                 "memory": "last1", "source": "arm 9a (existing, 7 Nov session)", "tag": "cheap_talk_d2_enumerated"},
    "n10_full": {"runs": X / "runs/olmo_core_cheap_talk/cheap_talk_n10_cycle_full", "n": 10,
                 "memory": "full", "source": "arm 9b (existing, 7-8 Nov session)", "tag": "cheap_talk_d2"},
    "n10_last1": {"runs": OUT / "runs/followup_olmo_n10_last1/cheap_talk_n10_cycle_last1", "n": 10,
                  "memory": "last1", "source": "new tonight (arm_id 92)", "tag": "cheap_talk_d2_enumerated"},
    "n4_full": {"runs": OUT / "runs/followup_olmo_n4_full/cheap_talk_n4_cycle_full", "n": 4,
                "memory": "full", "source": "new tonight (arm_id 93)", "tag": "cheap_talk_d2"},
}


def clopper_pearson(k: int, n: int, a: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def wilson(k: int, n: int, a: float = 0.05) -> tuple[float, float]:
    z = norm.ppf(1 - a / 2)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(c - h), float(c + h)


def newcombe(k1: int, n1: int, k2: int, n2: int, a: float = 0.05) -> dict:
    """Newcombe's hybrid score interval for p1 - p2 (method 10)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, a)
    l2, u2 = wilson(k2, n2, a)
    d = p1 - p2
    return {"diff": d, "lo": d - float(np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)),
            "hi": d + float(np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)),
            "method": f"Newcombe hybrid score (Wilson components), {100 * (1 - a):g}%"}


def boot_diff(a: np.ndarray, b: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    bs = np.array([a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
                   for _ in range(B)])
    d = float(a.mean() - b.mean())
    return {"diff": d, "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)),
            "method": f"independent whole-run bootstrap of each cell, B = {B}",
            "degenerate": bool(np.ptp(bs) == 0)}


P_START_H = float(np.sum(json.loads((TABLES / "cheap_talk_d2_enumerated_signal.json").read_text())
                         ["entries"]["START"]["kappa"][0]))


def cell_summary(name: str, spec: dict) -> dict:
    data = load_runs(spec["runs"])
    n = data.n
    y = stats.high_count(data.actions)
    F = y / (5 * n)
    k1, k0 = int((y == 5 * n).sum()), int((y == 0).sum())
    cons = stats.consensus(data.actions)
    tau = cons.tau
    rng = np.random.default_rng(SEED)
    mF = stats.bootstrap_mean(F, B, rng)
    return {
        "cell": name, "n": n, "memory": spec["memory"], "source": spec["source"],
        "runs": int(data.R), "run_dir": str(spec["runs"].relative_to(REPO)),
        "P_F1": k1 / data.R, "F1_count": k1, "P_F1_ci": clopper_pearson(k1, data.R),
        "P_F0": k0 / data.R, "F0_count": k0, "P_F0_ci": clopper_pearson(k0, data.R),
        "mean_F": mF["mean"], "mean_F_ci": (mF["lo"], mF["hi"]),
        "EX_t": stats.frac_high(data.actions).mean(0).tolist(),
        "consensus": {"reached": int(np.isfinite(tau).sum()), "censored_at_T_minus_2": int(np.isnan(tau).sum()),
                      "median_round_if_reached": float(np.nanmedian(tau)) if np.isfinite(tau).any() else None,
                      "side_counts": {s: int((cons.side == s).sum()) for s in ("H", "L")}},
        "round1": {"runs_all_H_round1": int(data.actions[:, 0, :].all(1).sum()),
                   "F1_among_round1_all_H": int(((y == 5 * n) & data.actions[:, 0, :].all(1)).sum()),
                   "F1_among_round1_not_all_H": int(((y == 5 * n) & ~data.actions[:, 0, :].all(1)).sum()),
                   "exact_P_round1_all_H": P_START_H ** n,
                   "note": "round 1 is the START prompt in every cell, so P(all high in round 1) = P(H | START)^n"},
        "evidence": "live runs", "_F": F, "_k1": k1, "_R": int(data.R), "_data": data,
    }


def main() -> None:
    fc = json.loads((OUT / "e3_forecast.json").read_text())
    cells = {k: cell_summary(k, v) for k, v in CELLS.items()}
    # forecasts (cached kernel or fitted tables; never live)
    fcast = {"n4_last1": (fc["n4_last1_reference_exact"]["P_F1"], "exact, enumerated table"),
             "n10_last1": (fc["n10_last1"]["both_signal_aware"]["P_F1"],
                           "Monte Carlo 100,000 runs, enumerated table (SE %.5f)"
                           % fc["n10_last1"]["both_signal_aware"]["P_F1_mc_se"]),
             "n4_full": (fc["n4_full"]["signal_aware_9b_table"]["P_F1"], "exact, 9b sampled table (approximation)")}
    a9b = json.loads((X / "analysis/olmo_core_cheap_talk/cheap_talk_n10_cycle_full.json").read_text())
    fcast["n10_full"] = (None, "no exact law at n = 10; 9b analysis: signal-aware simulation in "
                               "results/extra/analysis/olmo_core_cheap_talk")
    for k, (p, how) in fcast.items():
        cells[k]["forecast_P_F1"], cells[k]["forecast_source"] = p, how

    contrasts = {}
    pairs = {"memory_within_n4": ("n4_full", "n4_last1"), "memory_within_n10": ("n10_full", "n10_last1"),
             "size_within_last1": ("n10_last1", "n4_last1"), "size_within_full": ("n10_full", "n4_full")}
    for i, (name, (a, b)) in enumerate(pairs.items()):
        A, Bc = cells[a], cells[b]
        contrasts[name] = {"a": a, "b": b,
                           "P_F1_a_minus_b": newcombe(A["_k1"], A["_R"], Bc["_k1"], Bc["_R"]),
                           "mean_F_a_minus_b": boot_diff(A["_F"], Bc["_F"], SEED + i)}

    # Multiplicity sensitivity (independent review, 8 Nov 05:30; added after the frozen
    # plan, which specified nominal 95% intervals): Bonferroni-adjusted Newcombe intervals
    # over the two size contrasts and over all four contrasts.
    multiplicity = {"note": "post hoc sensitivity added after the independent review; the frozen "
                            "plan's nominal 95% intervals are the specified analysis",
                    "method": "Bonferroni-adjusted Newcombe hybrid score intervals"}
    for fam, names in (("two_size_contrasts", ["size_within_last1", "size_within_full"]),
                       ("all_four_contrasts", list(pairs))):
        a_adj = 0.05 / len(names)
        multiplicity[fam] = {"alpha_per_contrast": a_adj, "intervals": {
            nm: newcombe(cells[pairs[nm][0]]["_k1"], cells[pairs[nm][0]]["_R"],
                         cells[pairs[nm][1]]["_k1"], cells[pairs[nm][1]]["_R"], a_adj) for nm in names}}

    # checks and the core analysis of the new cells
    checks, analyses = {}, {}
    enum = build.load_tables(TABLES, "cheap_talk_d2_enumerated")
    full = build.load_tables(TABLES, "cheap_talk_d2")
    for k, tabs in (("n10_last1", enum), ("n4_full", full)):
        data = cells[k]["_data"]
        check_tables_match_runs(data.meta.get("kernel_fingerprint"),
                                {"actions": tabs.actions, "signal": tabs.signal, "actions_equal": tabs.actions_equal})
        checks[k] = {"kernel_fingerprint_matches_tables": True}
        if k == "n10_last1":
            checks[k]["per_decision_vs_enumerated_table"] = calibration.per_decision(data, enum.signal, 0.0)
        g = graph_of(data)
        an = CFG["analysis"]
        res = analyse_condition(data, g, tabs, an["n_model"], an["seed"], an["bootstrap"], build.ptest(CFG))
        res["inputs"] = {"runs": str(CELLS[k]["runs"].relative_to(REPO)), "tables": str(TABLES.relative_to(REPO)),
                         "tag": CELLS[k]["tag"], "probabilities": "logged (never re-queried)"}
        p = OUT / "analysis" / CELLS[k]["runs"].parent.name / f"{CELLS[k]['runs'].name}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(to_jsonable(res)) + "\n")
        analyses[k] = {"path": str(p.relative_to(REPO)), "crps": res["scores"]["crps"],
                       "paired_D": res["scores"].get("paired_D"),
                       "consensus_prob": {m: v["prob"] for m, v in res["consensus"].items()}}
    for k in ("n10_last1", "n4_full"):
        man = json.loads((CELLS[k]["runs"].parent / "manifest.json").read_text())
        dr = json.loads((CELLS[k]["runs"] / "drift.json").read_text())
        checks[k]["sessions"] = [{kk: s.get(kk) for kk in ("time", "runs_added", "seconds", "workspace", "git_commit")}
                                 for s in man["conditions"][CELLS[k]["runs"].name]["sessions"]]
        checks[k]["drift_last_session"] = {kk: dr.get(kk) for kk in ("start_vs_end", "start_vs_mixed", "floor", "action")}
        checks[k]["p_valid_below_0.9"] = man["conditions"][CELLS[k]["runs"].name]["summary"]["p_valid_below_0.9"]

    public = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in cells.items()}
    out = {"plan": "results/extra/followup/plan.json E3_size_by_memory", "cells": public,
           "contrasts": contrasts, "multiplicity_sensitivity": multiplicity, "checks": checks, "analyses_new_cells": analyses,
           "notes": {"P_F1_intervals": "Clopper-Pearson 95% (zero successes give a nonzero upper bound)",
                     "joint_randomisation": "the four cells were collected in different sessions and were not jointly randomised",
                     "n10_full_forecast": "see the 9b analysis", "no_phase_transition": "two sizes only"}}
    (OUT / "size_memory.json").write_text(json.dumps(to_jsonable(out), indent=1) + "\n")
    with open(OUT / "size_memory.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "n", "memory", "source", "runs", "F1_count", "P_F1", "P_F1_lo", "P_F1_hi",
                    "P_F0", "mean_F", "mean_F_lo", "mean_F_hi", "forecast_P_F1", "forecast_source", "evidence"])
        for k in ("n4_last1", "n4_full", "n10_last1", "n10_full"):
            c = public[k]
            w.writerow([k, c["n"], c["memory"], c["source"], c["runs"], c["F1_count"], f"{c['P_F1']:.4f}",
                        f"{c['P_F1_ci'][0]:.4f}", f"{c['P_F1_ci'][1]:.4f}", f"{c['P_F0']:.4f}",
                        f"{c['mean_F']:.4f}", f"{c['mean_F_ci'][0]:.4f}", f"{c['mean_F_ci'][1]:.4f}",
                        "" if c["forecast_P_F1"] is None else f"{c['forecast_P_F1']:.5f}", c["forecast_source"],
                        c["evidence"]])
    for k, c in public.items():
        print(k, c["F1_count"], "/", c["runs"], [round(x, 3) for x in c["P_F1_ci"]], "meanF", round(c["mean_F"], 4),
              "forecast", c["forecast_P_F1"])
    for k, c in contrasts.items():
        print(k, {kk: round(vv, 4) for kk, vv in c["P_F1_a_minus_b"].items() if isinstance(vv, float)})
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "sessions"} for k, v in checks.items()}, default=str)[:1500])


if __name__ == "__main__":
    main()
