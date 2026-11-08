#!/usr/bin/env python3
"""Assemble the follow-up evidence tables (results/followup/core/): numbers.json, claims.csv,
temporal_scores.csv, uncertainty.json and clock_audit.json.

Every claim is read from a committed JSON file by an explicit key path. Its row records the
value, the interval and its type, the source file and key, the commit that last changed the
source, the run set, and a status: original primary, secondary, exploratory, exact-model
calculation, or imported (extra arm).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FU = REPO / "results" / "followup" / "core"
P = "core_cheap_talk__cheap_talk_n10_cycle_full"

COND = "conditional on the audit tables and the forecast simulation; 2,000 whole-run resamples"


def get(path: str, keys: str):
    v = json.loads((REPO / path).read_text())
    for k in keys.split("/"):
        v = v[int(k)] if isinstance(v, list) else v[k]
    return v


def commit_of(path: str) -> str:
    out = subprocess.run(["git", "log", "-1", "--format=%h", "--", path], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
    return out or "uncommitted"


CLAIMS: list[dict] = []


def claim(cid, text, path, key, status, runs, interval_type="", lo=None, hi=None, scale=1.0):
    if not (REPO / path).exists():
        return
    try:
        v = get(path, key)
    except (KeyError, IndexError):
        return
    row = {"id": cid, "claim": text, "status": status, "run_set": runs, "source_file": path,
           "source_key": key, "source_commit": commit_of(path), "interval_type": interval_type}
    if isinstance(v, dict) and "mean" in v:
        row.update(value=v["mean"] * scale, lo=v.get("lo", None), hi=v.get("hi", None))
        if row["lo"] is not None:
            row["lo"] *= scale
            row["hi"] *= scale
    else:
        row.update(value=v, lo=lo, hi=hi)
    CLAIMS.append(row)


def build_claims() -> None:
    A = "results/analysis/core_cheap_talk/cheap_talk_n10_cycle_full.json"
    T = f"results/followup/core/temporal/{P}.json"
    C = f"results/followup/core/clock/{P}.json"
    run10 = "core_cheap_talk/cheap_talk_n10_cycle_full, runs 0-49 (Llama 3.1 8B, live)"
    k = "actions_only - signal_aware"
    claim("P1", "Primary contrast: paired endpoint CRPS D = CRPS(actions-only) - CRPS(signal-aware)",
          A, "scores/paired_D", "original primary", run10, "95% " + COND)
    claim("P2", "Endpoint CRPS, actions-only audit", A, "scores/crps/actions_only", "original primary", run10, "95% " + COND)
    claim("P3", "Endpoint CRPS, signal-aware audit", A, "scores/crps/signal_aware", "original primary", run10, "95% " + COND)
    claim("S1", "Max roundwise binned TV, actions-only (vs simultaneous null q95 in S2)", A,
          "scores/floor/actions_only/max_over_rounds", "secondary", run10, "simultaneous null 95% quantile")
    claim("S3", "Actions-only defect shift eps(p_swarm) - eps(p_test)", A,
          "defect_shift/actions_only/pooled/difference", "secondary", run10, "decomposition, no interval")
    claim("S4", "  of which within-state", A, "defect_shift/actions_only/pooled/within_state_term",
          "secondary", run10, "decomposition, no interval")
    claim("S5", "  of which state-frequency", A, "defect_shift/actions_only/pooled/state_frequency_term",
          "secondary", run10, "decomposition, no interval")
    claim("E1", "Simulation SD of endpoint D across 10 batches of 20,000 runs", T,
          f"monte_carlo/summary/{k}/endpoint/sd", "exploratory", run10, "SD over batches")
    claim("E2", "Endpoint D at the pre-set precision (740,000 simulated runs)", T,
          f"monte_carlo/precision_run/paired/{k}/endpoint", "exploratory", run10, "95% " + COND)
    claim("E3", "Early-round (2-7) CRPS contrast D_early (negative: actions-only better)", T,
          f"monte_carlo/precision_run/paired/{k}/2-7", "exploratory", run10, "95% " + COND)
    claim("E4", "D_early, nested runs x tables x simulation", T,
          f"table_resampling/summary/paired_nested/{k}/2-7", "exploratory", run10,
          "95% percentile over 200 nested resamples; excludes kernel drift and prompt/model choice")
    claim("E5", "Endpoint D, nested runs x tables x simulation", T,
          f"table_resampling/summary/paired_nested/{k}/endpoint", "exploratory", run10,
          "95% percentile over 200 nested resamples; excludes kernel drift and prompt/model choice")
    claim("E6", "Fraction of table resamples with D_early > 0", T,
          f"table_resampling/summary/paired/{k}/2-7/frac_positive", "exploratory", run10, "")
    claim("E7", "Mean signed one-step error, round 2, actions-only (teacher-forced)", T,
          "one_step/actions_only/by_round/signed", "exploratory", run10, "")
    for m, cid in (("signal_aware", "C1"), ("actions_only", "C2")):
        claim(cid, f"Clock-indexed audit: early CRPS improvement, pooled - clock ({m})", C,
              f"comparison/{m}/improvement_pooled_minus_clock/2-7", "exploratory", run10,
              "95% " + COND + "; 200,000 simulated runs per audit")
        claim(cid + "n", f"  same, nested runs x tables x simulation ({m})", C,
              f"table_sensitivity/results/{m}/2-7/nested_runs_tables_simulation", "exploratory", run10,
              "95% percentile over 200 nested resamples")
        claim(cid + "e", f"Clock-indexed audit: endpoint CRPS improvement, pooled - clock ({m})", C,
              f"comparison/{m}/improvement_pooled_minus_clock/endpoint", "exploratory", run10, "95% " + COND)
    for blk in ("8-13", "14-20"):
        claim(f"C1_{blk}", f"Clock-indexed audit: rounds {blk} improvement (signal_aware)", C,
              f"comparison/signal_aware/improvement_pooled_minus_clock/{blk}", "exploratory", run10, "95% " + COND)
    claim("X1", "Independent re-derivation agrees with the pipeline (signed errors exact; D within MC error)",
          "results/followup/core/independent_check.json", "pass", "exploratory", run10, "consistency check")
    # Imported from the extra arm (labels unchanged).
    O = "results/extra/analysis/olmo_core_cheap_talk/cheap_talk_n10_cycle_full.json"
    olmo = "olmo_core_cheap_talk/cheap_talk_n10_cycle_full, 50 runs (OLMo-3-7B, live; extra arm 9b, exploratory)"
    claim("I1", "OLMo: round-3 mean X_t, swarm", O, "X/swarm/mean/2", "imported (extra arm, exploratory)", olmo)
    claim("I2", "OLMo: round-3 mean X_t, actions-only forecast", O, "X/actions_only/mean/2", "imported (extra arm, exploratory)", olmo)
    claim("I3", "OLMo: P(no consensus by round 5), swarm", O, "consensus/swarm/survival/4", "imported (extra arm, exploratory)", olmo)
    claim("I4", "OLMo: P(no consensus by round 5), actions-only forecast", O, "consensus/actions_only/survival/4",
          "imported (extra arm, exploratory)", olmo, "simulation, 20,000 runs")
    N = "results/extra/arm8_numbers.json"
    n100 = "extra_rr4/counts_n100_random_regular_d4_seed20261108_full, 50 runs (Llama, live; extra arm 8)"
    claim("I5", "n = 100 counts: endpoint CRPS order-0 minus actions-only", N,
          "crps_paired_order0_minus_actions_only", "imported (extra arm 8)", n100, "95% " + COND)
    OC = f"results/followup/core/clock/olmo_core_cheap_talk__cheap_talk_n10_cycle_full.json"
    claim("O1", "OLMo (existing runs): clock-indexed early CRPS improvement, pooled - clock (signal_aware)", OC,
          "comparison/signal_aware/improvement_pooled_minus_clock/2-7", "exploratory (core analysis of imported runs)",
          olmo, "95% " + COND)
    claim("O2", "OLMo (existing runs): clock-indexed endpoint CRPS improvement (signal_aware)", OC,
          "comparison/signal_aware/improvement_pooled_minus_clock/endpoint", "exploratory (core analysis of imported runs)",
          olmo, "95% " + COND)


def temporal_csv() -> None:
    rows = []
    for f in sorted((FU / "temporal").glob("*.csv")):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    if rows:
        with open(FU / "temporal_scores.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def uncertainty_json() -> None:
    p = FU / "temporal" / f"{P}.json"
    r = json.loads(p.read_text())
    k = "actions_only - signal_aware"
    mc = r["monte_carlo"]
    out = {"condition": r["condition"], "label": r["label"],
           "layers": {
               "conditional_on_tables_and_simulation": r["original_simulation"]["paired"][k],
               "simulation_batches": mc["summary"][k],
               "materiality": mc.get("materiality"),
               "precision_run": {"n_model": mc.get("precision_run", {}).get("n_model"),
                                 **(mc.get("precision_run", {}).get("paired", {}).get(k, {}))},
               "table_resampling": r["table_resampling"]["summary"]["paired"][k],
               "nested_runs_tables_simulation": r["table_resampling"]["summary"]["paired_nested"][k]},
           "labels": r["table_resampling"]["labels"],
           "not_included": ["kernel drift beyond the measured panels (measured 0)",
                            "choice of prompt, model, payoffs, p_test", "serving numerics (floor 0)"],
           "source": str(p.relative_to(REPO))}
    (FU / "uncertainty.json").write_text(json.dumps(out, indent=1) + "\n")


def clock_json() -> None:
    out = {}
    for f in sorted((FU / "clock").glob("*.json")):
        r = json.loads(f.read_text())
        out[f.stem] = {"label": r["label"], "status": r["status"], "condition": r["condition"],
                       "design": r["design"],
                       "cells": {b: {kk: vv for kk, vv in v.items() if kk != "per_state"}
                                 for b, v in r["cells"].items()},
                       "eps_p_test_by_block": r["eps_p_test_by_block"],
                       "improvement_pooled_minus_clock": {m: v["improvement_pooled_minus_clock"]
                                                          for m, v in r["comparison"].items()},
                       "table_sensitivity": r["table_sensitivity"]["results"],
                       "round2_sensitivity": {"prompt_identity": r["round2_sensitivity"]["prompt_identity"],
                                              "label": r["round2_sensitivity"]["label"],
                                              "results": r["round2_sensitivity"]["results"]},
                       "source": str(f.relative_to(REPO))}
    if out:
        (FU / "clock_audit.json").write_text(json.dumps(out, indent=1) + "\n")


def main() -> None:
    build_claims()
    temporal_csv()
    uncertainty_json()
    clock_json()
    fields = ["id", "claim", "value", "lo", "hi", "interval_type", "status", "run_set",
              "source_file", "source_key", "source_commit"]
    for r in CLAIMS:
        if isinstance(r["value"], list):      # by-round arrays: keep round 2 (index 1)
            r["value"] = r["value"][1]
    with open(FU / "claims.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(CLAIMS)
    (FU / "numbers.json").write_text(json.dumps({r["id"]: r for r in CLAIMS}, indent=1, default=str) + "\n")
    print(f"{len(CLAIMS)} claims -> {FU / 'claims.csv'}")
    for r in CLAIMS:
        v = r["value"]
        s = f"{v:+.6f}" if isinstance(v, float) else str(v)
        if r.get("lo") is not None:
            s += f" [{r['lo']:+.6f}, {r['hi']:+.6f}]"
        print(f"  {r['id']:7s} {s:40s} {r['claim'][:80]}")


if __name__ == "__main__":
    sys.exit(main())
