#!/usr/bin/env python3
"""Investigate a G5 per-run miss (docs/handoff-build.md section 4: investigate, don't fix).

CPU only; reads the logged calibration runs and the enumerated table. Checks, in order:
1. sampling: each logged action and signal follows from its logged draw and probabilities;
2. wiring: each agent's logged state equals the state rebuilt from its own and its
   in-neighbours' logged outputs in the previous round;
3. random streams: the logged draws are uniform (Kolmogorov-Smirnov);
4. end to end: the swarm code, with the enumerated table as its kernel, reproduces the
   exact law of y over many runs (high power against a wiring error);
5. where the observed runs deviate (law of y, mean X_t by round).

    python scripts/diagnose_g5.py --runs results/runs/calibration_cheap_talk/cheap_talk_n4_cycle_last1 \
        --table results/audit/llama-3.1-8b/cheap_talk_d2_enumerated_signal.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from scipy.stats import kstest  # noqa: E402

from swarm_audit import calibration, stats  # noqa: E402
from swarm_audit.audit import AuditTable  # noqa: E402
from swarm_audit.ladder import Record  # noqa: E402
from swarm_audit.policies import KernelOut  # noqa: E402
from swarm_audit.prediction import ActionsChain, SignalChain, exact_law  # noqa: E402
from swarm_audit.states import actions_state, key, signal_state  # noqa: E402
from swarm_audit.swarm import (Condition, build_records, load_runs, read_run,  # noqa: E402
                               run_condition, sample_signal)
from swarm_audit.topology import make  # noqa: E402


class TableBackend:
    """The kernel read straight from an enumerated table (last round only)."""

    def __init__(self, table: AuditTable, own_signal: bool = True):
        self.t, self.own = table, own_signal
        self.name, self.params = "table", None

    def query(self, queries, fresh=False):
        out = []
        for q in queries:
            h = q.shown
            if self.t.kind == "signal":
                raw = self.t.entries[key(signal_state(h, self.own))]["raw"]
                out.append(KernelOut(p_H=raw["p_H"], sig={a: tuple(raw["sig"][a]) for a in "HL"},
                                     sig_p_valid={"H": 1.0, "L": 1.0}))
            else:
                raw = self.t.entries[key(actions_state(h))]["raw"]
                out.append(KernelOut(p_H=raw["p_H"]))
        return out

    def describe(self):
        return {"backend": "table", "source": self.t.meta.get("provenance", {}).get("fingerprint")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--sim-runs", type=int, default=20000)
    ap.add_argument("--out", default=str(REPO / "results" / "gates" / "G5_cheap_talk_investigation.json"))
    args = ap.parse_args()
    folder = Path(args.runs)
    table = AuditTable.load(Path(args.table))
    prov = json.loads((folder / "provenance.json").read_text())
    design = prov["payload"]["design"]
    own = design["own_signal_in_state"]
    g = make(design["graph"], design["n"])
    cheap = design["channel"] == "cheap_talk"
    lines = [ln for f in sorted(folder.glob("run_*.jsonl.gz")) for ln in read_run(f)]
    rep: dict = {"runs_folder": str(folder), "table": args.table, "decisions": len(lines)}

    # 1. sampling
    bad_a = sum((ln["u_action"] < ln["p_H"]) != (ln["action"] == "H") for ln in lines)
    bad_m = 0
    if cheap:
        bad_m = sum(sample_signal(ln["u_signal"], ln[f"sig_after_{ln['action']}"]) != ln["signal"]
                    for ln in lines)
    rep["1_sampling"] = {"action_mismatches": bad_a, "signal_mismatches": bad_m,
                         "pass": bad_a == 0 and bad_m == 0}

    # 2. wiring
    by = {(ln["run"], ln["round"], ln["agent"]): ln for ln in lines}
    runs = sorted({ln["run"] for ln in lines})
    T = max(ln["round"] for ln in lines)
    wrong = 0
    for r in runs:
        for t in range(2, T + 1):
            prev = [by[(r, t - 1, i)] for i in range(g.n)]
            recs = build_records(g, [p["action"] for p in prev],
                                 [p["signal"] for p in prev] if cheap else None)
            for i in range(g.n):
                want = key(signal_state((recs[i],), own)) if cheap else key(actions_state((recs[i],)))
                got = by[(r, t, i)]["state_sig" if cheap else "state"]
                wrong += want != got
        wrong += sum(by[(r, 1, i)]["state" if not cheap else "state_sig"] != "START" for i in range(g.n))
    rep["2_wiring"] = {"state_mismatches": wrong, "pass": wrong == 0}

    # 3. random streams
    ua = np.array([ln["u_action"] for ln in lines])
    rep["3_draws"] = {"u_action_ks_p": float(kstest(ua, "uniform").pvalue)}
    if cheap:
        rep["3_draws"]["u_signal_ks_p"] = float(kstest([ln["u_signal"] for ln in lines], "uniform").pvalue)

    # 4. end to end on CPU: swarm code + table kernel vs the exact law
    chain = SignalChain(table.lookup(), g.regular_degree, own) if cheap else \
        ActionsChain(table.lookup(), g.regular_degree)
    ex = exact_law(chain, g, T)
    with tempfile.TemporaryDirectory() as tmp:
        cond = Condition("diag", 999, 1, design["channel"], g.n, design["graph"], T,
                         design["last_k"], args.sim_runs, 12345, own_signal_in_state=own)
        sim = load_runs(run_condition(cond, TableBackend(table, own), Path(tmp), log=lambda *a: None))
    y_sim = stats.high_count(sim.actions)
    G = stats.cdf_from_law(ex["y"])
    gap_sim = stats.ks_gap(y_sim, G)
    null = stats.ks_null_gaps(ex["y"], args.sim_runs, 5000, np.random.default_rng(7))
    rep["4_end_to_end"] = {"sim_runs": args.sim_runs, "gap": gap_sim,
                           "p_value": stats.mc_pvalue(gap_sim, null),
                           "sim_mean_y": float(y_sim.mean()),
                           "exact_mean_y": float((np.arange(len(G)) * ex["y"]).sum()),
                           "max_abs_diff_in_law_of_y": float(np.abs(
                               np.bincount(y_sim, minlength=len(G)) / len(y_sim) - ex["y"]).max())}

    # 5. where the observed runs deviate
    data = load_runs(folder)
    y = stats.high_count(data.actions)
    emp = stats.cdf_from_samples(y, len(G) - 1)
    k = int(np.argmax(np.abs(emp - G)))
    rep["5_observed"] = {
        "runs": data.R, "mean_y": float(y.mean()), "sd_y": float(y.std(ddof=1)),
        "exact_sd_y": float(np.sqrt(((np.arange(len(G)) - rep["4_end_to_end"]["exact_mean_y"]) ** 2
                                     * ex["y"]).sum())),
        "largest_cdf_gap_at_k": k, "empirical_cdf_at_k": float(emp[k]), "exact_cdf_at_k": float(G[k]),
        "mean_X_t_swarm": stats.frac_high(data.actions).mean(0).round(3).tolist(),
        "mean_X_t_exact": (ex["X"] @ np.arange(g.n + 1) / g.n).round(3).tolist(),
        "per_run_test": calibration.per_run(data, chain, g, 20000)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "5_observed"}, indent=1, default=float))
    o = rep["5_observed"]
    print(json.dumps({k: v for k, v in o.items() if k != "per_run_test"}, indent=1, default=float))


if __name__ == "__main__":
    main()
