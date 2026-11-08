#!/usr/bin/env python3
"""Disclosed confirmation of an excepted G5 (Abdul, 7 Nov 2026).

After the cheap-talk G5 per-run miss was accepted as chance, arm 4b is run once more with
fresh seeds (run indices 50-99, same table, same folder, so provenance must match). This
script reports the per-run test on the original 50, on the new 50, and on all 100 pooled,
checks the new runs per decision, and writes results/gates/G5_<channel>_confirmation.json.
It passes only if the new 50 and the pooled 100 both pass (Abdul's rule 3); otherwise the
cheap-talk core is paused (counts is not affected). Exits 1 on a failure.

    python scripts/run_swarm.py --config calibration.yaml --arm calibration_cheap_talk --runs 100
    python scripts/confirm_g5.py --runs results/runs/calibration_cheap_talk/cheap_talk_n4_cycle_last1 \
        --tables results/audit/llama-3.1-8b --tag cheap_talk_d2_enumerated --original 50
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from swarm_audit import build, calibration  # noqa: E402
from swarm_audit.analysis import to_jsonable  # noqa: E402
from swarm_audit.prediction import ActionsChain, SignalChain  # noqa: E402
from swarm_audit.provenance import check_tables_match_runs  # noqa: E402
from swarm_audit.swarm import git_commit, load_runs  # noqa: E402
from swarm_audit.topology import make  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="calibration.yaml")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--original", type=int, default=50, help="runs 0..original-1 are the original set")
    args = ap.parse_args()
    cfg = build.load(args.config)
    floor = build.numerical_floor(cfg)
    sims = cfg["analysis"]["ks_sims"]
    data = load_runs(Path(args.runs))
    t = build.load_tables(Path(args.tables), args.tag)
    tabs = {"actions": t.actions, "signal": t.signal}
    check_tables_match_runs(data.meta["kernel_fingerprint"], tabs)
    g = make(data.graph, data.n)
    d = g.regular_degree
    cheap = data.channel == "cheap_talk"
    chain = (SignalChain(t.signal.lookup(), d, t.signal.own_signal) if cheap
             else ActionsChain(t.actions.lookup(), d))
    gate_table = t.signal if cheap else t.actions
    orig = np.nonzero(data.runs < args.original)[0]
    new = np.nonzero(data.runs >= args.original)[0]
    if len(new) == 0:
        raise SystemExit("no new runs: run the confirmation first")
    sets = {"original": data.subset(orig), "new": data.subset(new), "pooled": data}
    res = {k: calibration.per_run(v, chain, g, sims) for k, v in sets.items()}
    pd_new = calibration.per_decision(sets["new"], gate_table, floor)
    finding = ({k: calibration.per_run(v, ActionsChain(t.actions.lookup(), d), g, sims)
                for k, v in sets.items()} if cheap else None)
    ok = bool(pd_new["pass"] and res["new"]["pass"] and res["pooled"]["pass"])
    rec = {"gate": "G5", "channel": data.channel, "confirmation": True, "pass": ok,
           "rule": "pass if the new runs pass per decision and the per-run test passes on the "
                   "new runs and on all runs pooled (Abdul, 7 Nov 2026, rule 3)",
           "runs": {"original": len(orig), "new": len(new), "pooled": data.R,
                    "new_run_indices": [int(data.runs[new].min()), int(data.runs[new].max())]},
           "per_decision_new": pd_new, "per_run": res,
           "finding_actions_only_chain": finding,
           "kernel_fingerprint": data.meta["kernel_fingerprint"],
           "run_fingerprint": data.meta["fingerprint"], "floor": floor,
           "git_commit": git_commit(), "time": dt.datetime.now(dt.timezone.utc).isoformat()}
    path = calibration.GATES_DIR / f"G5_{data.channel}_confirmation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(rec), indent=1) + "\n")
    with open(calibration.GATES_DIR / "history.jsonl", "a") as fh:
        fh.write(json.dumps(to_jsonable(rec)) + "\n")
    for k, r in res.items():
        print(f"{k:9s} ({sets[k].R:3d} runs): gap {r['gap']:.3f}, p = {r['p_value']:.4f}, "
              f"mean y {r['swarm_mean_y']:.2f} vs exact {r['exact_mean_y']:.2f} -> "
              f"{'pass' if r['pass'] else 'FAIL'}")
    print(f"per decision, new runs: {'pass' if pd_new['pass'] else 'FAIL'} "
          f"(max difference {pd_new['max_abs_diff_p_H']}, {pd_new['decisions']} decisions)")
    print(f"confirmation: {'PASS' if ok else 'FAIL'} -> {path}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
