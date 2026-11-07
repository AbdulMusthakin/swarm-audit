#!/usr/bin/env python3
"""Gate G4, throughput benchmark (GPU; docs/handoff-build.md section 4).

One full core run each of counts at n = 10 and cheap talk at n = 20 (cycle, full
history). Reports decisions per second, prompt tokens at round 20, wall-clock time and
dollars per run, then extrapolates to the whole manifest (and to the audit tables).
Report to Abdul before launching anything larger.

    python scripts/benchmark.py --config core.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from swarm_audit import build  # noqa: E402
from swarm_audit.audit import table_states  # noqa: E402
from swarm_audit.swarm import Condition, read_run, run_condition, run_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="core.yaml")
    args = ap.parse_args()
    cfg = build.load(args.config)
    root = REPO / "results" / "runs"
    conds = [Condition("benchmark", 90, 10, "counts", 10, runs=1, T=cfg["T"], params=build.params(cfg)),
             Condition("benchmark", 90, 20, "cheap_talk", 20, runs=1, T=cfg["T"], params=build.params(cfg))]
    build.check_cost(cfg, cfg["serving"]["timeout_minutes"])
    res = {}
    with build.gpu_session(cfg, ("model_confirmed", "prompts_approved", "modal_account")):
        be, server = build.modal_backend(cfg, log_name="g4")
        info = server.info.remote()
        for c in conds:
            t0 = time.time()
            out = run_condition(c, be, root)
            sec = time.time() - t0
            usd = build.record_spend(f"G4 {c.name}", sec, cfg)
            lines = read_run(run_file(out, 0))
            tok20 = [ln["n_tokens"] for ln in lines if ln["round"] == c.T]
            res[c.name] = {"decisions": len(lines), "seconds": sec,
                           "decisions_per_s": len(lines) / sec, "usd_per_run": usd,
                           "prompt_tokens_round_T": {"min": min(tok20), "max": max(tok20)},
                           "fits_max_model_len": max(tok20) < cfg["serving"]["max_model_len"] - 8}
    # Extrapolation (decision-weighted; cheap talk costs three label read-outs per decision).
    per_dec = {"counts": res[conds[0].name]["seconds"] / res[conds[0].name]["decisions"],
               "cheap_talk": res[conds[1].name]["seconds"] / res[conds[1].name]["decisions"]}
    plan = []
    for cfile in ("calibration.yaml", "core.yaml", "secondary.yaml"):
        c2 = build.load(cfile)
        for arm in c2["arms"]:
            for c in build.conditions(c2, arm):
                dec = c.runs * c.n * c.T
                plan.append({"arm": arm, "condition": c.name, "decisions": dec,
                             "seconds": dec * per_dec[c.channel]})
    a, s = cfg["audit"]["actions"], cfg["audit"]["signal"]
    own = cfg["own_signal_in_state"]
    n_sig = len(table_states("cheap_talk", 2, "signal", own)) - 1
    audit_q = {"counts_d2": 6 * (a["n_fit"] + a["n_score"]) + 2,
               "cheap_talk_d2_signal": n_sig * (s["n_fit"] + s["n_score"]) + 2}
    plan.append({"arm": "audit", "condition": "counts_d2", "decisions": audit_q["counts_d2"],
                 "seconds": audit_q["counts_d2"] * per_dec["counts"]})
    plan.append({"arm": "audit", "condition": "cheap_talk_d2_signal",
                 "decisions": audit_q["cheap_talk_d2_signal"],
                 "seconds": audit_q["cheap_talk_d2_signal"] * per_dec["cheap_talk"]})
    price = cfg["serving"]["l4_usd_per_hour"]
    total_s = sum(p["seconds"] for p in plan)
    summary = {"server": info, "benchmark": res, "plan": plan,
               "total_hours": total_s / 3600, "total_usd": total_s / 3600 * price,
               "note": "single container, lockstep batches as in the benchmark; arm 7 included "
                       "but runs only if approved; startup and queueing time excluded",
               "spent_so_far_usd": build.spent_usd(), "cap_usd": cfg["serving"]["cost_cap_usd"]}
    out = REPO / "results" / "serving" / "g4_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "server"}, indent=1, default=str))
    print("Report to Abdul before launching anything larger.")


if __name__ == "__main__":
    main()
