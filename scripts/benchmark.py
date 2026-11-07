#!/usr/bin/env python3
"""Gate G4, throughput benchmark (GPU; docs/handoff-build.md section 4).

First, Llama's signal responsiveness on the final cheap-talk prompt (screening grid,
d = 4 and d = 2; results/serving/signal_responsiveness.json). Then one full core run each
of counts at n = 10 and cheap talk at n = 20 (cycle, full history). Reports decisions per second, prompt tokens at round 20, wall-clock time and
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

import numpy as np  # noqa: E402

from swarm_audit import build, decisions, responsiveness  # noqa: E402
from swarm_audit.audit import synthetic_history  # noqa: E402
from swarm_audit.policies import Query  # noqa: E402
from swarm_audit.states import all_actions_states  # noqa: E402
from swarm_audit.swarm import Condition, read_run, run_condition, run_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="core.yaml")
    args = ap.parse_args()
    cfg = build.load(args.config)
    if cfg.get("results_dir"):   # it writes the core's G4 plan and results/serving files
        raise SystemExit("benchmark.py is the core's G4 and writes core files; not for an extra config")
    root = REPO / "results" / "runs"
    conds = [Condition("benchmark", 90, 10, "counts", 10, runs=1, T=cfg["T"], params=build.params(cfg)),
             Condition("benchmark", 90, 20, "cheap_talk", 20, runs=1, T=cfg["T"], params=build.params(cfg))]
    build.check_cost(cfg, cfg["serving"]["timeout_minutes"])
    res = {}
    with build.gpu_session(cfg, ("model_confirmed", "prompts_approved", "modal_account")):
        t_session = time.time()
        be, server = build.modal_backend(cfg, log_name="g4")
        be = build.FreshBackend(be)          # timings and responsiveness are measurements
        info = server.info.remote()
        # Abdul, 7 Nov 2026: re-measure Llama's signal responsiveness on the final
        # cheap-talk prompt (messages about the next bid). Done before the timed runs.
        resp = responsiveness.measure(be)
        rpath = REPO / "results" / "serving" / "signal_responsiveness.json"
        rpath.parent.mkdir(parents=True, exist_ok=True)
        rpath.write_text(json.dumps(resp, indent=1) + "\n")
        for d in ("d4", "d2"):
            print(f"signal responsiveness {d}: length 1 mean "
                  f"{resp[d]['signal_responsiveness_length1_mean']:.3f}, length 5 mean "
                  f"{resp[d]['signal_responsiveness_length5_mean']:.3f} (bar 0.05; screening d4: 0.051)")
        for c in conds:
            t0 = time.time()
            out = run_condition(c, be, root)
            sec = time.time() - t0
            usd = sec / 3600 * cfg["serving"]["l4_usd_per_hour"]
            lines = read_run(run_file(out, 0))
            tok20 = [ln["n_tokens"] for ln in lines if ln["round"] == c.T]
            res[c.name] = {"decisions": len(lines), "seconds": sec,
                           "decisions_per_s": len(lines) / sec, "usd_per_run": usd,
                           "prompt_tokens_round_T": {"min": min(tok20), "max": max(tok20)},
                           "fits_max_model_len": max(tok20) < cfg["serving"]["max_model_len"] - 8}
        # Batched throughput: real runs send 50 x n prompts per round, so one run's tiny
        # batches overstate the cost. Time 1,000-prompt batches of p_test histories
        # (lengths uniform on 1..19, as in the swarm on average) per channel.
        rng = np.random.default_rng(4)
        pt = build.ptest(cfg)
        sts = all_actions_states(2, start=False)
        sec_per_dec = {}
        for ch in ("counts", "cheap_talk"):
            qs = [Query(ch, synthetic_history(rng, ch, 2, sts[j % 6], pt), 2) for j in range(1000)]
            t0 = time.time()
            be.query(qs, fresh=True)
            sec_per_dec[ch] = (time.time() - t0) / len(qs)
        build.record_spend("G4 (whole GPU session)", time.time() - t_session, cfg)
    cfgs = {f: build.load(f) for f in ("core.yaml", "calibration.yaml", "secondary.yaml",
                                       "pilot.yaml")}
    plan = decisions.plan_g4(cfgs, sec_per_dec, build.spent_usd())
    if plan["status"] == "go" and plan["signal_audit"] != cfg["audit"]["signal"]:
        a_ = plan["signal_audit"]
        build.set_config_line("signal", f"{{n_fit: {a_['n_fit']}, n_score: {a_['n_score']}}}",
                              note="reduced by the settled G4 rule (Abdul: histories, never the prompt)")
    build.PLAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    build.PLAN_FILE.write_text(json.dumps(plan, indent=1) + "\n")
    summary = {"server": info, "benchmark_single_runs": res,
               "batched_seconds_per_decision": sec_per_dec,
               "signal_responsiveness": {d: {k: v for k, v in resp[d].items()
                                             if k not in ("groups", "prompts")}
                                         for d in ("d4", "d2")},
               "plan": {k: v for k, v in plan.items() if k not in ("run", "cut")},
               "spent_so_far_usd": build.spent_usd(), "cap_usd": cfg["serving"]["cost_cap_usd"]}
    out = REPO / "results" / "serving" / "g4_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "server"}, indent=1, default=str))
    if plan["status"] != "go":
        raise SystemExit("The G4 plan cannot fit arms 1-4b plus n = 10: stop and ask Abdul.")
    print(f"G4 plan: {plan['steps'] or 'everything fits'}; projected {plan['projected_hours']:.1f} "
          f"GPU-hours, ${plan['projected_usd']:.2f} (x{plan['margin']} margin applied in the check).")


if __name__ == "__main__":
    main()
