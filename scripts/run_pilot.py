#!/usr/bin/env python3
"""Gate G6, the pilot (docs/handoff-build.md section 4; configs/pilot.yaml).

With inputs.pilot_rule = "none" in core.yaml, the gate is skipped. Otherwise: 5 counts
runs at n = 10 on the cycle with the current payoffs, logged to results/pilot/ (never
analysed), and a summary (X_t, consensus) for applying Abdul's rule once. The rule
itself, and the frozen settings for both channels, are recorded by hand in
configs/core.yaml and configs/calibration.yaml after Abdul decides.

    python scripts/run_pilot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from swarm_audit import build, stats  # noqa: E402
from swarm_audit.swarm import Condition, load_runs, run_condition  # noqa: E402


def main() -> None:
    cfg = build.load("core.yaml")
    pil = build.load("pilot.yaml")
    rule = cfg["inputs"].get("pilot_rule")
    if rule == "none":
        print("pilot_rule is 'none': gate G6 skipped")
        return
    if not rule:
        raise SystemExit("no pilot rule recorded (inputs.pilot_rule in core.yaml): ask Abdul")
    c = Condition("pilot", pil["arm_id"], 0, pil["channel"], pil["n"], pil["graph"], cfg["T"],
                  pil["last_k"], pil["runs"], cfg["seeds"]["base_seed"], build.params(cfg))
    build.check_cost(cfg, cfg["serving"]["timeout_minutes"])
    with build.gpu_session(cfg, ("model_confirmed", "prompts_approved", "modal_account")):
        be, _ = build.modal_backend(cfg, log_name="pilot")
        out = run_condition(c, be, REPO / "results" / "pilot")
    data = load_runs(out)
    con = stats.consensus(data.actions)
    summ = {"rule": rule, "payoffs": cfg["payoffs"], "X_t_mean": stats.frac_high(data.actions).mean(0).tolist(),
            "consensus": con.prob(), "tau": [None if x != x else x for x in con.tau.tolist()],
            "note": "pilot runs never enter the analysis"}
    (REPO / "results" / "pilot" / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
