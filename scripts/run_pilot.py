#!/usr/bin/env python3
"""Gate G6, the pilot (docs/handoff-build.md section 4; rule in configs/pilot.yaml).

5 counts runs at n = 10 on the cycle with the current payoffs, logged to results/pilot/
(never analysed). Then Abdul's pre-registered rule is applied mechanically: if 3 or more
runs reach all-low consensus by round 5, uLH becomes 11 in every config (both channels)
and is frozen; otherwise nothing changes. If the alternative readings of the rule
disagree, nothing is changed and the script stops for Abdul.

    python scripts/run_pilot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from swarm_audit import build, pilot  # noqa: E402
from swarm_audit.swarm import Condition, load_runs, run_condition  # noqa: E402

OUT = REPO / "results" / "pilot"


def main() -> None:
    cfg = build.load("core.yaml")
    pil = build.load("pilot.yaml")
    if cfg["inputs"].get("pilot_rule") != pil["rule"]["name"]:
        raise SystemExit("inputs.pilot_rule in core.yaml does not name the rule in pilot.yaml")
    if (OUT / "summary.json").exists():
        raise SystemExit(f"{OUT / 'summary.json'} exists: the pilot runs once. Ask Abdul.")
    c = Condition("pilot", pil["arm_id"], 0, pil["channel"], pil["n"], pil["graph"], cfg["T"],
                  pil["last_k"], pil["runs"], cfg["seeds"]["base_seed"], build.params(cfg))
    build.check_cost(cfg, cfg["serving"]["timeout_minutes"])
    with build.gpu_session(cfg, ("model_confirmed", "prompts_approved", "modal_account")):
        be, _ = build.modal_backend(cfg, log_name="pilot")
        folder = run_condition(c, be, OUT)
    data = load_runs(folder)
    dec = pilot.decide(data.actions, pil["rule"])
    summ = {"rule": pil["rule"], "payoffs_in_pilot": cfg["payoffs"], **dec,
            "note": "pilot runs never enter the analysis"}
    if dec["ambiguous"]:
        summ["action"] = "none: the readings of the rule disagree; Abdul decides"
    elif dec["apply_change"]:
        newp = pilot.new_payoffs(cfg["payoffs"], pil["rule"]["change"])
        summ["frozen_payoffs"] = newp
        summ["configs_changed"] = pilot.freeze_payoffs(newp)
        summ["action"] = f"fallback applied and frozen for both channels: {pil['rule']['change']}"
    else:
        summ["frozen_payoffs"] = cfg["payoffs"]
        summ["action"] = "no change; payoffs frozen as they are"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(json.dumps(summ, indent=1))
    if dec["ambiguous"]:
        raise SystemExit("The readings of the pilot rule disagree for these runs: ask Abdul.")


if __name__ == "__main__":
    main()
