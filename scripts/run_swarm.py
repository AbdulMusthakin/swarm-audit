#!/usr/bin/env python3
"""Launch the swarm runs of one arm (docs/handoff-build.md sections 3 and 9).

    python scripts/run_swarm.py --config core.yaml --arm core_counts [--n 10]
        [--backend modal | --backend scripted:copy] [--chunk 25]

With --backend modal (the default) the run needs Abdul's inputs in the config
(model, prompts, Modal account; and the pilot rule for core arms), and is bracketed by
the drift panel: queried at the start and end of the arm, and mixed once into a
full-size swarm batch. Logs: results/runs/<arm>/<condition>/run_<r>.jsonl.gz, plus
results/runs/<arm>/manifest.json. Runs already logged are skipped (resume).
Commit and push after each arm.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from swarm_audit import build, calibration, decisions  # noqa: E402
from swarm_audit.provenance import fingerprint, kernel_identity  # noqa: E402
from swarm_audit.swarm import run_condition, run_file  # noqa: E402


def check_pilot_frozen(cfg: dict) -> None:
    """Core arms launch only after G6 has frozen the payoffs (results/pilot/summary.json)."""
    path = REPO / "results" / "pilot" / "summary.json"
    if not path.exists():
        raise SystemExit("Gate G6 has not run (no results/pilot/summary.json): run scripts/run_pilot.py")
    summ = json.loads(path.read_text())
    frozen = summ.get("frozen_payoffs")
    if frozen is None:
        raise SystemExit("The pilot did not freeze the payoffs (see results/pilot/summary.json): ask Abdul")
    if {k: float(v) for k, v in frozen.items()} != {k: float(v) for k, v in cfg["payoffs"].items()}:
        raise SystemExit(f"config payoffs {cfg['payoffs']} differ from the frozen {frozen}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--n", type=int, nargs="*")
    ap.add_argument("--backend", default="modal")
    ap.add_argument("--chunk", type=int, help="runs per lockstep batch (default: all)")
    ap.add_argument("--runs", type=int, help="extend each condition to this many runs (resume "
                    "adds the missing run indices with their own seeds)")
    ap.add_argument("--runs-root", default=str(REPO / "results" / "runs"))
    args = ap.parse_args()
    cfg = build.load(args.config)
    if cfg.get("approved") is False:
        raise SystemExit("this arm needs Abdul's approval (approved: false in the config)")
    # approved: if_budget (arm 7) is enforced through the G4 plan below.
    conds = build.conditions(cfg, args.arm, args.n)
    if args.runs:
        from dataclasses import replace

        conds = [replace(c, runs=args.runs) for c in conds]
    root = Path(args.runs_root)

    if args.backend.startswith("scripted:"):
        be = build.scripted(args.backend.split(":", 1)[1], cfg)
        for c in conds:
            run_condition(c, be, root, args.chunk)
        return

    need = ("model_confirmed", "prompts_approved", "modal_account")
    floor = build.numerical_floor(cfg)              # G3 must have run
    pause_by = cfg["decisions"]["drift_pause_above_floor_by"]
    if args.arm.startswith(("core", "second")):
        need += ("pilot_rule",)
        check_pilot_frozen(cfg)
        build.check_in_plan(args.arm, [c.name for c in conds])
    build.check_cost(cfg, cfg["serving"]["timeout_minutes"] * len(conds))
    with build.gpu_session(cfg, need):
        be, _ = build.modal_backend(cfg, log_name=f"swarm_{args.arm}")
        if args.arm.startswith(("core", "second")):   # review C3: G5 must have passed
            calibration.require_g5(fingerprint(kernel_identity(be)), conds[0].channel)
        panel = build.drift_panel(cfg)
        for c in conds:
            t0 = time.time()
            start = build.panel_outputs(be, panel)
            todo = sum(not run_file(root / c.arm / c.name, r).exists() for r in range(c.runs))
            mixer = build.PanelMixer(be, panel,
                                     min_batch=c.n * max(1, min(todo, args.chunk or todo)))
            run_condition(c, mixer, root, args.chunk, extra_manifest={"drift_panel_start": start})
            end = build.panel_outputs(be, panel)
            drift = {"start_vs_end": build.panel_max_change(start, end),
                     "start_vs_mixed": (build.panel_max_change(start, mixer.mixed)
                                        if mixer.mixed else None),
                     "panel_size": len(panel)}
            sec = time.time() - t0
            usd = build.record_spend(f"{args.arm}/{c.name}", sec, cfg)
            path = root / args.arm / c.name / "drift.json"
            worst = max(v for v in (drift["start_vs_end"], drift["start_vs_mixed"]) if v is not None)
            action = decisions.drift_action(worst, floor, pause_by)
            drift.update(floor=floor, action=action)
            path.write_text(json.dumps({**drift, "start": start, "end": end,
                                        "mixed": mixer.mixed}) + "\n")
            print(f"{c.name}: drift {worst:.3g} (floor {floor:.3g}) -> {action}; {sec:.0f}s, "
                  f"about ${usd:.2f}", flush=True)
            if action == "report":
                print("DRIFT above the G3 floor: recorded; report it to Abdul at once "
                      "(runs continue, by the settled rule).", flush=True)
            if action == "pause":
                raise SystemExit(f"DRIFT {worst:.3g} is more than {pause_by} above the floor: "
                                 "paused by the settled rule; report to Abdul.")


if __name__ == "__main__":
    main()
