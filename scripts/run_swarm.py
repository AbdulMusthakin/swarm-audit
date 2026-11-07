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

from swarm_audit import build  # noqa: E402
from swarm_audit.swarm import run_condition  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--n", type=int, nargs="*")
    ap.add_argument("--backend", default="modal")
    ap.add_argument("--chunk", type=int, help="runs per lockstep batch (default: all)")
    ap.add_argument("--runs-root", default=str(REPO / "results" / "runs"))
    args = ap.parse_args()
    cfg = build.load(args.config)
    if cfg.get("approved") is False:
        raise SystemExit("this arm needs Abdul's approval (approved: false in the config)")
    conds = build.conditions(cfg, args.arm, args.n)
    root = Path(args.runs_root)

    if args.backend.startswith("scripted:"):
        be = build.scripted(args.backend.split(":", 1)[1], cfg)
        for c in conds:
            run_condition(c, be, root, args.chunk)
        return

    need = ("model_confirmed", "prompts_approved", "modal_account")
    if args.arm.startswith(("core", "second")):
        need += ("pilot_rule",)
    build.check_cost(cfg, cfg["serving"]["timeout_minutes"] * len(conds))
    with build.gpu_session(cfg, need):
        be, _ = build.modal_backend(cfg, log_name=f"swarm_{args.arm}")
        panel = build.drift_panel(cfg)
        for c in conds:
            t0 = time.time()
            start = build.panel_outputs(be, panel)
            mixer = build.PanelMixer(be, panel, min_batch=c.n * min(c.runs, args.chunk or c.runs))
            run_condition(c, mixer, root, args.chunk, extra_manifest={"drift_panel_start": start})
            end = build.panel_outputs(be, panel)
            drift = {"start_vs_end": build.panel_max_change(start, end),
                     "start_vs_mixed": (build.panel_max_change(start, mixer.mixed)
                                        if mixer.mixed else None),
                     "panel_size": len(panel)}
            sec = time.time() - t0
            usd = build.record_spend(f"{args.arm}/{c.name}", sec, cfg)
            path = root / args.arm / c.name / "drift.json"
            path.write_text(json.dumps({**drift, "start": start, "end": end,
                                        "mixed": mixer.mixed}) + "\n")
            print(f"{c.name}: drift {drift}; {sec:.0f}s, about ${usd:.2f}", flush=True)
            print("Report any drift above the G3 floor to Abdul at once "
                  "(results/validation.md).", flush=True)


if __name__ == "__main__":
    main()
