#!/usr/bin/env python3
"""Fit and score the audit tables of one channel (docs/handoff-build.md section 5).

    python scripts/run_audit.py --config core.yaml --channel cheap_talk [--d 2]
        [--enumerated] [--actions-only] [--backend modal | --backend scripted:copy]
        [--dry-run]

counts: the actions-only table. cheap talk: the signal-aware table, and the actions-only
table derived from it (primary weights and equal weights). --enumerated: the
calibration tables (one query per state, last round only). --actions-only: a directly
fitted actions-only table (K_10, arm 7). Tables go to results/audit/<backend>/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from swarm_audit import build  # noqa: E402
from swarm_audit.audit import table_states  # noqa: E402


def tag(channel: str, d: int, enumerated: bool, actions_only: bool) -> str:
    return f"{channel}_d{d}" + ("_enumerated" if enumerated else "") + ("_actions" if actions_only else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--channel", required=True, choices=["counts", "cheap_talk"])
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--enumerated", action="store_true")
    ap.add_argument("--actions-only", action="store_true")
    ap.add_argument("--backend", default="modal")
    ap.add_argument("--dry-run", action="store_true", help="print the number of queries and stop")
    args = ap.parse_args()
    cfg = build.load(args.config)
    own = cfg["own_signal_in_state"]
    kind = "actions" if args.channel == "counts" or args.actions_only else "signal"
    n_states = len(table_states(args.channel, args.d, kind, own))
    sz = cfg["audit"][kind]
    per_q = 3 if args.channel == "cheap_talk" else 1
    n_q = n_states if args.enumerated else 1 * 2 + (n_states - 1) * (sz["n_fit"] + sz["n_score"])
    print(f"{args.channel}, d = {args.d}, {kind} table: {n_states} states (incl. START), "
          f"{n_q} histories, {n_q * per_q} label read-outs")
    if args.dry_run:
        return
    t = tag(args.channel, args.d, args.enumerated, args.actions_only)
    if args.backend.startswith("scripted:"):
        be = build.scripted(args.backend.split(":", 1)[1], cfg)
        tables = build.build_tables(be, cfg, args.channel, args.d, args.enumerated, args.actions_only)
        build.save_tables(tables, build.table_dir(cfg, be.name), t)
        return
    if cfg.get("check_core_pilot"):
        build.check_core_pilot_payoffs(cfg)
    build.check_cost(cfg, cfg["serving"]["timeout_minutes"])
    with build.gpu_session(cfg, ("model_confirmed", "prompts_approved", "modal_account")):
        be, _ = build.modal_backend(cfg, log_name=f"audit_{t}")
        t0 = time.time()
        tables = build.build_tables(be, cfg, args.channel, args.d, args.enumerated, args.actions_only)
        sec = time.time() - t0
        usd = build.record_spend(f"audit/{t}", sec, cfg)
        folder = build.table_dir(cfg, cfg["model"]["name"])
        build.save_tables(tables, folder, t)
        print(f"wrote {folder}/{t}_*.json ({sec:.0f}s, about ${usd:.2f})")


if __name__ == "__main__":
    main()
