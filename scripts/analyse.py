#!/usr/bin/env python3
"""Analysis of logged swarm runs against the audit tables (docs/handoff-build.md section 7).

One condition:
    python scripts/analyse.py --runs results/runs/core_cheap_talk/cheap_talk_n10_cycle_full \
        --tables results/audit/llama-3.1-8b --tag cheap_talk_d2

Gate G5 on a calibration arm (enumerated tables; floor from G3):
    python scripts/analyse.py --runs results/runs/calibration_counts/counts_n4_cycle_last1 \
        --tables results/audit/llama-3.1-8b --tag counts_d2_enumerated --gate-g5        # floor from G3 (configs)

Figure 2 and the summary tables over every analysed condition:
    python scripts/analyse.py --collect

Outputs: results/analysis/<arm>/<condition>.json, results/figures/fig1_<arm>_<condition>.png,
results/figures/fig2.png, results/tables/*.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from swarm_audit import build, calibration, plots  # noqa: E402
from swarm_audit.analysis import analyse_condition, to_jsonable  # noqa: E402
from swarm_audit.provenance import ProvenanceError, check_tables_match_runs  # noqa: E402
from swarm_audit.swarm import graph_of, load_runs  # noqa: E402

OUT = REPO / "results" / "analysis"


def one(args) -> None:
    cfg = build.load(args.config)
    an = cfg["analysis"]
    data = load_runs(Path(args.runs))
    g = graph_of(data)
    root = build.results_root(cfg)
    OUT = root / "analysis"
    tables = build.load_tables(Path(args.tables), args.tag)
    try:   # review C2: tables must come from the runs' kernel (prompts, model, payoffs...)
        check_tables_match_runs(data.meta.get("kernel_fingerprint"),
                                {"actions": tables.actions, "signal": tables.signal,
                                 "actions_equal": tables.actions_equal})
    except ProvenanceError as e:
        raise SystemExit(f"Stopping: {e}")
    arm = Path(args.runs).parent.name
    cond = Path(args.runs).name
    if args.gate_g5:
        floor = args.floor if args.floor is not None else build.numerical_floor(cfg)
        tabs = {"actions": tables.actions, "signal": tables.signal}
        res = calibration.check_arm(data, tabs, g, floor, an["ks_sims"])
        path = OUT / arm / f"{cond}_G5.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_jsonable(res), indent=1) + "\n")
        print(json.dumps(to_jsonable(res), indent=1))
        rec = calibration.gate_record(to_jsonable(res), data, tabs, floor)
        gpath = calibration.write_gate(rec, build.gates_dir(cfg))
        print(f"G5 {data.channel}: {'PASS' if rec['pass'] else 'FAIL'} ({rec['required']}); "
              f"record {gpath}")
        if not rec["pass"]:     # review C3: a failed required check is a failed command
            raise SystemExit(1)
        return
    res = analyse_condition(data, g, tables, an["n_model"], an["seed"], an["bootstrap"],
                            build.ptest(cfg))
    res["inputs"] = {"runs": args.runs, "tables": args.tables, "tag": args.tag,
                     "probabilities": "logged (never re-queried)"}
    path = OUT / arm / f"{cond}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(res)) + "\n")
    plots.figure1(res, root / "figures" / f"fig1_{arm}_{cond}.png", f"{arm}: {cond}")
    print(f"wrote {path}")
    sc = res["scores"]
    for k, v in sc["crps"].items():
        print(f"  CRPS {k}: {v['mean']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}]")
    if "paired_D" in sc:
        D = sc["paired_D"]
        print(f"  paired D: {D['mean']:.4f} [{D['lo']:.4f}, {D['hi']:.4f}]")


def collect(root: Path | None = None) -> None:
    """Figure 2 and the summary tables over the analysed conditions under `root`
    (a config's results_dir; default results/)."""
    import numpy as np

    root = root or REPO / "results"
    OUT = root / "analysis"
    rows = []
    for p in sorted(OUT.glob("*/*.json")):
        if p.name.endswith("_G5.json"):
            continue
        res = json.loads(p.read_text())
        c = res["condition"]
        rows.append({"name": f"{p.parent.name}/{p.stem}", "channel": c["channel"], "n": c["n"],
                     "res": _arrays(res)})
    if not rows:
        raise SystemExit(f"no analysed conditions in {OUT}")
    plots.figure2(rows, root / "figures" / "fig2.png")
    tdir = root / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    lines = ["| condition | predictor | CRPS (95% CI) | E_swarm F − E_model F (95% CI) | max Δ_t | floor |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        sc = r["res"]["scores"]
        for k, v in sc["crps"].items():
            e = sc["F_error"][k]
            lines.append(f"| {r['name']} | {k} | {v['mean']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}] | "
                         f"{e['mean']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}] | "
                         f"{np.max(sc['delta'][k]):.3f} | {sc['floor'][k]['max_over_rounds']:.3f} |")
    (tdir / "scores.md").write_text("\n".join(lines) + "\n")
    lines = ["| condition | table | ε(p_test) | ε(p_swarm) | B_T | first saturation |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        for k, d in r["res"]["defects"].items():
            lines.append(f"| {r['name']} | {k} | {d['eps_test_pooled']:.4f} | {d['eps_swarm']:.4f} | "
                         f"{d['bound']['B'][-1]:.2f} | {d['bound']['first_saturation_round']} |")
    (tdir / "defects.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {root / 'figures' / 'fig2.png'} and {tdir}/scores.md, defects.md")


def _arrays(x):
    import numpy as np

    if isinstance(x, dict):
        return {k: _arrays(v) for k, v in x.items()}
    if isinstance(x, list) and x and all(isinstance(v, (int, float)) or v is None for v in x):
        return np.array([np.nan if v is None else v for v in x], float)
    if isinstance(x, list):
        return [_arrays(v) for v in x]
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="core.yaml")
    ap.add_argument("--runs")
    ap.add_argument("--tables")
    ap.add_argument("--tag")
    ap.add_argument("--gate-g5", action="store_true")
    ap.add_argument("--floor", type=float, help="override the G3 floor in the config")
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    if args.collect:
        collect(build.results_root(build.load(args.config)))
    elif args.runs and args.tables and args.tag:
        one(args)
    else:
        ap.error("give --runs, --tables and --tag, or --collect")


if __name__ == "__main__":
    main()
