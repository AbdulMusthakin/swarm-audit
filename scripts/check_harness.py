#!/usr/bin/env python3
"""Gate G2, the harness check (CPU only): scripted policies through the LLM's code path.

For each policy and channel in configs/harness.yaml, at n = 4 and 10 on the cycle with
200 runs: swarm runs -> audit tables fitted on p_test -> predictors -> analysis, exactly
as for the LLM (only the backend differs). At n = 4 the swarm is also compared with the
exact law of the chain, and a last-round-only run exercises the gate-G5 checks against
an enumerated table.

Writes results/harness/{summary.json,expected_vs_observed.md}, Figure 1 per condition in
results/figures/harness/, and the G2 section of results/validation.md. Run logs go to
data/runs/harness/ (git-ignored; reproducible from the seeds).

    python scripts/check_harness.py [--runs 200] [--quick]
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

from swarm_audit import build, calibration, plots  # noqa: E402
from swarm_audit.analysis import Tables, analyse_condition, to_jsonable  # noqa: E402
from swarm_audit.prediction import ActionsChain, SignalChain  # noqa: E402
from swarm_audit.provenance import check_tables_match_runs  # noqa: E402
from swarm_audit.swarm import Condition, load_runs, run_condition  # noqa: E402
from swarm_audit.topology import make  # noqa: E402

ZERO = 1e-12
EXPECTED = {
    "coin": "ε = 0; Δ inside the noise floor",
    "copy": "actions-only ε = 0; Δ inside the noise floor (full history)",
    "two_round": "actions-only ε > 0; bound above the measured Δ",
    "hidden_signal": "actions-only ε > 0; signal-aware ε = 0; signal-aware Δ inside the floor",
}


def judge(policy: str, res: dict) -> tuple[bool, dict]:
    dfx = res["defects"]["actions_only"]
    sc = res["scores"]
    obs = {"eps_test_actions": dfx["eps_test_pooled"], "eps_swarm_actions": dfx["eps_swarm"],
           "max_delta_actions": float(np.max(sc["delta"]["actions_only"])),
           "floor_actions": sc["floor"]["actions_only"]["max_over_rounds"],
           "bound_T": float(dfx["bound"]["B"][-1])}
    if "signal_aware" in res["defects"]:
        dfs = res["defects"]["signal_aware"]
        obs.update(eps_test_signal=dfs["eps_test_pooled"], eps_swarm_signal=dfs["eps_swarm"],
                   max_delta_signal=float(np.max(sc["delta"]["signal_aware"])),
                   floor_signal=sc["floor"]["signal_aware"]["max_over_rounds"])
        obs["paired_D"] = sc["paired_D"]
    inside = obs["max_delta_actions"] <= obs["floor_actions"]
    if policy in ("coin", "copy"):
        ok = obs["eps_test_actions"] < ZERO and obs["eps_swarm_actions"] < ZERO and inside
        if "eps_test_signal" in obs:
            ok = ok and obs["eps_test_signal"] < ZERO
    elif policy == "two_round":
        # The bound is on the true TV; measured Delta_t carries sampling noise (in round 1
        # the bound is 0, since START is exact, while Delta_1 is noise alone). So: the
        # bound plus the per-round noise floor covers Delta_t in every round, and the
        # bound alone is above Delta_t from round 2 on.
        B = np.asarray(dfx["bound"]["B"])
        D = np.asarray(sc["delta"]["actions_only"])
        fl = np.asarray(sc["floor"]["actions_only"]["per_round"])
        obs["bound_plus_floor_covers_delta"] = bool(np.all(D <= B + fl))
        obs["bound_above_delta_rounds_2_on"] = bool(np.all(B[1:] >= D[1:]))
        ok = (obs["eps_test_actions"] > 0.01 and obs["bound_plus_floor_covers_delta"]
              and obs["bound_above_delta_rounds_2_on"])
    else:
        ok = (obs["eps_test_actions"] > 0.01 and obs["eps_test_signal"] < ZERO
              and obs["eps_swarm_signal"] < ZERO
              and obs["max_delta_signal"] <= obs["floor_signal"])
    return bool(ok), obs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="harness.yaml")
    ap.add_argument("--runs", type=int)
    ap.add_argument("--quick", action="store_true", help="smaller sizes, for a fast check")
    ap.add_argument("--out", default=str(REPO / "results" / "harness"))
    ap.add_argument("--runs-dir", default=str(REPO / "data" / "runs"))
    args = ap.parse_args()
    cfg = build.load(args.config)
    if args.quick:
        cfg["audit"] = {"actions": {"n_fit": 20, "n_score": 10}, "signal": {"n_fit": 10, "n_score": 5}}
        cfg["analysis"].update(n_model=4000, bootstrap=300, ks_sims=4000)
    runs = args.runs or (40 if args.quick else cfg["runs"])
    an = cfg["analysis"]
    out, runs_dir = Path(args.out), Path(args.runs_dir)
    pt = build.ptest(cfg)
    rows, summary = [], {"config": {k: v for k, v in cfg.items() if not k.startswith("_")},
                         "runs": runs, "conditions": {}}
    t_all = time.time()
    for policy, channels in cfg["policies"].items():
        be = build.scripted(policy, cfg)
        for ch in channels:
            tables = build.build_tables(be, cfg, ch, 2)
            build.save_tables(tables, out / "audit", f"{policy}_{ch}_d2")
            tb = Tables(tables["actions"], tables.get("signal"), tables.get("actions_equal"))
            for n in cfg["n"]:
                t0 = time.time()
                g = make(cfg["graph"], n)
                cond = Condition(f"harness_{policy}", cfg["arm_id"], n * 10 + (ch == "cheap_talk"),
                                 ch, n, cfg["graph"], cfg["T"], None, runs,
                                 cfg["seeds"]["base_seed"], build.params(cfg),
                                 cfg["own_signal_in_state"])
                data = load_runs(run_condition(cond, be, runs_dir, log=lambda *a: None))
                check_tables_match_runs(data.meta["kernel_fingerprint"], tables)
                res = analyse_condition(data, g, tb, an["n_model"], an["seed"], an["bootstrap"], pt)
                ok, obs = judge(policy, res)
                name = f"{policy}/{ch}/n{n}"
                entry = {"expected": EXPECTED[policy], "pass": ok, "observed": obs}
                if n == 4:  # exact law of the chain the policy should match
                    chain = (SignalChain(tables["signal"].lookup(), 2, cfg["own_signal_in_state"])
                             if policy == "hidden_signal" else ActionsChain(tables["actions"].lookup(), 2))
                    ex = calibration.per_run(data, chain, g, an["ks_sims"])
                    entry["exact_law"] = ex
                    if policy != "two_round":
                        entry["pass"] = ok = ok and ex["pass"]
                    # Calibration code path (G5) on a last-round-only run.
                    cal_cond = Condition(f"harness_{policy}", cfg["arm_id"], 1000 + n * 10 + (ch == "cheap_talk"),
                                         ch, n, cfg["graph"], cfg["T"], 1, runs,
                                         cfg["seeds"]["base_seed"], build.params(cfg),
                                         cfg["own_signal_in_state"])
                    cal = load_runs(run_condition(cal_cond, be, runs_dir, log=lambda *a: None))
                    etab = build.build_tables(be, cfg, ch, 2, enumerated=True)
                    g5 = calibration.check_arm(cal, etab, g, floor=ZERO, sims=an["ks_sims"])
                    entry["calibration_G5"] = g5
                    entry["pass"] = ok = ok and g5["per_decision"]["pass"] and g5["per_run"]["pass"]
                summary["conditions"][name] = to_jsonable({**entry, "analysis": res})
                plots.figure1(res, REPO / "results" / "figures" / "harness" / f"{policy}_{ch}_n{n}.png",
                              f"Harness: {policy}, {ch}, n = {n}, {runs} runs")
                rows.append((name, entry))
                print(f"{name}: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.1f}s)", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary) + "\n")
    md = table_md(rows, runs)
    (out / "expected_vs_observed.md").write_text(md)
    write_validation_g2(md, all(e["pass"] for _, e in rows), runs, time.time() - t_all)
    print(md)


def f(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def table_md(rows, runs) -> str:
    lines = [f"Scripted policies, cycle, T = 20, {runs} runs per condition; predictors simulated "
             "20,000 times. ε(p_test) is pooled over states with p_test's weights. "
             "Δ is the largest binned TV over rounds; the floor is the 95th percentile of the "
             "same maximum between two model samples of the swarm's size.", "",
             "| condition | expected | ε_test actions | ε_test signal | max Δ actions (floor) | "
             "max Δ signal (floor) | bound B_T | exact law, n = 4 (gap / q99) | G5 path, n = 4 | result |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, e in rows:
        o = e["observed"]
        ex = e.get("exact_law")
        g5 = e.get("calibration_G5")
        ex_s = f"{f(ex['gap'])} / {f(ex['threshold_q99'])} {'✓' if ex['pass'] else '✗'}" if ex else "–"
        g5_s = ("✓" if g5["per_decision"]["pass"] and g5["per_run"]["pass"] else "✗") if g5 else "–"
        sig = (f"{f(o['max_delta_signal'])} ({f(o['floor_signal'])})"
               if "max_delta_signal" in o else "–")
        lines.append(f"| {name} | {e['expected']} | {f(o['eps_test_actions'], 4)} | "
                     f"{f(o.get('eps_test_signal'), 4)} | {f(o['max_delta_actions'])} "
                     f"({f(o['floor_actions'])}) | {sig} | {f(o['bound_T'], 2)} | {ex_s} | {g5_s} | "
                     f"{'PASS' if e['pass'] else 'FAIL'} |")
    pd_rows = [(n, e["observed"]["paired_D"]) for n, e in rows if "paired_D" in e["observed"]]
    if pd_rows:
        lines += ["", "Paired CRPS difference D = CRPS(actions-only) − CRPS(signal-aware) "
                  "(> 0 favours keeping the signals), 95% whole-run bootstrap:", "",
                  "| condition | D | 95% interval |", "|---|---|---|"]
        lines += [f"| {n} | {f(d['mean'], 4)} | [{f(d['lo'], 4)}, {f(d['hi'], 4)}] |" for n, d in pd_rows]
    return "\n".join(lines) + "\n"


def write_validation_g2(md: str, ok: bool, runs: int, seconds: float) -> None:
    path = REPO / "results" / "validation.md"
    text = path.read_text() if path.exists() else ""
    start, end = "<!-- G2 start -->", "<!-- G2 end -->"
    block = (f"{start}\n## G2, harness check (CPU): {'PASS' if ok else 'FAIL'}\n\n"
             f"Generated by `scripts/check_harness.py` ({seconds:.0f} s). Every scripted policy "
             "goes through the same swarm, audit, prediction and analysis code as the LLM. "
             "Probabilities are logged, never re-queried. Details: "
             "`results/harness/summary.json`.\n\n" + md + f"\n{end}")
    if start in text:
        text = text[:text.index(start)] + block + text[text.index(end) + len(end):]
    else:
        text = text + ("\n" if text else "") + block + "\n"
    path.write_text(text)


if __name__ == "__main__":
    main()
