#!/usr/bin/env python3
"""C1 of docs/overnight-core-plan.md (exploratory): what endpoint scoring misses.

For one condition: reproduces the published endpoint CRPS and paired D from the original
analysis (same seeds), adds round-block CRPS of X_t (rounds 2-7, 8-13, 14-20), signed and
absolute one-step errors by round and coarse state, Monte Carlo sensitivity (10 independent
simulation batches) and audit-table sensitivity (resampled fitting histories, with a nested
run x table x simulation interval). CPU only; reads logged runs and the original tables.

    python scripts/followup_core_temporal.py \
        --runs results/runs/core_cheap_talk/cheap_talk_n10_cycle_full \
        --tables results/audit/llama-3.1-8b --tag cheap_talk_d2 --status primary
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from swarm_audit import build, stats  # noqa: E402
from swarm_audit import followup as fu  # noqa: E402
from swarm_audit.analysis import build_predictors, to_jsonable  # noqa: E402
from swarm_audit.prediction import Order0  # noqa: E402
from swarm_audit.provenance import check_tables_match_runs  # noqa: E402
from swarm_audit.swarm import git_commit, graph_of, load_runs  # noqa: E402

OUT = REPO / "results" / "followup" / "core"


def scores_from_pools(pools: dict, data) -> dict[str, dict[str, np.ndarray]]:
    """Per-run scores: endpoint and each round block, for every predictor. A pool is the
    simulated runs, or an Order0 chain (exact binomial laws, as in the original analysis)."""
    out = {}
    y = stats.high_count(data.actions)
    K = stats.WINDOW * data.n
    for name, A in pools.items():
        if isinstance(A, Order0):
            Gx, Gy = fu.order0_cdfs(A.p, data.n, data.T, stats.WINDOW)
            rc = fu.round_crps_cdf(Gx, data.actions)
        else:
            Gy = stats.cdf_from_samples(stats.high_count(A), K)
            rc = fu.round_crps(A, data.actions)
        out[name] = {"endpoint": stats.crps(Gy, y), **fu.block_means(rc), "_round": rc.mean(0)}
    return out


def mean_x(A, T: int) -> np.ndarray:
    return np.full(T, A.p) if isinstance(A, Order0) else stats.frac_high(A).mean(0)


def contrasts(names) -> list[tuple[str, str]]:
    pairs = []
    if "signal_aware" in names:
        pairs += [("actions_only", "signal_aware"), ("actions_only_equal_w", "signal_aware")]
    pairs += [(m, "order0") for m in names if m != "order0"]
    return pairs


SCORE_KEYS = ("endpoint", "2-7", "8-13", "14-20")


def summarise(sc: dict, B: int, rng) -> dict:
    res = {"crps": {}, "paired": {}, "crps_by_round": {}}
    for m, v in sc.items():
        res["crps"][m] = {k: fu.boot_mean(v[k], B, rng) for k in SCORE_KEYS}
        res["crps_by_round"][m] = v["_round"]
    for a, b in contrasts(list(sc)):
        res["paired"][f"{a} - {b}"] = {k: fu.boot_mean(sc[a][k] - sc[b][k], B, rng)
                                       for k in SCORE_KEYS}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="core.yaml")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--status", default="secondary", choices=["primary", "secondary"])
    ap.add_argument("--mc-batches", type=int, default=10)
    ap.add_argument("--table-resamples", type=int, default=200)
    ap.add_argument("--published", help="the original analysis JSON to reproduce")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    t0 = time.time()
    cfg = build.load(args.config)
    an, pt = cfg["analysis"], build.ptest(cfg)
    B = an["bootstrap"]
    data = load_runs(Path(args.runs))
    g = graph_of(data)
    d = g.regular_degree
    T = data.T
    tables = build.load_tables(Path(args.tables), args.tag)
    check_tables_match_runs(data.meta["kernel_fingerprint"],
                            {"actions": tables.actions, "signal": tables.signal,
                             "actions_equal": tables.actions_equal})
    arm, cond = Path(args.runs).parent.name, Path(args.runs).name
    published = Path(args.published) if args.published else \
        REPO / "results" / "analysis" / arm / f"{cond}.json"
    log: list[str] = []

    # 1. Reproduce the original predictors and endpoint scores (same seeds).
    preds = build_predictors(tables, g, T, an["n_model"], an["seed"] + 1, pt=pt)
    pools0 = {k: (fu.ChainSet.from_tables(tables.signal, tables.actions if tables.signal is None
                                          else None, d, pt).chains["order0"]
                  if k == "order0" else m.A) for k, m in preds.items()}
    sc0 = scores_from_pools(pools0, data)
    repro = {"published": str(published.relative_to(REPO)) if published.exists() else None}
    if published.exists():
        pub = json.loads(published.read_text())["scores"]
        diffs = {m: float(np.max(np.abs(np.asarray(pub["crps_per_run"][m]) - sc0[m]["endpoint"])))
                 for m in sc0 if m in pub["crps_per_run"]}
        repro["max_abs_diff_per_run_crps"] = diffs
        if "paired_D" in pub:
            Dm = float((sc0["actions_only"]["endpoint"] - sc0["signal_aware"]["endpoint"]).mean())
            repro["paired_D_published"] = pub["paired_D"]["mean"]
            repro["paired_D_reproduced"] = Dm
            repro["paired_D_abs_diff"] = abs(Dm - pub["paired_D"]["mean"])
        repro["exact"] = all(v < 1e-12 for v in diffs.values())
        if not repro["exact"]:
            log.append(f"DISCREPANCY: reproduced per-run CRPS differs from {published}: {diffs}")
    rng = fu.seeded(1)
    original = summarise(sc0, B, rng)
    original["delta_max"] = {}

    # Signed and absolute one-step errors (teacher-forced).
    errs = {"actions_only": fu.one_step_errors(data.p_H, data.state, fu.kappa_H_of(tables.actions))}
    if tables.signal is not None:
        errs["signal_aware"] = fu.one_step_errors(data.p_H, data.state_sig,
                                                  fu.kappa_H_of(tables.signal))
    one_step = {k: fu.error_tables(e, data.state, B, fu.seeded(2)) for k, e in errs.items()}

    # 2. Monte Carlo sensitivity: independent simulation batches of the same size.
    base = fu.ChainSet.from_tables(tables.signal, None if tables.signal is not None else tables.actions, d, pt)
    mc = {"batches": [], "n_model": an["n_model"]}
    pooled_pools: dict[str, list] = {m: [] for m in base.chains}
    for b in range(args.mc_batches):
        pools = base.simulate(g, T, an["n_model"], fu.seeded(101, b))
        for m, A in pools.items():
            pooled_pools[m].append(A)
        sc = scores_from_pools(pools, data)
        row = {"batch": b, "crps": {m: {k: float(v[k].mean()) for k in SCORE_KEYS} for m, v in sc.items()},
               "paired": {f"{a} - {c}": {k: float((sc[a][k] - sc[c][k]).mean()) for k in SCORE_KEYS}
                          for a, c in contrasts(list(sc))}}
        mc["batches"].append(row)
    mc["summary"] = {}
    for key in mc["batches"][0]["paired"]:
        for k in SCORE_KEYS:
            v = np.array([r["paired"][key][k] for r in mc["batches"]])
            mc["summary"].setdefault(key, {})[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                                                    "min": float(v.min()), "max": float(v.max())}
    # Pooled (10 x n_model runs per predictor).
    big = {m: (v[0] if isinstance(v[0], Order0) else np.concatenate(v))
           for m, v in pooled_pools.items()}
    scb = scores_from_pools(big, data)
    mc["pooled"] = {"n_model": int(args.mc_batches * an["n_model"]),
                    **summarise(scb, B, fu.seeded(3))}
    if "actions_only - signal_aware" in mc["summary"]:
        pubD = original["paired"]["actions_only - signal_aware"]["endpoint"]
        half = (pubD["hi"] - pubD["lo"]) / 2
        sd = mc["summary"]["actions_only - signal_aware"]["endpoint"]["sd"]
        thr = 0.1 * half
        mc["materiality"] = {"rule": "material if the SD of per-batch endpoint D exceeds 10% of the "
                                     "half-width of the conditional interval (plan C1.3)",
                             "half_width": half, "threshold": thr, "batch_sd": sd,
                             "material": bool(sd > thr),
                             "pooled_sd": sd / np.sqrt(args.mc_batches),
                             "pooled_below_threshold": bool(sd / np.sqrt(args.mc_batches) <= thr)}
        if sd > thr and sd / np.sqrt(args.mc_batches) > thr:
            need = int(an["n_model"] * np.ceil((sd / thr) ** 2))
            need = min(need, 2_000_000)
            pools = base.simulate(g, T, need, fu.seeded(104))
            sc = scores_from_pools(pools, data)
            mc["precision_run"] = {"n_model": need, **summarise(sc, B, fu.seeded(5))}
    del big, pooled_pools

    # 3. Audit-table sensitivity (resampled fitting histories) and the nested interval.
    tab = {"resamples": args.table_resamples, "n_model": an["n_model"], "rows": []}
    xcurves: dict[str, list] = {m: [] for m in base.chains}
    for j in range(args.table_resamples):
        r = fu.seeded(202, j)
        if tables.signal is not None:
            cs = fu.ChainSet.from_tables(fu.resample_table(tables.signal, r), None, d, pt)
        else:
            cs = fu.ChainSet.from_tables(None, fu.resample_table(tables.actions, r), d, pt)
        pools = cs.simulate(g, T, an["n_model"], r)
        sc = scores_from_pools(pools, data)
        idx = r.integers(data.R, size=data.R)          # nested: one whole-run resample
        row = {"j": j,
               "crps": {m: {k: float(v[k].mean()) for k in SCORE_KEYS} for m, v in sc.items()},
               "paired": {f"{a} - {c}": {k: float((sc[a][k] - sc[c][k]).mean()) for k in SCORE_KEYS}
                          for a, c in contrasts(list(sc))},
               "paired_nested": {f"{a} - {c}": {k: float((sc[a][k] - sc[c][k])[idx].mean())
                                                for k in SCORE_KEYS}
                                 for a, c in contrasts(list(sc))}}
        tab["rows"].append(row)
        for m, A in pools.items():
            xcurves[m].append(mean_x(A, T))
    tab["summary"] = {}
    for kind in ("paired", "paired_nested"):
        for key in tab["rows"][0][kind]:
            for k in SCORE_KEYS:
                v = np.array([rr[kind][key][k] for rr in tab["rows"]])
                ent = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                       "lo": float(np.quantile(v, 0.025)), "hi": float(np.quantile(v, 0.975)),
                       "frac_positive": float((v > 0).mean())}
                if kind == "paired":
                    sd_mc = mc["summary"][key][k]["sd"]
                    ent["table_only_sd_approx"] = float(np.sqrt(max(ent["sd"] ** 2 - sd_mc ** 2, 0.0)))
                tab["summary"].setdefault(kind, {}).setdefault(key, {})[k] = ent
    tab["x_band"] = {m: {"lo": np.quantile(np.array(v), 0.025, 0), "hi": np.quantile(np.array(v), 0.975, 0)}
                     for m, v in xcurves.items()}
    tab["labels"] = {
        "paired": "D under each resampled table, scored on the same 50 runs: table estimation "
                  "+ simulation noise (conditional on the runs)",
        "paired_nested": "each table resample paired with one whole-run resample: run sampling + "
                         "table estimation + simulation; excludes kernel drift, prompt and model "
                         "choice, so it is not total uncertainty",
        "table_only_sd_approx": "sqrt(SD_resample^2 - SD_MC^2), an approximation",
    }

    # Observed and predicted X_t, binned TV (as published), consensus.
    xs = stats.frac_high(data.actions)
    idx = fu.seeded(6).integers(data.R, size=(B, data.R))
    bs = xs[idx].mean(1)
    X = {"swarm": {"mean": xs.mean(0), "lo": np.quantile(bs, 0.025, 0), "hi": np.quantile(bs, 0.975, 0)},
         **{m: {"mean": mean_x(A, T)} for m, A in pools0.items()}}
    cons = {"swarm": stats.consensus(data.actions),
            **{m: stats.consensus(A) for m, A in pools0.items() if not isinstance(A, Order0)}}
    if published.exists():
        pubsc = json.loads(published.read_text())["scores"]
        original["delta_max"] = {m: {"max_delta": float(max(pubsc["delta"][m])),
                                     "simultaneous_null_q95": pubsc["floor"][m]["max_over_rounds"],
                                     "rounds_above_pointwise": [t + 1 for t, (dv, fv) in enumerate(
                                         zip(pubsc["delta"][m], pubsc["floor"][m]["per_round"])) if dv > fv]}
                                 for m in pubsc["delta"]}

    res = {"label": "exploratory follow-up (docs/overnight-core-plan.md C1); "
                    f"{args.status} condition",
           "status": args.status,
           "condition": {"arm": arm, "condition": cond, "channel": data.channel, "n": data.n,
                         "graph": data.graph, "T": T, "runs": data.R,
                         "run_fingerprint": data.meta["fingerprint"],
                         "kernel_fingerprint": data.meta["kernel_fingerprint"]},
           "inputs": {"runs": args.runs, "tables": args.tables, "tag": args.tag,
                      "config": args.config},
           "reproduction": repro, "original_simulation": original, "one_step": one_step,
           "monte_carlo": mc, "table_resampling": tab, "X": X,
           "consensus": {k: {"prob": c.prob(), "survival": c.survival(),
                             "definition": "first t <= T-2 with X_t = X_{t+1} = X_{t+2} in {0, 1}; "
                                           "none: censored at T"} for k, c in cons.items()},
           "log": log, "git_commit": git_commit(), "seconds": time.time() - t0,
           "time": dt.datetime.now(dt.timezone.utc).isoformat()}
    out = Path(args.out) / "temporal"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{arm}__{cond}.json"
    path.write_text(json.dumps(to_jsonable(res)) + "\n")

    # CSV rows (one condition); the collector merges them into temporal_scores.csv.
    rows = []
    for m, v in original["crps"].items():
        for k in SCORE_KEYS:
            rows.append({"arm": arm, "condition": cond, "status": args.status, "quantity": f"CRPS {m}",
                         "block": k, "mean": v[k]["mean"], "lo": v[k]["lo"], "hi": v[k]["hi"],
                         "interval": "95% whole-run bootstrap, conditional on tables and simulation"})
    for key, v in original["paired"].items():
        for k in SCORE_KEYS:
            rows.append({"arm": arm, "condition": cond, "status": args.status, "quantity": f"paired {key}",
                         "block": k, "mean": v[k]["mean"], "lo": v[k]["lo"], "hi": v[k]["hi"],
                         "interval": "95% whole-run bootstrap, conditional on tables and simulation"})
            t_ = tab["summary"]["paired_nested"][key][k]
            rows.append({"arm": arm, "condition": cond, "status": args.status,
                         "quantity": f"paired {key} (nested)", "block": k, "mean": t_["mean"],
                         "lo": t_["lo"], "hi": t_["hi"],
                         "interval": "95% nested: runs x tables x simulation"})
    with open(out / f"{arm}__{cond}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({time.time() - t0:.0f}s)")
    if repro.get("paired_D_reproduced") is not None:
        print(f"reproduction: D {repro['paired_D_reproduced']:.8f} vs published "
              f"{repro['paired_D_published']:.8f} (exact: {repro['exact']})")
    for key, v in original["paired"].items():
        print(f"  {key}: " + "; ".join(f"{k} {v[k]['mean']:+.5f} [{v[k]['lo']:+.5f}, {v[k]['hi']:+.5f}]"
                                        for k in SCORE_KEYS))
    if "materiality" in mc:
        print("  MC:", {k: (round(v, 7) if isinstance(v, float) else v) for k, v in mc["materiality"].items()
                        if k != "rule"})
    for line in log:
        print(line)


if __name__ == "__main__":
    main()
