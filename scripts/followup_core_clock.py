#!/usr/bin/env python3
"""C2 of docs/overnight-core-plan.md (exploratory): a clock-indexed audit.

The original table pools synthetic histories of lengths 1..19. The clock-indexed audit fits
a table per length block (1-6, 7-12, 13-19) from the same stored fitting histories and, at
round t >= 2, uses the block containing L = t - 1: a time-inhomogeneous composed chain. It
uses no swarm data. Compared with the unchanged pooled audit on the same runs; the primary
follow-up metric is the paired early-round (2-7) CRPS improvement, pooled - clock, of the
signal-aware predictor. Chosen after the original mismatch was seen: exploratory.

    python scripts/followup_core_clock.py \
        --runs results/runs/core_cheap_talk/cheap_talk_n10_cycle_full \
        --tables results/audit/llama-3.1-8b --tag cheap_talk_d2 --status primary
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from swarm_audit import build, stats  # noqa: E402
from swarm_audit import followup_core as fu  # noqa: E402
from swarm_audit.analysis import to_jsonable  # noqa: E402
from swarm_audit.audit import AuditTable, derive_actions_table, marginal_weights, state_weights  # noqa: E402
from swarm_audit.ladder import render  # noqa: E402
from swarm_audit.prediction import ActionsChain, SignalChain  # noqa: E402
from swarm_audit.provenance import check_tables_match_runs  # noqa: E402
from swarm_audit.states import START, parse, record_for_state  # noqa: E402
from swarm_audit.swarm import git_commit, graph_of, load_runs  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from followup_core_temporal import SCORE_KEYS, scores_from_pools  # noqa: E402

OUT = REPO / "results" / "followup" / "core"


def chains_for(signal: AuditTable | None, actions: AuditTable | None, d: int) -> dict:
    """The audit chains of one table set (no order-zero: it has no clock)."""
    if signal is not None:
        return {"signal_aware": SignalChain(signal.lookup(), d, signal.own_signal),
                "actions_only": ActionsChain(derive_actions_table(signal, "primary").lookup(), d),
                "actions_only_equal_w": ActionsChain(derive_actions_table(signal, "equal").lookup(), d)}
    return {"actions_only": ActionsChain(actions.lookup(), d)}


def clock_set(base: AuditTable, d: int, cheap: bool) -> tuple[dict, dict]:
    blocks, info = [], []
    for b in range(len(fu.LENGTH_BLOCKS)):
        t, inf = fu.clock_table(base, b)
        blocks.append(t)
        info.append(inf)
    per = [chains_for(t, None, d) if cheap else chains_for(None, t, d) for t in blocks]
    clock = {m: fu.ClockChain([p[m] for p in per], m) for m in per[0]}
    return clock, {"tables": blocks, "info": info}


def weighted_eps(t: AuditTable, pt, kind_marginal: bool = False) -> dict:
    """eps on the entries' scoring sets, p_test-weighted over non-START states (renormalised
    over states with scoring histories). kind_marginal: the action marginal |p_H - kappa_H|."""
    w = state_weights([parse(k) for k in t.entries], t.d, pt, include_start=False)
    tot, wsum, n = 0.0, 0.0, 0
    for k, e in t.entries.items():
        if k == "START" or len(e.get("score", [])) == 0:
            continue
        sc = np.asarray(e["score"], float)
        if t.kind == "signal" and kind_marginal:
            ph = sc.reshape(len(sc), 2, 3)[:, 0, :].sum(1)
            ek = float(np.mean(np.abs(ph - t.kappa_H(k))))
        elif t.kind == "signal":
            ek = float(np.mean(0.5 * np.abs(sc.reshape(len(sc), -1) - np.asarray(e["kappa"]).reshape(-1)).sum(1)))
        else:
            ek = float(np.mean(np.abs(sc - e["kappa"])))
        tot += w[k] * ek
        wsum += w[k]
        n += len(sc)
    return {"eps": tot / wsum if wsum else None, "scoring_histories": n}


def derived_eps(sig: AuditTable, scheme: str, pt) -> dict:
    """eps of a derived actions-only table on the signal-aware scoring sets (the original
    derive_actions_table's definition), renormalised over states with scoring histories."""
    W = marginal_weights(sig.d, sig.own_signal, scheme)
    der = derive_actions_table(sig, scheme)
    pt_w = state_weights([parse(k) for k in der.entries], sig.d, pt, include_start=False)
    tot, wsum = 0.0, 0.0
    for s, ws in W.items():
        if s == "START":
            continue
        e_s, w_s = 0.0, 0.0
        for sp, w in ws.items():
            sc = np.asarray(sig.entries[sp].get("score", []), float)
            if len(sc) == 0:
                continue
            ph = sc.reshape(len(sc), 2, 3)[:, 0, :].sum(1)
            e_s += w * float(np.mean(np.abs(ph - der.kappa_H(s))))
            w_s += w
        if w_s:
            tot += pt_w[s] * e_s / w_s
            wsum += pt_w[s]
    return {"eps": tot / wsum if wsum else None}


def prompt_identity(cfg: dict, channel: str, d: int, own: bool) -> dict:
    """Is the L = 1 full-history prompt byte-identical to the last-round-only prompt, for
    every state? (Round-2 sensitivity is valid only then.)"""
    from swarm_audit.states import all_actions_states, all_signal_states

    lab, par = build.labels(cfg), build.params(cfg)
    states = all_signal_states(d, own) if channel == "cheap_talk" else all_actions_states(d)
    same = 0
    diff = []
    for s in states:
        if s == START:
            continue
        h = (record_for_state(s, d),)
        a = render(channel, h, d, lab, par, None)
        b = render(channel, h, d, lab, par, 1)
        if (a.system, a.user, a.assistant) == (b.system, b.user, b.assistant):
            same += 1
        else:
            diff.append(str(s))
    return {"states": same + len(diff), "identical": same, "different": diff,
            "pass": not diff,
            "note": "rendered text compared (system, user, assistant); the chat template and "
                    "tokenizer are applied identically to both"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="core.yaml")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--status", default="secondary", choices=["primary", "secondary"])
    ap.add_argument("--mc-batches", type=int, default=10)
    ap.add_argument("--table-resamples", type=int, default=200)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    t0 = time.time()
    cfg = build.load(args.config)
    an, pt = cfg["analysis"], build.ptest(cfg)
    B = an["bootstrap"]
    data = load_runs(Path(args.runs))
    g = graph_of(data)
    d, T = g.regular_degree, data.T
    tables = build.load_tables(Path(args.tables), args.tag)
    check_tables_match_runs(data.meta["kernel_fingerprint"],
                            {"actions": tables.actions, "signal": tables.signal})
    cheap = tables.signal is not None
    base = tables.signal if cheap else tables.actions
    if base.source != "p_test":
        raise SystemExit(f"need a p_test-fitted table, got {base.source}")
    arm, cond = Path(args.runs).parent.name, Path(args.runs).name

    # Clock tables, counts and fallbacks.
    pooled = chains_for(tables.signal, None, d) if cheap else chains_for(None, tables.actions, d)
    clock, ck = clock_set(base, d, cheap)
    cells = {}
    for b, inf in enumerate(ck["info"]):
        c = np.array(list(inf["fit_counts"].values()))
        cells[f"L{fu.LENGTH_BLOCKS[b][0]}-{fu.LENGTH_BLOCKS[b][1]}"] = {
            "states": int(len(c)), "min": int(c.min()), "median": float(np.median(c)),
            "max": int(c.max()), "states_below_5": int((c < 5).sum()),
            "total_fit_histories": int(c.sum()), "fallback_states": inf["fallback_states"],
            "per_state": inf["fit_counts"]}
    eps = {}
    for b, t in enumerate(ck["tables"]):
        lab_b = f"L{fu.LENGTH_BLOCKS[b][0]}-{fu.LENGTH_BLOCKS[b][1]}"
        pr = fu.restrict_scoring(base, b)
        if cheap:
            eps[lab_b] = {"signal_aware_joint": {"pooled": weighted_eps(pr, pt), "clock": weighted_eps(t, pt)},
                          "signal_aware_action_marginal": {"pooled": weighted_eps(pr, pt, True),
                                                           "clock": weighted_eps(t, pt, True)},
                          "actions_only_derived_primary": {"pooled": derived_eps(pr, "primary", pt),
                                                           "clock": derived_eps(t, "primary", pt)}}
        else:
            eps[lab_b] = {"actions_only": {"pooled": weighted_eps(pr, pt), "clock": weighted_eps(t, pt)}}

    # Forecast comparison: 10 simulation batches for both, scored on the same runs.
    names = list(pooled)
    batches, pools_all = [], {f"{k}|{m}": [] for k in ("pooled", "clock") for m in names}
    for b in range(args.mc_batches):
        rng = fu.seeded(301, b)
        pp = {m: c.simulate(g, T, an["n_model"], rng) for m, c in pooled.items()}
        cc = {m: c.simulate(g, T, an["n_model"], rng) for m, c in clock.items()}
        sp, sc = scores_from_pools(pp, data), scores_from_pools(cc, data)
        batches.append({m: {k: float((sp[m][k] - sc[m][k]).mean()) for k in SCORE_KEYS} for m in names})
        for m in names:
            pools_all[f"pooled|{m}"].append(pp[m])
            pools_all[f"clock|{m}"].append(cc[m])
    big = {k: np.concatenate(v) for k, v in pools_all.items()}
    del pools_all
    sbig = scores_from_pools(big, data)
    rng = fu.seeded(302)
    comp = {}
    for m in names:
        v = np.array([[bb[m][k] for k in SCORE_KEYS] for bb in batches])
        comp[m] = {"improvement_pooled_minus_clock": {
                       k: {**fu.boot_mean(sbig[f"pooled|{m}"][k] - sbig[f"clock|{m}"][k], B, rng),
                           "mc_batch_sd": float(v[:, i].std(ddof=1)),
                           "mc_sd_of_pooled_estimate": float(v[:, i].std(ddof=1) / np.sqrt(len(v)))}
                       for i, k in enumerate(SCORE_KEYS)},
                   "crps_pooled": {k: fu.boot_mean(sbig[f"pooled|{m}"][k], B, rng) for k in SCORE_KEYS},
                   "crps_clock": {k: fu.boot_mean(sbig[f"clock|{m}"][k], B, rng) for k in SCORE_KEYS},
                   "crps_by_round_pooled": sbig[f"pooled|{m}"]["_round"],
                   "crps_by_round_clock": sbig[f"clock|{m}"]["_round"],
                   "mean_X_pooled": stats.frac_high(big[f"pooled|{m}"]).mean(0),
                   "mean_X_clock": stats.frac_high(big[f"clock|{m}"]).mean(0)}
    del big

    # Table sensitivity: resample each state's fitting histories once; refit pooled and
    # clock tables from the same resample, so their dependence is kept.
    rows = []
    for j in range(args.table_resamples):
        r = fu.seeded(303, j)
        rt = resample_with_lengths(base, r)
        pj = chains_for(rt, None, d) if cheap else chains_for(None, rt, d)
        cj, _ = clock_set(rt, d, cheap)
        pp = {m: c.simulate(g, T, an["n_model"], r) for m, c in pj.items()}
        cc = {m: c.simulate(g, T, an["n_model"], r) for m, c in cj.items()}
        sp, sc = scores_from_pools(pp, data), scores_from_pools(cc, data)
        idx = r.integers(data.R, size=data.R)
        rows.append({m: {k: [float((sp[m][k] - sc[m][k]).mean()), float((sp[m][k] - sc[m][k])[idx].mean())]
                         for k in SCORE_KEYS} for m in names})
    tsens = {}
    for m in names:
        for k in SCORE_KEYS:
            for i, lab in enumerate(("table_and_simulation", "nested_runs_tables_simulation")):
                v = np.array([rw[m][k][i] for rw in rows])
                tsens.setdefault(m, {}).setdefault(k, {})[lab] = {
                    "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                    "lo": float(np.quantile(v, 0.025)), "hi": float(np.quantile(v, 0.975)),
                    "frac_positive": float((v > 0).mean())}

    # Round-2 sensitivity with the enumerated last-round table (separately labelled).
    r2 = {"prompt_identity": prompt_identity(cfg, data.channel, d, base.own_signal)}
    enum_tag = f"{args.tag}_enumerated"
    et = build.load_tables(Path(args.tables), enum_tag)
    if r2["prompt_identity"]["pass"] and (et.signal if cheap else et.actions) is not None:
        check_tables_match_runs(data.meta["kernel_fingerprint"], {"enumerated": et.signal if cheap else et.actions})
        echains = chains_for(et.signal, None, d) if cheap else chains_for(None, et.actions, d)
        res2 = {}
        for m in names:
            # round 2 uses the enumerated table; START and rounds 3.. as in the reference
            # (pooled, or the clock tables for the clock variant).
            out_m = {}
            for vn in ("pooled_with_enumerated_round2", "clock_with_enumerated_round2"):
                if vn.startswith("clock"):
                    cc = Round2Chain(echains[m], clock[m])
                else:
                    cc = Round2Chain(echains[m], fu.ClockChain([pooled[m]] * 3, m))
                diffs = []
                for b in range(args.mc_batches):
                    rng = fu.seeded(304, b)
                    A_ref = (pooled[m] if vn.startswith("pooled") else clock[m]).simulate(g, T, an["n_model"], rng)
                    A_new = cc.simulate(g, T, an["n_model"], rng)
                    s_ref, s_new = scores_from_pools({"x": A_ref}, data)["x"], scores_from_pools({"x": A_new}, data)["x"]
                    diffs.append({k: float((s_ref[k] - s_new[k]).mean()) for k in SCORE_KEYS})
                    if b == 0:
                        rc_ref = fu.round_crps(A_ref, data.actions)[:, 1]
                        rc_new = fu.round_crps(A_new, data.actions)[:, 1]
                out_m[vn] = {"improvement_vs_reference": {k: {"mean": float(np.mean([x[k] for x in diffs])),
                                                              "mc_sd": float(np.std([x[k] for x in diffs], ddof=1))}
                                                          for k in SCORE_KEYS},
                             "round2_crps_reference_batch0": fu.boot_mean(rc_ref, B, rng),
                             "round2_crps_enumerated_batch0": fu.boot_mean(rc_new, B, rng)}
            res2[m] = out_m
        r2["results"] = res2
        r2["label"] = ("round-2 sensitivity: round 2 uses the enumerated last-round table (exact for "
                       "L = 1), later rounds unchanged; distinct from the three-block repair")
    else:
        r2["results"] = None
        r2["label"] = "not done: prompt identity failed or no enumerated table"

    res = {"label": "exploratory follow-up (docs/overnight-core-plan.md C2): clock-indexed audit; "
                    "chosen after the original mismatch was seen, scored on the same runs, so "
                    "not an independent validation",
           "status": args.status,
           "condition": {"arm": arm, "condition": cond, "channel": data.channel, "n": data.n,
                         "T": T, "runs": data.R, "run_fingerprint": data.meta["fingerprint"],
                         "kernel_fingerprint": data.meta["kernel_fingerprint"]},
           "design": {"length_blocks": fu.LENGTH_BLOCKS, "round_to_block": "round t >= 2 uses the block "
                      "containing L = t - 1; round 1 uses START", "fallback": "empty cell -> pooled entry",
                      "derived_actions_only": "per block from the block's signal table, original p_test "
                                              "weights (primary and equal)",
                      "primary_metric": "signal_aware early (2-7) CRPS, pooled - clock (positive favours "
                                        "the clock audit)" if cheap else
                                        "counts control: actions_only early (2-7) CRPS, pooled - clock",
                      "simulation": f"{args.mc_batches} batches x {an['n_model']} runs per predictor, "
                                    "pooled for the estimate; batch SD reported"},
           "cells": cells, "eps_p_test_by_block": eps, "comparison": comp,
           "table_sensitivity": {"resamples": args.table_resamples, "n_model": an["n_model"],
                                 "results": tsens,
                                 "labels": {"table_and_simulation": "same 50 runs; pooled and clock "
                                            "refitted from the same resampled histories",
                                            "nested_runs_tables_simulation": "plus one whole-run "
                                            "resample per table resample; not total uncertainty"}},
           "round2_sensitivity": r2,
           "git_commit": git_commit(), "seconds": time.time() - t0,
           "time": dt.datetime.now(dt.timezone.utc).isoformat()}
    out = Path(args.out) / "clock"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{arm}__{cond}.json"
    path.write_text(json.dumps(to_jsonable(res)) + "\n")
    print(f"wrote {path} ({time.time() - t0:.0f}s)")
    for k, v in cells.items():
        print(f"  cells {k}: min {v['min']}, median {v['median']}, max {v['max']}, "
              f"<5: {v['states_below_5']}, fallbacks {len(v['fallback_states'])}")
    for m in names:
        im = comp[m]["improvement_pooled_minus_clock"]
        print(f"  {m}: " + "; ".join(f"{k} {im[k]['mean']:+.5f} [{im[k]['lo']:+.5f}, {im[k]['hi']:+.5f}] "
                                      f"(MC sd {im[k]['mc_sd_of_pooled_estimate']:.5f})" for k in SCORE_KEYS))
        ts = tsens[m]["2-7"]["nested_runs_tables_simulation"]
        print(f"     early, nested: {ts['mean']:+.5f} [{ts['lo']:+.5f}, {ts['hi']:+.5f}]")


class Round2Chain:
    """Round 2 from `first` (the enumerated table's chain); every other round as `rest`."""

    def __init__(self, first, rest: fu.ClockChain):
        self.first, self.rest = first, rest

    def simulate(self, g, T, R, rng):
        orig = self.rest.chain_at
        self.rest.chain_at = lambda t: self.first if t == 2 else orig(t)
        try:
            return self.rest.simulate(g, T, R, rng)
        finally:
            self.rest.chain_at = orig


def resample_with_lengths(t: AuditTable, rng) -> AuditTable:
    """Resample each state's fitting histories with replacement, keeping each history's
    probability vector and its length together; START fixed."""
    entries = {}
    for k, e in t.entries.items():
        f = np.asarray(e["fit"], float)
        if len(f) <= 1:
            entries[k] = e
            continue
        i = rng.integers(len(f), size=len(f))
        m = f[i].mean(0)
        entries[k] = {**e, "fit": f[i].tolist(), "fit_L": np.asarray(e["fit_L"])[i].tolist(),
                      "kappa": float(m) if t.kind == "actions" else m.tolist()}
    return AuditTable(t.channel, t.d, t.kind, t.source, t.own_signal, entries, t.meta)


if __name__ == "__main__":
    main()
