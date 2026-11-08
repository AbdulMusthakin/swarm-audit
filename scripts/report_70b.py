#!/usr/bin/env python3
"""The report's numbers for the exploratory Llama 3.1 70B rerun, next to the 8B's.

CPU only; reads committed tables, analyses, gate records and contraction results:

- message responsiveness on the cached last-round tables (the report's screen: two
  neighbours, one bidding high, four neighbour-message variants, own message "none",
  spread of P(high) averaged over the agent's own last bid);
- the prespecified contrast D, re-derived from per-run CRPS and checked against the
  committed analysis;
- paired CRPS differences of each audit against the order-zero audit, whole-run bootstrap;
- live and forecast late (rounds 16-20) high-bid shares and F_error (live minus forecast);
- the roundwise check (largest TV, its round and the simultaneous cutoff);
- the four-agent last-round calibration (G5) and the contraction coefficients.

    python scripts/report_70b.py

Output: results/followup/report_70b.json. It also replaces its own rows (ids R70_* and,
from results/extra/followup/persistence.json, P4_*) in results/followup/core/claims.csv;
run scripts/report_persistence.py first.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

OUT = REPO / "results/followup/report_70b.json"
M70 = REPO / "interest/model-70b"
TABLES = {"8b": REPO / "results/audit/llama-3.1-8b/cheap_talk_d2_enumerated_signal.json",
          "70b": M70 / "audit/llama-3.1-70b/cheap_talk_d2_enumerated_signal.json"}
ANALYSES = {
    ("8b", 10): REPO / "results/analysis/core_cheap_talk/cheap_talk_n10_cycle_full.json",
    ("8b", 20): REPO / "results/analysis/core_cheap_talk/cheap_talk_n20_cycle_full.json",
    ("70b", 10): M70 / "analysis/m70_core_cheap_talk/cheap_talk_n10_cycle_full.json",
    ("70b", 20): M70 / "analysis/m70_core_cheap_talk_n20/cheap_talk_n20_cycle_full.json",
}
G5 = {"70b": M70 / "gates/llama-3.1-70b", "8b": REPO / "results/gates"}
DOBRUSHIN = {"8b": REPO / "results/followup/dobrushin/dobrushin_results.json",
             "70b": M70 / "dobrushin/dobrushin_results.json"}
AUDITS = ("order0", "actions_only", "signal_aware")
CLAIMS = REPO / "results/followup/core/claims.csv"
PERSISTENCE = REPO / "results/extra/followup/persistence.json"
SOURCE_COMMIT = "c3ed9cb"   # the report commit that produced these files
B, SEED = 2000, 20261108


def responsiveness(path: Path) -> dict:
    entries = json.loads(path.read_text())["entries"]
    spreads = {}
    for own in ("H", "L"):
        ph = [sum(entries[f"{own}-1-{v}-none"]["kappa"][0]) for v in ("2-0-0", "0-2-0", "0-0-2", "1-1-0")]
        spreads[own] = max(ph) - min(ph)
    return {"by_own_last_bid": spreads, "mean": float(np.mean(list(spreads.values())))}


def boot(v: np.ndarray, rng) -> dict:
    idx = rng.integers(len(v), size=(B, len(v)))
    m = v[idx].mean(axis=1)
    return {"mean": float(v.mean()), "lo": float(np.quantile(m, 0.025)),
            "hi": float(np.quantile(m, 0.975))}


def condition(path: Path, rng) -> dict:
    a = json.loads(path.read_text())
    s = a["scores"]
    c = {k: np.asarray(s["crps_per_run"][k]) for k in AUDITS}
    D = boot(c["actions_only"] - c["signal_aware"], rng)
    X = a["X"]
    late = {k: float(np.mean(X[k]["mean"][15:20])) for k in ("swarm",) + AUDITS}
    rw = {}
    for k in ("actions_only", "signal_aware"):
        d = np.asarray(s["delta"][k])
        rw[k] = {"max_TV": float(d.max()), "round": int(d.argmax()) + 1,
                 "cutoff": float(s["floor"][k]["max_over_rounds"]),
                 "flagged": bool(d.max() > s["floor"][k]["max_over_rounds"])}
    return {
        "source": str(path.relative_to(REPO)),
        "crps": {k: float(c[k].mean()) for k in AUDITS},
        "D_rederived": D,
        "D_committed": {k: s["paired_D"][k] for k in ("mean", "lo", "hi")},
        "D_matches_committed_mean": bool(abs(D["mean"] - s["paired_D"]["mean"]) < 1e-12),
        "minus_order0": {k: boot(c[k] - c["order0"], rng) for k in ("actions_only", "signal_aware")},
        "F_error_live_minus_forecast": {k: {kk: s["F_error"][k][kk] for kk in ("mean", "lo", "hi")}
                                        for k in AUDITS},
        "late_high_share_rounds_16_20": late,
        "high_share_by_round": {k: [float(x) for x in X[k]["mean"]] for k in ("swarm",) + AUDITS},
        "roundwise": rw,
    }


def g5(folder: Path) -> dict:
    out = {}
    for ch in ("cheap_talk", "counts"):
        g = json.loads((folder / f"G5_{ch}.json").read_text())
        out[ch] = {"pass": g["pass"], "per_run": g.get("result", {}).get("per_run", g.get("per_run")),
                   "per_decision_pass": (g.get("result", {}).get("per_decision") or g.get("per_decision") or {}).get("pass")}
    return out


def contraction(path: Path) -> dict:
    res = json.loads(path.read_text())["results"]
    full = [r for r in res if r["condition"]["last_k"] is None]
    return {"delta_min": float(min(r["delta"] for r in full)),
            "delta_max": float(max(r["delta"] for r in full)),
            "conditions": {f"{r['condition']['channel']}_n{r['condition']['n']}_{r['condition']['graph']}":
                           {"delta": r["delta"], "delta_kind": r["delta_kind"],
                            "tv_joint_round20_uncapped": r["tv_contraction_joint_defect"][-1],
                            "first_round_joint_bound_ge_1": next((i + 1 for i, v in enumerate(r["tv_contraction_joint_defect"]) if v >= 1), None)}
                           for r in full}}


def update_claims(out: dict) -> None:
    """Replace this script's rows in the claim map (idempotent)."""
    with CLAIMS.open(newline="") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())
    rows = [r for r in rows if not r["id"].startswith(("R70_", "P4_"))]
    src = str(OUT.relative_to(REPO))
    status = "exploratory, post hoc (70B rerun)"
    boot_note = "95% whole-run bootstrap, 2,000 resamples, conditional on the tables"
    new = []

    def add(i, claim, v, key, status=status, src=src, run_set="", lo="", hi="", kind=""):
        new.append({"id": i, "claim": claim, "value": v, "lo": lo, "hi": hi,
                    "interval_type": kind or ("point value" if lo == "" else boot_note),
                    "status": status, "run_set": run_set, "source_file": src,
                    "source_key": key, "source_commit": SOURCE_COMMIT})

    for m in ("8b", "70b"):
        add(f"R70_resp_{m}", f"Message responsiveness, Llama 3.1 {m.upper()} (mean spread of P(high), "
            "cached last-round table)", out["responsiveness"][m]["mean"], f"responsiveness/{m}/mean",
            run_set="cached enumerated d = 2 signal table")
    for n in (10, 20):
        c = out["conditions"][f"70b_n{n}"]
        rs = f"m70_core_cheap_talk, cheap_talk_n{n}_cycle_full, 50 runs (Llama 3.1 70B, live)"
        D = c["D_committed"]
        add(f"R70_D_n{n}", f"70B, n = {n}: D = CRPS(actions-only) - CRPS(signal-aware)", D["mean"],
            f"conditions/70b_n{n}/D_committed", run_set=rs, lo=D["lo"], hi=D["hi"])
        for a in ("signal_aware", "actions_only"):
            o = c["minus_order0"][a]
            add(f"R70_{a}_minus_o0_n{n}", f"70B, n = {n}: CRPS({a}) - CRPS(order-0)", o["mean"],
                f"conditions/70b_n{n}/minus_order0/{a}", run_set=rs, lo=o["lo"], hi=o["hi"])
        for a in ("actions_only", "signal_aware"):
            r = c["roundwise"][a]
            add(f"R70_rw_{a}_n{n}", f"70B, n = {n}: largest roundwise TV, {a} (round {r['round']}, "
                f"cutoff {r['cutoff']:.3f})", r["max_TV"], f"conditions/70b_n{n}/roundwise/{a}",
                run_set=rs)
        add(f"R70_late_live_n{n}", f"70B, n = {n}: live high-bid share, rounds 16-20",
            c["late_high_share_rounds_16_20"]["swarm"],
            f"conditions/70b_n{n}/late_high_share_rounds_16_20/swarm", run_set=rs)
    if PERSISTENCE.exists():
        p = json.loads(PERSISTENCE.read_text())
        ps = str(PERSISTENCE.relative_to(REPO))
        rs = "cached OLMo-3-7B tables, n = 4 cycle, last round (exact)"
        for a in ("actions_only", "signal_aware"):
            add(f"P4_survival_{a}", f"OLMo n = 4: P(stay all-high in rounds 2-20 | all-high in round 1), {a}",
                p["survival_rounds_2_20"][a], f"survival_rounds_2_20/{a}", status="exploratory",
                src=ps, run_set=rs, kind="exact")
        add("P4_survival_ratio", "OLMo n = 4: survival ratio, signal-aware / actions-only",
            p["survival_ratio_signal_over_actions"], "survival_ratio_signal_over_actions",
            status="exploratory", src=ps, run_set=rs, kind="exact")
    with CLAIMS.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows + new)


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {"status": "exploratory, post hoc: the 70B runs were decided after the 8B results",
           "bootstrap": f"{B} whole-run resamples, seed {SEED}; conditional on the fitted tables",
           "responsiveness": {m: responsiveness(p) for m, p in TABLES.items()},
           "conditions": {f"{m}_n{n}": condition(p, rng) for (m, n), p in ANALYSES.items()},
           "g5": {m: g5(f) for m, f in G5.items()},
           "contraction": {m: contraction(p) for m, p in DOBRUSHIN.items()}}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    update_claims(out)
    r = out["responsiveness"]
    print(f"responsiveness: 8B {r['8b']['mean']:.4f}, 70B {r['70b']['mean']:.4f}")
    for k, c in out["conditions"].items():
        D, o = c["D_rederived"], c["minus_order0"]
        print(f"{k}: D {D['mean']:+.4f} [{D['lo']:+.4f}, {D['hi']:+.4f}] (committed "
              f"{c['D_committed']['mean']:+.4f}); CRPS {', '.join(f'{a} {v:.3f}' for a, v in c['crps'].items())}; "
              f"SA-o0 {o['signal_aware']['mean']:+.4f} [{o['signal_aware']['lo']:+.4f}, {o['signal_aware']['hi']:+.4f}]; "
              f"AO-o0 {o['actions_only']['mean']:+.4f} [{o['actions_only']['lo']:+.4f}, {o['actions_only']['hi']:+.4f}]; "
              f"late {', '.join(f'{a} {v:.3f}' for a, v in c['late_high_share_rounds_16_20'].items())}; "
              f"rw {', '.join(f'{a} {v['max_TV']:.3f}@{v['round']} vs {v['cutoff']:.3f}' for a, v in c['roundwise'].items())}")
    print("g5", json.dumps(out["g5"])[:400])
    print("contraction", {m: (round(c["delta_min"], 3), round(c["delta_max"], 3)) for m, c in out["contraction"].items()})
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
