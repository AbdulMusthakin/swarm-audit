#!/usr/bin/env python3
"""E3 forecasts before any spend (docs/overnight-extra-plan.md; results/extra/followup/plan.json).

- n = 10, last-1: 100,000 simulated runs of the signal-aware (`both`) chain and of the
  actions-only chain from the OLMo enumerated last-1 tables (cached kernel; no new LLM call).
- n = 4, full history: the exact law on 1,296 profiles from the 9b sampled full-history
  tables (an approximation of a full-history swarm by a last-round state).

    python scripts/extra_followup_forecast.py

Output: results/extra/followup/e3_forecast.json. Monte Carlo SEs are binomial / sample SEs.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from swarm_audit import build, stats  # noqa: E402
from swarm_audit.prediction import ActionsChain, SignalChain, exact_law  # noqa: E402
from swarm_audit.topology import make  # noqa: E402

CFG = build.load("extra-followup-olmo.yaml")
ROOT = build.results_root(CFG)
TABLES = ROOT / "audit" / "olmo-3-7b"
OUT = ROOT / "followup"
SIMS, SEED, T = 100_000, 20261108, 20


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summarise_sims(A: np.ndarray) -> dict:
    n = A.shape[2]
    y = stats.high_count(A)
    K = 5 * n
    p1 = float((y == K).mean())
    p0 = float((y == 0).mean())
    F = y / K
    return {"runs": int(len(A)), "P_F1": p1, "P_F1_mc_se": float(np.sqrt(p1 * (1 - p1) / len(A))),
            "P_F0": p0, "P_F0_mc_se": float(np.sqrt(p0 * (1 - p0) / len(A))),
            "mean_F": float(F.mean()), "mean_F_mc_se": float(F.std(ddof=1) / np.sqrt(len(A))),
            "EX_t": stats.frac_high(A).mean(0).tolist(), "evidence": "Monte Carlo (cached kernel)"}


def summarise_law(ex: dict, n: int) -> dict:
    y = np.asarray(ex["y"])
    K = 5 * n
    return {"P_F1": float(y[K]), "P_F0": float(y[0]),
            "mean_F": float((np.arange(K + 1) * y).sum() / K),
            "EX_t": (np.asarray(ex["X"]) @ np.arange(n + 1) / n).tolist(),
            "y_law": y.tolist(), "evidence": "exact (fitted table)"}


def main() -> None:
    rng = np.random.default_rng(SEED)
    enum = build.load_tables(TABLES, "cheap_talk_d2_enumerated")
    full = build.load_tables(TABLES, "cheap_talk_d2")
    g10, g4 = make("cycle", 10), make("cycle", 4)
    out = {"plan": "results/extra/followup/plan.json E3.cpu_forecast_before_spend",
           "seed": SEED, "sims": SIMS,
           "tables": {p.name: sha(p) for p in sorted(TABLES.glob("cheap_talk_d2*.json"))}}
    sig = SignalChain(enum.signal.lookup(), 2, enum.signal.own_signal)
    act = ActionsChain(enum.actions.lookup(), 2)
    out["n10_last1"] = {"both_signal_aware": summarise_sims(sig.simulate(g10, T, SIMS, rng)),
                        "actions_only": summarise_sims(act.simulate(g10, T, SIMS, rng))}
    out["n4_full"] = {
        "signal_aware_9b_table": summarise_law(exact_law(
            SignalChain(full.signal.lookup(), 2, full.signal.own_signal), g4, T), 4),
        "actions_only_9b_table": summarise_law(exact_law(ActionsChain(full.actions.lookup(), 2), g4, T), 4),
        "note": "the full-history swarm is approximated by the last-round state of the 9b sampled tables"}
    out["n4_last1_reference_exact"] = summarise_law(exact_law(sig, g4, T), 4)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e3_forecast.json").write_text(json.dumps(out, indent=1) + "\n")
    for k in ("n10_last1", "n4_full"):
        for m, v in out[k].items():
            if isinstance(v, dict):
                print(k, m, {x: round(v[x], 5) for x in ("P_F1", "P_F0", "mean_F")})


if __name__ == "__main__":
    main()
