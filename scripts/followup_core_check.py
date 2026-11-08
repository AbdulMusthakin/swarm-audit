#!/usr/bin/env python3
"""Independent re-derivation of the follow-up's headline numbers (plan C3).

Written separately from the pipeline: it imports nothing from swarm_audit. It reads the
raw run files (gzip JSON lines) and the audit-table JSON directly, derives the actions-only
table with its own multinomial weights, simulates its own chains on the cycle, and scores
with a different but equivalent CRPS formula: for an integer-valued forecast X and outcome y,
sum_k (F(k) - 1[y <= k])^2 = E|X - y| - E|X - X'| / 2.

Checks (primary condition, Llama cheap talk n = 10):
1. signed one-step error by round: must equal the pipeline to rounding (deterministic);
2. paired D (endpoint and rounds 2-7): agreement with the pipeline's precision run within
   the combined simulation error (both are Monte Carlo estimates).

    python scripts/followup_core_check.py
"""

from __future__ import annotations

import gzip
import json
from math import factorial
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "results/runs/core_cheap_talk/cheap_talk_n10_cycle_full"
SIG = REPO / "results/audit/llama-3.1-8b/cheap_talk_d2_signal.json"
FU = REPO / "results/followup/core"
SIGNALS = ("high", "low", "none")
N_SIM = 200_000


def load_runs():
    lines = []
    for f in sorted(RUNS.glob("run_*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            lines += [json.loads(x) for x in fh]
    runs = sorted({x["run"] for x in lines})
    R, T, n = len(runs), max(x["round"] for x in lines), max(x["agent"] for x in lines) + 1
    ri = {r: i for i, r in enumerate(runs)}
    A = np.zeros((R, T, n), int)
    P = np.zeros((R, T, n))
    S = np.empty((R, T, n), object)
    S2 = np.empty((R, T, n), object)
    for x in lines:
        i, t, a = ri[x["run"]], x["round"] - 1, x["agent"]
        A[i, t, a] = x["action"] == "H"
        P[i, t, a] = x["p_H"]
        S[i, t, a] = x["state"]
        S2[i, t, a] = x["state_sig"]
    return A, P, S, S2


def tables():
    j = json.loads(SIG.read_text())
    joint = {k: np.asarray(e["kappa"], float).reshape(6) for k, e in j["entries"].items()}
    d = j["d"]
    # Derived actions-only: p_test weights over signal counts (multinomial, p = 1/3 each)
    # and a uniform own signal.
    kao = {"START": joint["START"][:3].sum()}
    for a in "HL":
        for k in range(d + 1):
            num = den = 0.0
            for sh in range(d + 1):
                for sl in range(d + 1 - sh):
                    sn = d - sh - sl
                    w = factorial(d) / (factorial(sh) * factorial(sl) * factorial(sn)) / 3 ** d / 3
                    for m in SIGNALS:
                        num += w * joint[f"{a}-{k}-{sh}-{sl}-{sn}-{m}"][:3].sum()
                        den += w
            kao[f"{a}-{k}"] = num / den
    return joint, kao, d


def simulate_signal(joint, n, T, R, rng):
    """Own implementation: agent i hears i-1 and i+1 (cycle); output index a*3+m, a=0 is H."""
    keys = {}
    out = np.zeros((R, T, n), int)
    start = joint["START"] / joint["START"].sum()
    out[:, 0] = (rng.random((R, n))[..., None] > np.cumsum(start)).sum(-1).clip(max=5)
    # Lookup array indexed by (own a, n_high, sig_high, sig_low, own m).
    L = np.zeros((2, 3, 3, 3, 3, 6))
    for a in range(2):
        for k in range(3):
            for sh in range(3):
                for sl in range(3 - sh):
                    for m in range(3):
                        key = f"{'HL'[a]}-{k}-{sh}-{sl}-{2 - sh - sl}-{SIGNALS[m]}"
                        v = joint[key]
                        L[a, k, sh, sl, m] = v / v.sum()
                        keys[key] = 1
    cum = np.cumsum(L, -1)
    for t in range(1, T):
        prev = out[:, t - 1]
        a, m = prev // 3, prev % 3
        left, right = np.roll(a, 1, 1), np.roll(a, -1, 1)
        lm, rm = np.roll(m, 1, 1), np.roll(m, -1, 1)
        k = (left == 0).astype(int) + (right == 0)
        sh = (lm == 0).astype(int) + (rm == 0)
        sl = (lm == 1).astype(int) + (rm == 1)
        c = cum[a, k, sh, sl, m]
        out[:, t] = (rng.random((R, n))[..., None] > c).sum(-1).clip(max=5)
    return (out // 3 == 0).astype(int)


def simulate_actions(kao, n, T, R, rng):
    A = np.zeros((R, T, n), int)
    A[:, 0] = rng.random((R, n)) < kao["START"]
    K = np.array([[kao[f"{a}-{k}"] for k in range(3)] for a in "HL"])
    for t in range(1, T):
        prev = A[:, t - 1]
        k = np.roll(prev, 1, 1) + np.roll(prev, -1, 1)
        A[:, t] = rng.random((R, n)) < K[1 - prev, k]
    return A


def crps_energy(sample_counts: np.ndarray, y: np.ndarray, K: int) -> np.ndarray:
    """E|X - y| - E|X - X'|/2 for an integer forecast given by its sample, divided by K."""
    p = np.bincount(sample_counts, minlength=K + 1)[: K + 1] / len(sample_counts)
    v = np.arange(K + 1)
    e_xy = np.abs(v[None, :] - y[:, None]) @ p
    e_xx = p @ np.abs(v[:, None] - v[None, :]) @ p
    return (e_xy - e_xx / 2) / K


def scores(A_model, A_swarm):
    n = A_swarm.shape[2]
    y_m = A_model[:, 15:20].sum((1, 2))
    y_s = A_swarm[:, 15:20].sum((1, 2))
    endpoint = crps_energy(y_m, y_s, 5 * n)
    early = np.mean([crps_energy(A_model[:, t].sum(1), A_swarm[:, t].sum(1), n)
                     for t in range(1, 7)], axis=0)
    return endpoint, early


def main():
    A, P, S, S2 = load_runs()
    joint, kao, d = tables()
    R, T, n = A.shape
    kH_sig = {k: v[:3].sum() for k, v in joint.items()}
    e_ao = P - np.vectorize(kao.__getitem__)(S)
    e_sa = P - np.vectorize(kH_sig.__getitem__)(S2)
    pipe = json.loads((FU / "temporal/core_cheap_talk__cheap_talk_n10_cycle_full.json").read_text())
    out = {"label": "independent re-derivation (imports nothing from swarm_audit)", "checks": {}}
    for name, e in (("actions_only", e_ao), ("signal_aware", e_sa)):
        mine = e.mean((0, 2))
        theirs = np.asarray(pipe["one_step"][name]["by_round"]["signed"]["mean"], float)
        diff = float(np.max(np.abs(mine - theirs)))
        out["checks"][f"signed_error_by_round_{name}"] = {"max_abs_diff": diff, "pass": bool(diff < 1e-10),
                                                          "early_2_7_mean": float(e[:, 1:7].mean())}
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261108, 909])))
    D = {"endpoint": [], "early": []}
    for b in range(4):           # 4 independent batches, for an error estimate
        sa = scores(simulate_signal(joint, n, T, N_SIM // 4, rng), A)
        ao = scores(simulate_actions(kao, n, T, N_SIM // 4, rng), A)
        D["endpoint"].append(float((ao[0] - sa[0]).mean()))
        D["early"].append(float((ao[1] - sa[1]).mean()))
    mc = pipe["monte_carlo"]
    ref = mc.get("precision_run", mc["pooled"])
    key = "actions_only - signal_aware"
    for k, pk in (("endpoint", "endpoint"), ("early", "2-7")):
        v = np.array(D[k])
        mine, se_mine = float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
        theirs = ref["paired"][key][pk]["mean"]
        batch_sd = mc["summary"][key][pk]["sd"]
        se_theirs = batch_sd * np.sqrt(mc["n_model"] / ref.get("n_model", mc["n_model"] * 10))
        z = (mine - theirs) / np.sqrt(se_mine ** 2 + se_theirs ** 2)
        out["checks"][f"paired_D_{k}"] = {"independent": mine, "independent_mc_se": se_mine,
                                          "pipeline": theirs, "pipeline_mc_se": float(se_theirs),
                                          "z": float(z), "pass": bool(abs(z) < 3),
                                          "independent_sims": N_SIM}
    out["pass"] = bool(all(c["pass"] for c in out["checks"].values()))
    (FU / "independent_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
