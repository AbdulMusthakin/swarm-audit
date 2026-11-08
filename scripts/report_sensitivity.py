#!/usr/bin/env python3
"""Diagnostic sensitivity within the four-agent OLMo setting (exploratory, CPU only).

Implements results/extra/followup/sensitivity/plan.md, which was committed before this
script was run. Null: the exact actions-only cached chain. Alternative: the exact
signal-aware cached chain. Each batch is 50 whole 20-round trajectories. Three checks at
nominal 5%:

- S1 shared maximum TV over rounds: S1a with the original cutoff, S1b recalibrated;
- S2 per-round tail probabilities, combined by their minimum and calibrated jointly;
- S3 a fixed persistence event: any trajectory with every agent high in rounds 16-20.

    python scripts/report_sensitivity.py

Output: results/extra/followup/sensitivity/results.json and summary.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from scipy.stats import beta, binom  # noqa: E402

from swarm_audit import build, stats  # noqa: E402
from swarm_audit.prediction import ActionsChain, SignalChain, exact_law  # noqa: E402
from swarm_audit.swarm import load_runs  # noqa: E402
from swarm_audit.topology import cycle  # noqa: E402

OUT = REPO / "results/extra/followup/sensitivity"
TABLES = REPO / "results/extra/audit/olmo-3-7b"
ORIGINAL = REPO / "results/extra/analysis/olmo_calibration_cheap_talk/cheap_talk_n4_cycle_last1.json"
COHORTS = {
    "original": REPO / "results/extra/runs/olmo_calibration_cheap_talk/cheap_talk_n4_cycle_last1",
    "fresh": REPO / "results/extra/followup/runs/followup_obs_unmodified/cheap_talk_n4_cycle_last1",
}
N, T, R, ALPHA = 4, 20, 50, 0.05
SIZES = {"A": 20000, "B": 20000, "null": 20000, "alternative": 20000}
SEED = 20261108
CHUNK = 2000          # batches simulated at once
DECIMALS = 12         # TV values are rounded so that equal counts give equal statistics


def cp(k: int, n: int) -> list[float]:
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def batch_stats(A: np.ndarray, law: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A: (batches, R, T, n) bids. Returns Delta (batches, T) and K (batches,)."""
    X = A.mean(axis=-1)                                        # (B, R, T)
    onehot = np.eye(len(stats.BINS))[stats.bin_index(X)]       # (B, R, T, 5)
    emp = onehot.mean(axis=1)                                  # (B, T, 5)
    delta = np.round(0.5 * np.abs(emp - law[None]).sum(-1), DECIMALS)
    K = A[:, :, 15:, :].all(axis=(2, 3)).sum(axis=1)
    return delta, K


def simulate(chain, g, batches: int, rng, law) -> tuple[np.ndarray, np.ndarray]:
    deltas, Ks = [], []
    for start in range(0, batches, CHUNK):
        b = min(CHUNK, batches - start)
        A = chain.simulate(g, T, b * R, rng).reshape(b, R, T, N)
        d, k = batch_stats(A, law)
        deltas.append(d)
        Ks.append(k)
    return np.concatenate(deltas), np.concatenate(Ks)


class TailTables:
    """p_t(x) = (1 + #{set A batches with Delta_t >= x}) / (1 + N_A), per round."""

    def __init__(self, delta_A: np.ndarray) -> None:
        self.sorted = np.sort(delta_A, axis=0)
        self.n = len(delta_A)

    def min_p(self, delta: np.ndarray) -> np.ndarray:
        p = np.empty_like(delta, dtype=float)
        for t in range(delta.shape[1]):
            ge = self.n - np.searchsorted(self.sorted[:, t], delta[:, t], side="left")
            p[:, t] = (1 + ge) / (1 + self.n)
        return p.min(axis=1), p


def upper_cutoff(values: np.ndarray, alpha: float) -> float:
    """Smallest v among the values with share(values > v) <= alpha."""
    u = np.unique(values)
    share = 1 - np.searchsorted(np.sort(values), u, side="right") / len(values)
    return float(u[np.argmax(share <= alpha)])


def lower_cutoff(values: np.ndarray, alpha: float) -> float:
    """Largest v among the values with share(values <= v) <= alpha (0 if none)."""
    u = np.unique(values)
    share = np.searchsorted(np.sort(values), u, side="right") / len(values)
    ok = u[share <= alpha]
    return float(ok.max()) if len(ok) else 0.0


def rate(reject: np.ndarray) -> dict:
    k, n = int(reject.sum()), int(len(reject))
    return {"rate": k / n, "ci95": cp(k, n), "rejections": k, "batches": n}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = build.load_tables(TABLES, "cheap_talk_d2_enumerated")
    g = cycle(N)
    null = ActionsChain(tables.actions.lookup(), g.regular_degree)
    alt = SignalChain(tables.signal.lookup(), g.regular_degree, tables.signal.own_signal)
    exact0 = exact_law(null, g, T)
    law = stats.binned_law_exact(exact0["X"])
    pi0 = float(exact0["y"][-1])
    pi1 = float(exact_law(alt, g, T)["y"][-1])
    c_orig = float(json.loads(ORIGINAL.read_text())["scores"]["floor"]["actions_only"]["max_over_rounds"])

    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(4)]
    dA, _ = simulate(null, g, SIZES["A"], streams[0], law)
    dB, _ = simulate(null, g, SIZES["B"], streams[1], law)
    d0, K0 = simulate(null, g, SIZES["null"], streams[2], law)
    d1, K1 = simulate(alt, g, SIZES["alternative"], streams[3], law)

    c_M = upper_cutoff(dA.max(axis=1), ALPHA)
    tails = TailTables(dA)
    c_m = lower_cutoff(tails.min_p(dB)[0], ALPHA)
    k_star = next(k for k in range(1, R + 1) if binom.sf(k - 1, R, pi0) <= ALPHA)

    def decide(delta: np.ndarray, K: np.ndarray) -> dict[str, np.ndarray]:
        M = delta.max(axis=1)
        m, _ = tails.min_p(delta)
        return {"S1a_shared_max_TV_original_cutoff": M > c_orig,
                "S1b_shared_max_TV_recalibrated": M > c_M,
                "S2_per_round_tail_joint": m <= c_m,
                "S3_persistence_event": K >= k_star}

    ev0, ev1 = decide(d0, K0), decide(d1, K1)
    tests = {}
    for name in ev0:
        tests[name] = {"null": rate(ev0[name]), "alternative": rate(ev1[name])}
    tests["S3_persistence_event"]["exact_null_level"] = float(binom.sf(k_star - 1, R, pi0))
    tests["S3_persistence_event"]["exact_alternative_power"] = float(binom.sf(k_star - 1, R, pi1))

    live = {}
    for cname, folder in COHORTS.items():
        A = load_runs(folder).actions[None]               # (1, R, T, n)
        delta, K = batch_stats(A, law)
        m, p = tails.min_p(delta)
        dec = decide(delta, K)
        live[cname] = {"max_TV": float(delta.max()), "round_of_max": int(delta.argmax()) + 1,
                       "min_tail_p": float(m[0]), "round_of_min_p": int(p[0].argmin()) + 1,
                       "rounds_flagged_by_S2": [t + 1 for t in range(T) if p[0, t] <= c_m],
                       "persistent_runs": int(K[0]),
                       "flagged": {k: bool(v[0]) for k, v in dec.items()}}

    out = {
        "plan": "results/extra/followup/sensitivity/plan.md",
        "status": "exploratory, model-based; conditional on the cached OLMo kernels",
        "setting": "OLMo-3-7B, n = 4 cycle, cheap talk, last-round history, T = 20",
        "null": "exact actions-only cached chain", "alternative": "exact signal-aware cached chain",
        "batch_runs": R, "sizes": SIZES, "seed": SEED, "alpha": ALPHA,
        "pi0_P_F1_actions_only": pi0, "pi1_P_F1_signal_aware": pi1,
        "cutoffs": {"S1a_original_simultaneous": c_orig, "S1b_recalibrated": c_M,
                    "S2_min_tail_p": c_m, "S3_k_star": k_star},
        "tests": tests, "live_cohorts_illustration": live,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=1) + "\n")

    lines = ["# Diagnostic sensitivity within the four-agent OLMo setting: results", "",
             "Exploratory and model-based (plan: `plan.md`, committed before running).", "",
             f"Cutoffs: S1a {c_orig:.4f} (original); S1b {c_M:.4f}; S2 min tail p <= {c_m:.6f}; "
             f"S3 at least {k_star} persistent trajectory of {R}.", "",
             "| test | false-positive rate (null) | rejection rate (alternative) |", "|---|---|---|"]
    for name, v in tests.items():
        f0, f1 = v["null"], v["alternative"]
        lines.append(f"| {name} | {f0['rate']:.4f} [{f0['ci95'][0]:.4f}, {f0['ci95'][1]:.4f}] | "
                     f"{f1['rate']:.4f} [{f1['ci95'][0]:.4f}, {f1['ci95'][1]:.4f}] |")
    lines += ["", "Live cohorts (illustration only):", ""]
    for cname, v in live.items():
        flags = ", ".join(f"{k}: {'flagged' if f else 'not flagged'}" for k, f in v["flagged"].items())
        lines.append(f"- {cname}: max TV {v['max_TV']:.3f} (round {v['round_of_max']}), min tail p "
                     f"{v['min_tail_p']:.6f} (first reached in round {v['round_of_min_p']}; S2 flags rounds "
                     f"{v['rounds_flagged_by_S2'][0]}-{v['rounds_flagged_by_S2'][-1]}), persistent runs "
                     f"{v['persistent_runs']}; {flags}.")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
