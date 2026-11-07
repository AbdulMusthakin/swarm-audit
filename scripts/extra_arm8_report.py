#!/usr/bin/env python3
"""Arm 8 (extra): the numbers quoted in results/extra/arm8_summary.md and the defects figure.

Reads only logged runs, the fitted degree-4 audit table and the analysis JSON written by
scripts/analyse.py; it makes no model queries.

    python scripts/extra_arm8_report.py

Outputs: results/extra/arm8_numbers.json, results/extra/figures/arm8_defects.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from swarm_audit import build, stats  # noqa: E402
from swarm_audit.plots import COLOURS, GRID, INK, INK2, SURFACE, _style  # noqa: E402
from swarm_audit.swarm import load_runs  # noqa: E402

CFG = build.load("extra-scale.yaml")
ROOT = build.results_root(CFG)
COND = "counts_n100_random_regular_d4_seed20261108_full"
RUNS = ROOT / "runs" / "extra_rr4" / COND
TABLES = ROOT / "audit" / "llama-3.1-8b"
ANALYSIS = ROOT / "analysis" / "extra_rr4" / f"{COND}.json"
LATE = slice(15, 20)          # rounds 16-20


def band(x_rt: np.ndarray, B: int = 2000, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """95% whole-run bootstrap interval of the run mean, per column."""
    rng = np.random.default_rng(seed)
    R = x_rt.shape[0]
    bs = np.stack([x_rt[rng.integers(0, R, R)].mean(0) for _ in range(B)])
    return np.quantile(bs, 0.025, 0), np.quantile(bs, 0.975, 0)


def paired(a: list, b: list, B: int = 2000, seed: int = 12) -> dict:
    """Mean of the per-run differences a - b with a 95% whole-run bootstrap interval
    (descriptive; not the core's primary contrast)."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return {"mean": float(d.mean()), "lo": float(np.quantile(bs, 0.025)),
            "hi": float(np.quantile(bs, 0.975)), "runs_a_better": int((d < 0).sum()),
            "n": len(d), "B": B}


def main() -> None:
    res = json.loads(ANALYSIS.read_text())
    data = load_runs(RUNS)
    tables = build.load_tables(TABLES, "counts_d4")
    kap = tables.actions.lookup()
    eps = stats.swarm_defects(data.p_H, data.state, kap)            # (R, T, n)
    R, T, n = eps.shape
    eps_rt = eps.mean(axis=2)                                        # (R, T)
    lo, hi = band(eps_rt)
    states = sorted(kap, key=lambda k: (k != "START", k))
    eps_test = res["defects"]["actions_only"]["eps_test_state"]
    per_state = {}
    for s in states:
        m = data.state == s
        per_state[s] = {"decisions": int(m.sum()),
                        "eps_swarm": float(eps[m].mean()) if m.any() else None,
                        "eps_test": eps_test.get(s),
                        "kappa_H": float(kap[s]),
                        "swarm_mean_p_H": float(data.p_H[m].mean()) if m.any() else None}

    sc = res["scores"]
    X = np.asarray(res["X"]["swarm"]["mean"])
    late_y = data.actions[:, LATE, :].mean(axis=(1, 2))              # y per run
    delta = {k: {"max": float(np.max(v)), "argmax_round": int(np.argmax(v)) + 1,
                 "floor_simultaneous": sc["floor"][k]["max_over_rounds"],
                 "rounds_above_pointwise": [t + 1 for t, (d, f) in
                                            enumerate(zip(v, sc["floor"][k]["per_round"])) if d > f]}
             for k, v in sc["delta"].items()}
    sn = {cls: {"swarm_mean_rounds_2_20": float(np.mean(v["swarm"]["mean"][1:])),
                "order0_mean_rounds_2_20": float(np.mean(v["order0"][1:])),
                "actions_only_mean_rounds_2_20": float(np.mean(v["actions_only"][1:]))}
          for cls, v in res["shared_neighbour"].items()}
    g = json.loads((ROOT / "graph_rr4_n100.json").read_text())
    out = {
        "condition": res["condition"],
        "runs_logged": int(R), "decisions": int(R * T * n),
        "graph": {"diameter": g.get("diameter"), "edges": len(g.get("edges", []))},
        "swarm_X": {"round1": float(X[0]), "rounds_2_20_mean": float(X[1:].mean()),
                    "rounds_16_20_mean": float(X[LATE].mean()),
                    "min": float(X.min()), "max": float(X.max())},
        "late_y_per_run": {"mean": float(late_y.mean()), "sd": float(late_y.std(ddof=1)),
                           "min": float(late_y.min()), "max": float(late_y.max())},
        "crps": sc["crps"], "F_error": sc["F_error"],
        "crps_paired_order0_minus_actions_only": paired(sc["crps_per_run"]["order0"],
                                                        sc["crps_per_run"]["actions_only"]),
        "held_out": {
            k: res["held_out"][k] for k in ("runs", "crps", "F_error")},
        "delta": delta,
        "defects": {"eps_test_pooled": res["defects"]["actions_only"]["eps_test_pooled"],
                    "eps_swarm": res["defects"]["actions_only"]["eps_swarm"],
                    "eps_swarm_by_round": eps_rt.mean(0).tolist(),
                    "eps_swarm_by_round_lo": lo.tolist(), "eps_swarm_by_round_hi": hi.tolist(),
                    "per_state": per_state},
        "bound": {"B_T": res["defects"]["actions_only"]["bound"]["B"][-1],
                  "B_2": res["defects"]["actions_only"]["bound"]["B"][1],
                  "first_saturation_round":
                      res["defects"]["actions_only"]["bound"]["first_saturation_round"]},
        "defect_shift": res["defect_shift"]["actions_only"],
        "edge_disagreement": {k: float(np.mean(v[1:])) if not isinstance(v, dict) else None
                              for k, v in res["edge_disagreement"].items()},
        "consensus": {k: v["prob"] for k, v in res["consensus"].items()},
        "shared_neighbour": sn,
    }
    path = ROOT / "arm8_numbers.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}")

    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [3, 2]},
                                 facecolor=SURFACE)
    for ax in (a1, a2):
        _style(ax)
    t = np.arange(1, T + 1)
    c = COLOURS["actions_only"]
    a1.fill_between(t, lo, hi, color=INK, alpha=0.12, linewidth=0)
    a1.plot(t, eps_rt.mean(0), color=INK, linewidth=2, marker="o", markersize=4,
            label="ε(p_swarm) per round, mean over agents and runs")
    a1.axhline(out["defects"]["eps_test_pooled"], color=c, linewidth=2,
               label=f"ε(p_test) pooled = {out['defects']['eps_test_pooled']:.3f}")
    a1.set_xlabel("round t", color=INK2, fontsize=9)
    a1.set_ylabel("local defect |p_H − κ(H|s)|", color=INK2, fontsize=9)
    a1.set_ylim(0, max(0.12, float(hi.max()) * 1.15))
    a1.legend(frameon=False, fontsize=8, loc="upper right")
    import matplotlib.ticker
    a1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ks = [s for s in states if s != "START"]
    y = np.arange(len(ks))
    a2.barh(y + 0.2, [per_state[s]["eps_test"] for s in ks], height=0.38, color=c,
            label="ε(p_test), audit scoring set")
    a2.barh(y - 0.2, [per_state[s]["eps_swarm"] or 0 for s in ks], height=0.38, color=INK,
            label="ε(p_swarm), swarm decisions")
    a2.set_yticks(y, [f"{s} ({per_state[s]['decisions']:,})" for s in ks], fontsize=7)
    a2.invert_yaxis()
    a2.set_xlabel("local defect by state (swarm decisions in brackets)", color=INK2, fontsize=9)
    a2.grid(True, axis="x", color=GRID, linewidth=0.6)
    a2.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.suptitle("Arm 8 (extra): local defects of the degree-4 actions-only audit, "
                 "100 agents, one random 4-regular graph", fontsize=10, color=INK)
    fig.tight_layout()
    fpath = ROOT / "figures" / "arm8_defects.png"
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fpath, dpi=160)
    plt.close(fig)
    print(f"wrote {fpath}")


if __name__ == "__main__":
    main()
