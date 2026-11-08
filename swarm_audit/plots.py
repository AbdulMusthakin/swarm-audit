"""Figures for the report (static PNGs under results/figures/).

Colours follow the entity, never its rank: the swarm is drawn in ink, and each predictor
keeps its slot of the reference categorical palette in every figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLOURS = {
    "swarm": INK,
    "actions_only": "#2a78d6",          # slot 1, blue
    "signal_aware": "#eb6834",          # slot 2, orange
    "order0": "#1baf7a",                # slot 3, aqua
    "fitted_from_swarm": "#eda100",     # slot 4, yellow
    "actions_only_equal_w": "#e87ba4",  # slot 5, magenta
}
LABELS = {
    "swarm": "swarm", "actions_only": "actions-only audit", "signal_aware": "signal-aware audit",
    "order0": "order-0 audit", "fitted_from_swarm": "fitted from swarm",
    "actions_only_equal_w": "actions-only (equal weights)",
}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figure1(res: dict, path: Path, title: str = "", contraction: dict | None = None) -> None:
    """X_t for the swarm (with its run-level band) against each predictor; the bound
    B_t in a separate panel, because its scale hides the measured gap.

    `contraction` (optional): one condition's result from scripts/analyse_dobrushin.py
    (`analyse`). It adds, for the actions-only audit, the Dobrushin TV contraction bounds
    next to Theorem 1 (right panel), and the local Dobrushin bound on the mean as a band
    around the audit's expected X_t (left panel). Without it the figure is unchanged."""
    T = len(res["X"]["swarm"]["mean"])
    t = np.arange(1, T + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.9 if contraction else 3.6),
                                 gridspec_kw={"width_ratios": [3, 2]}, facecolor=SURFACE)
    for ax in (a1, a2):
        _style(ax)
    sw = res["X"]["swarm"]
    a1.fill_between(t, sw["lo"], sw["hi"], color=INK, alpha=0.12, linewidth=0)
    a1.plot(t, sw["mean"], color=INK, linewidth=2, marker="o", markersize=4, label=LABELS["swarm"])
    for name, v in res["X"].items():
        if name != "swarm":
            a1.plot(t, v["mean"], color=COLOURS.get(name, INK2), linewidth=2,
                    label=LABELS.get(name, name))
    blue = COLOURS["actions_only"]
    if contraction:   # |E_swarm X_t - E_audit X_t| <= M_t (local Dobrushin bound on the mean)
        mu = np.asarray(contraction["model_mean"])
        m = np.asarray(contraction["mean_local_bound"])
        lo, hi = np.clip(mu - m, 0, 1), np.clip(mu + m, 0, 1)
        a1.fill_between(t, lo, hi, color=blue, alpha=0.08, linewidth=0)
        for edge in (lo, hi):
            a1.plot(t, edge, color=blue, linewidth=1, linestyle=":")
        a1.plot([], [], color=blue, linewidth=1, linestyle=":",
                label=f"actions-only mean ± local Dobrushin bound (α = {contraction['alpha']:.2f})")
    a1.set_xlabel("round t", color=INK2, fontsize=9)
    a1.set_ylabel("mean fraction bidding high, X_t", color=INK2, fontsize=9)
    a1.set_ylim(0, 1)
    if contraction:
        a1.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
                   handlelength=3.5)
    else:
        a1.legend(frameon=False, fontsize=8, loc="best")
    for name, d in res["defects"].items():
        b = d["bound"]
        c = COLOURS.get(name, INK2)
        a2.fill_between(t, np.minimum(b["lo"], 1), np.minimum(b["hi"], 1), color=c, alpha=0.15,
                        linewidth=0)
        a2.plot(t, b["B_capped"], color=c, linewidth=2,
                label=f"bound, {LABELS.get(name, name)} (uncapped B_T = {b['B'][-1]:.2f})")
    if contraction:   # Dobrushin TV contraction for the actions-only audit (current round)
        n = contraction["condition"]["n"]
        rel = "=" if contraction["delta_kind"].startswith("exact") else "≤"
        a2.plot(t, np.minimum(contraction["tv_contraction_same_local_defects"], 1), color=blue,
                linewidth=1.6, linestyle=":", label="contraction, same local defects")
        lo, hi = np.minimum(contraction["tv_contraction_joint_defect_ci"], 1)
        a2.fill_between(t, lo, hi, color=blue, alpha=0.12, linewidth=0)
        joint = contraction["tv_contraction_joint_defect"]
        a2.plot(t, np.minimum(joint, 1), color=blue, linewidth=2, linestyle="-.",
                label=f"contraction + joint defect (δ {rel} {contraction['delta']:.3f}, "
                      f"uncapped {joint[-1]:.2f})")
    delta = res["scores"]["delta"].get("actions_only")
    if delta is not None:
        a2.plot(t, delta, color=INK, linewidth=2, linestyle="--", label="measured Δ_t (actions-only)")
        fl = res["scores"]["floor"].get("actions_only")
        if fl is not None:
            a2.plot(t, fl["per_round"], color=INK2, linewidth=1, linestyle=":",
                    label="null 95% (pointwise)")
            a2.axhline(fl["max_over_rounds"], color=INK2, linewidth=1, linestyle="-.",
                       label="null 95% of max over rounds (simultaneous)")
    a2.set_xlabel("round t", color=INK2, fontsize=9)
    a2.set_ylabel("TV distance (bound: estimate, capped at 1)", color=INK2, fontsize=9)
    a2.set_ylim(0, 1.05)
    if contraction:
        a2.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.15),
                   handlelength=3.5)
    else:
        a2.legend(frameon=False, fontsize=7, loc="center right")
    for ax in (a1, a2):
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    if title:
        fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure2(rows: list[dict], path: Path) -> None:
    """Left: paired CRPS differences D (actions-only minus signal-aware) with 95% run-level
    intervals, per cheap-talk condition. Middle: eps(p_test) against eps(p_swarm) for the
    actions-only table. Right: actions-only against signal-aware defects on the same
    swarm histories. `rows`: one dict per condition with keys name, channel, n and the
    analysis result `res`."""
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), facecolor=SURFACE)
    for ax in axs:
        _style(ax)
    a1, a2, a3 = axs
    paired = [r for r in rows if r["res"]["scores"].get("paired_D")]
    for i, r in enumerate(paired):
        D = r["res"]["scores"]["paired_D"]
        a1.errorbar([D["mean"]], [i], xerr=[[D["mean"] - D["lo"]], [D["hi"] - D["mean"]]],
                    fmt="o", color=COLOURS["signal_aware"], markersize=6, capsize=3, linewidth=2)
    a1.axvline(0, color=INK2, linewidth=0.8)
    a1.set_yticks(range(len(paired)), [r["name"] for r in paired], fontsize=7)
    a1.set_xlabel("paired CRPS difference D (> 0 favours signals)", color=INK2, fontsize=8)
    for r in rows:
        d = r["res"]["defects"]["actions_only"]
        a2.scatter([d["eps_test_pooled"]], [d["eps_swarm"]], s=40,
                   color=COLOURS["actions_only" if r["channel"] == "counts" else "signal_aware"],
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        a2.annotate(r["name"], (d["eps_test_pooled"], d["eps_swarm"]), fontsize=6, color=INK2,
                    xytext=(4, 2), textcoords="offset points")
    lim = max([0.01] + [max(r["res"]["defects"]["actions_only"]["eps_test_pooled"],
                            r["res"]["defects"]["actions_only"]["eps_swarm"]) for r in rows])
    a2.plot([0, lim * 1.1], [0, lim * 1.1], color=INK2, linewidth=0.8, linestyle=":")
    a2.set_xlabel("ε(p_test), actions-only", color=INK2, fontsize=8)
    a2.set_ylabel("ε(p_swarm), actions-only", color=INK2, fontsize=8)
    for r in rows:
        os_ = r["res"].get("one_step")
        if os_:
            a3.scatter([os_["mean_abs_pH_actions_only"]], [os_["mean_abs_pH_signal_aware"]], s=40,
                       color=COLOURS["signal_aware"], edgecolor=SURFACE, linewidth=2, zorder=3)
            a3.annotate(r["name"], (os_["mean_abs_pH_actions_only"],
                                    os_["mean_abs_pH_signal_aware"]),
                        fontsize=6, color=INK2, xytext=(4, 2), textcoords="offset points")
    a3.plot([0, 0.5], [0, 0.5], color=INK2, linewidth=0.8, linestyle=":")
    a3.set_xlabel("one-step |p_H − κ(H|s)|, actions-only", color=INK2, fontsize=8)
    a3.set_ylabel("one-step |p_H − κ′(H|s′)|, signal-aware", color=INK2, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
