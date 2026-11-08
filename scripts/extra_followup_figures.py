#!/usr/bin/env python3
"""Figures for the overnight follow-up (results/extra/followup/figures/).

Figure 2 candidate (vector PDF and PNG, 7.0 x 2.5 in, final print size):
  A  P(F = 1) under the four audit information variants (exact), with the live 8/50.
  B  P(F = 1) under the four observation interventions (exact, cached kernel).
  C  finite-horizon persistence in the all-high set from a common entry distribution.
Appendix: occupancy and own-message composition; the K(H | s, c, o) grid; the Llama
control; size by memory (if size_memory.json exists).

    python scripts/extra_followup_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import beta  # noqa: E402

from swarm_audit.plots import GRID, INK, INK2, SURFACE, _style  # noqa: E402

F = REPO / "results" / "extra" / "followup"
OUT = F / "figures"
VCOL = {"both": "#eb6834", "own_only": "#1baf7a", "neighbours_only": "#e87ba4", "neither": "#2a78d6"}
VLAB = {"both": "full table", "own_only": "own msg only", "neighbours_only": "nbr msgs only",
        "neither": "actions only"}
PLAB = {"unmodified": "as logged", "own_null": "own → none", "neighbour_null": "nbrs → none",
        "both_null": "both → none"}
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


def fmt(p):
    return f"{p:.3f}" if p >= 0.001 else (f"{p:.1e}" if p > 0 else "0")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def hbars(ax, labels, vals, colours, xmax=0.25):
    ax.tick_params(labelsize=6.5)
    y = np.arange(len(labels))
    ax.barh(y, vals, height=0.62, color=colours, edgecolor=SURFACE, linewidth=1)
    for yi, v in zip(y, vals):
        ax.text(v + xmax * 0.015, yi, fmt(v), va="center", ha="left", color=INK, fontsize=6.5, zorder=5,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.6))
    ax.set_yticks(y, labels, color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def figure2(abl, obs):
    m = abl["models"]["olmo-3-7b"]
    e2 = obs["models"]["olmo-3-7b"]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), facecolor=SURFACE,
                            gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]})
    for ax in axs:
        _style(ax)
    a, b, c = axs
    vs = ["both", "own_only", "neighbours_only", "neither"]
    hbars(a, [VLAB[v] for v in vs], [m["variants"][v]["P_F1"] for v in vs], [VCOL[v] for v in vs])
    k, n = m["runs"]["live_P_F1_count"], m["runs"]["n_runs"]
    lo, hi = cp(k, n)
    a.axvspan(lo, hi, color=INK, alpha=0.08, linewidth=0)
    a.axvline(k / n, color=INK, linewidth=1, linestyle="--")
    a.text(k / n + 0.004, -0.62, f"live {k}/{n} (95% CI shaded)", ha="left", va="center", fontsize=6, color=INK)
    a.set_ylim(3.5, -0.95)
    a.set_xlabel("P(all high in rounds 16–20)", color=INK2)
    a.set_title("A  Audit keeps which messages? (exact)", loc="left", color=INK)
    ps = ["unmodified", "own_null", "neighbour_null", "both_null"]
    hbars(b, [PLAB[p] for p in ps], [e2["processes"][p]["P_F1"] for p in ps], [INK2] * 4)
    b.set_ylim(3.5, -0.95)
    b.set_xlabel("P(all high in rounds 16–20)", color=INK2)
    b.set_title("B  Agents shown which messages? (exact)", loc="left", color=INK)
    sv = m["mechanism"]["survival_from_both_entry_distribution"]
    h = np.arange(len(sv["both"]))
    c.tick_params(labelsize=6.5)
    for v in vs:
        ls = (0, (3, 2)) if v == "neighbours_only" else "-"
        lab = VLAB[v] + (" (on top of\nactions only)" if v == "neighbours_only" else "")
        c.plot(h, sv[v], color=VCOL[v], linewidth=1.6, linestyle=ls, label=lab, zorder=3 if v == "neighbours_only" else 2)
    c.set_ylim(0, 1.02)
    c.set_xlim(0, h[-1])
    c.set_xlabel("rounds since entering all-high", color=INK2)
    c.set_ylabel("P(still all-high)", color=INK2)
    c.grid(True, color=GRID, linewidth=0.5)
    c.legend(frameon=False, fontsize=5.5, loc="center left", bbox_to_anchor=(0.22, 0.5), handlelength=1.6)
    c.set_title("C  Staying all-high", loc="left", color=INK)
    a.set_title("A  Audit keeps which messages? (exact)", loc="left", color=INK)
    b.set_title("B  Agents shown which? (exact)", loc="left", color=INK)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig2_mechanism")


def appendix(abl, obs):
    for model, mm in abl["models"].items():
        d = mm["mechanism"]
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.4), facecolor=SURFACE)
        for ax in (a, b):
            _style(ax)
        t = np.arange(1, 21)
        a.plot(t, d["occupancy_all_H"], color=VCOL["both"], linewidth=1.6, label="P(all-high), full chain")
        a.plot(t, d["entry_probability_per_round"], color=INK2, linewidth=1.2, linestyle="--", label="P(enter all-high)")
        a.set_xlabel("round", color=INK2)
        a.set_ylim(0, max(0.2, max(d["occupancy_all_H"]) * 1.15))
        a.legend(frameon=False, fontsize=6)
        a.grid(True, color=GRID, linewidth=0.5)
        a.set_title(f"{model}: all-high occupancy and entry (exact)", loc="left", color=INK)
        sh = d["own_message_share_within_all_H_per_round"]
        cols = {"high": "#eb6834", "none": "#2a78d6", "low": "#e87ba4"}
        for msg in ("high", "none", "low"):
            b.plot(t, sh[msg], color=cols[msg], linewidth=1.6, label=f"own last message {msg}")
        b.axhline(1 / 3, color=INK, linewidth=0.8, linestyle=":")
        b.text(1, 1 / 3 + 0.02, "audit weight 1/3", ha="left", fontsize=6, color=INK)
        b.set_ylim(0, 1)
        b.set_xlabel("round", color=INK2)
        b.set_ylabel("share within all-high profiles", color=INK2)
        b.legend(frameon=False, fontsize=6, loc="lower right", bbox_to_anchor=(1.0, 0.04))
        b.grid(True, color=GRID, linewidth=0.5)
        b.set_title("own-message composition inside all-high (exact)", loc="left", color=INK)
        fig.tight_layout()
        save(fig, f"appx_occupancy_{model}")

        grid = mm["kernel_grid_P_H"]
        rows = list(grid)
        comps = list(grid[rows[0]])
        M = np.array([[grid[r][c][o] for c in comps for o in ("high", "low", "none")] for r in rows])
        fig, ax = plt.subplots(figsize=(7.0, 2.2), facecolor=SURFACE)
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(rows)), [f"own last bid {r[0]}, {r[2]} high nbrs" for r in rows], fontsize=6)
        ax.set_xticks(range(M.shape[1]), [f"{c}\n{o}" for c in comps for o in ("high", "low", "none")], fontsize=5)
        ax.set_xlabel("neighbour message counts (high-low-none) / own last message", color=INK2)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=4.5,
                        color="white" if M[i, j] > 0.6 else INK)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="P(H next)")
        ax.set_title(f"{model}: K(H | own bid, high neighbours, neighbour messages, own message), enumerated table",
                     loc="left", color=INK, fontsize=7)
        fig.tight_layout()
        save(fig, f"appx_kernel_grid_{model}")

    # Llama control and the factorial, both models
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2), facecolor=SURFACE)
    for ax, model in zip(axs, abl["models"]):
        _style(ax)
        mv = abl["models"][model]["variants"]
        vs = ["both", "own_only", "neighbours_only", "neither"]
        xs = [mv[v]["E_F"] for v in vs]
        hbars(ax, [VLAB[v] for v in vs], xs, [VCOL[v] for v in vs], xmax=max(0.35, max(xs) * 1.3))
        ax.set_xlabel("E[F] (mean late high-bid fraction), exact", color=INK2)
        p1 = ", ".join(f"{fmt(mv[v]['P_F1'])}" for v in vs)
        ax.set_title(f"{model}: P(F=1) = {p1}", loc="left", color=INK, fontsize=6.5)
    fig.tight_layout()
    save(fig, "appx_llama_control")


def size_memory():
    p = F / "size_memory.json"
    if not p.exists():
        print("size_memory.json missing: skipped")
        return
    s = json.loads(p.read_text())
    cells = s["cells"]
    order = [("n4_last1", "n = 4, last round"), ("n4_full", "n = 4, full history"),
             ("n10_last1", "n = 10, last round"), ("n10_full", "n = 10, full history")]
    fig, ax = plt.subplots(figsize=(3.4, 2.2), facecolor=SURFACE)
    _style(ax)
    y = np.arange(len(order))
    for yi, (k, lab) in zip(y, order):
        c = cells[k]
        lo, hi = c["P_F1_ci"]
        ax.plot([lo, hi], [yi, yi], color=INK2, linewidth=1.2)
        ax.plot(c["P_F1"], yi, "o", color=INK, markersize=4)
        ax.text(hi + 0.01, yi, f"{c['F1_count']}/{c['runs']}", va="center", fontsize=6.5, color=INK)
        if c.get("forecast_P_F1") is not None:
            ax.plot(c["forecast_P_F1"], yi, "D", color=VCOL["both"], markersize=4, markerfacecolor="none")
    ax.plot([], [], "o", color=INK, markersize=4, label="live, 95% Clopper–Pearson")
    ax.plot([], [], "D", color=VCOL["both"], markersize=4, markerfacecolor="none", label="table forecast")
    ax.set_yticks(y, [lab for _, lab in order], color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.45)
    ax.set_xlabel("P(all agents high, rounds 16–20)", color=INK2)
    ax.grid(True, axis="x", color=GRID, linewidth=0.5)
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    ax.set_title("OLMo: size by memory (live, 50 runs per cell)", loc="left", color=INK)
    fig.tight_layout()
    save(fig, "appx_size_memory")


def main():
    abl = json.loads((F / "audit_ablations.json").read_text())
    obs = json.loads((F / "observation_interventions.json").read_text())
    figure2(abl, obs)
    appendix(abl, obs)
    size_memory()


if __name__ == "__main__":
    main()
