#!/usr/bin/env python3
"""Figures for the final report (report/figures/), drawn at their printed size.

- fig1_olmo_lead.pdf: the lead figure. (A) OLMo, 10 agents, full history: live and
  forecast high-bid fractions by round. (B) OLMo, 4 agents, last-round history: the
  probability that every agent bids high in every round 16-20, under both audits and in
  both live cohorts.
- figA_llama_endpoint.pdf, figA_llama_clock.pdf: the Llama follow-up figures of
  scripts/followup_core_figures.py, re-laid out to the report's text width (one wide panel
  over two) without their titles, with the early window labelled "(early)".
- figA_olmo_mechanism.pdf: Figure 2 of scripts/extra_followup_figures.py, redrawn at the
  report's text width in the report's font, with its text enlarged to at least 7 pt.

CPU only; reads committed results. Run scripts/report_fresh_cohort.py first.

    python scripts/report_figures.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from swarm_audit.plots import COLOURS, GRID, INK, INK2, SURFACE  # noqa: E402

OUT = REPO / "report" / "figures"
OLMO10 = REPO / "results/extra/analysis/olmo_core_cheap_talk/cheap_talk_n10_cycle_full.json"
FRESH = REPO / "results/extra/followup/fresh_cohort_roundwise.json"
WIDTH = 6.5  # inches: the width at which the report prints every figure

if any("Carlito" in f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams["font.family"] = "Carlito"   # metric-compatible with the report's body font
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "pdf.fonttype": 42})


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def pct(p: float) -> str:
    return f"{100 * p:.1f}%"


def lead_figure() -> None:
    ten = json.loads(OLMO10.read_text())
    four = json.loads(FRESH.read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.9), facecolor=SURFACE,
                               gridspec_kw={"width_ratios": [1.0, 1.08]})
    for ax in (a, b):
        style(ax)

    # (A) timing, 10 agents.
    X = ten["X"]
    t = np.arange(1, len(X["swarm"]["mean"]) + 1)
    a.fill_between(t, X["swarm"]["lo"], X["swarm"]["hi"], color=INK, alpha=0.12, linewidth=0)
    a.plot(t, X["actions_only"]["mean"], color=COLOURS["actions_only"], linewidth=2,
           label="actions-only audit")
    a.plot(t, X["signal_aware"]["mean"], color=COLOURS["signal_aware"], linewidth=2,
           label="signal-aware audit")
    a.plot(t, X["swarm"]["mean"], color=INK, linewidth=2, marker="o", markersize=3.5,
           label="live runs (95% interval)")
    a.axvline(3, color=INK2, linewidth=0.8, linestyle=":")
    sc, fl = ten["scores"], ten["scores"]["floor"]
    da, ds = max(sc["delta"]["actions_only"]), max(sc["delta"]["signal_aware"])
    a.annotate(f"round 3: live {pct(X['swarm']['mean'][2])}, audits "
               f"{pct(X['actions_only']['mean'][2])} and {pct(X['signal_aware']['mean'][2])}\n"
               f"max roundwise TV {da:.3f} and {ds:.3f}\n"
               f"(cutoffs {fl['actions_only']['max_over_rounds']:.3f} and "
               f"{fl['signal_aware']['max_over_rounds']:.3f}): flagged",
               xy=(3, X["actions_only"]["mean"][2]), xytext=(5.2, 0.43), fontsize=7.5,
               color=INK, va="center",
               arrowprops={"arrowstyle": "-", "color": INK2, "linewidth": 0.8})
    a.text(20, 0.13, f"all low in rounds 16–20:\nlive 50 of 50; audits "
           f"{100 * sc['G']['actions_only'][0]:.1f}% and {100 * sc['G']['signal_aware'][0]:.1f}%",
           fontsize=7.5, color=INK, ha="right", va="bottom")
    a.set_xlim(1, 20)
    a.set_ylim(0, 0.7)
    a.set_xticks([1, 5, 10, 15, 20])
    a.set_xlabel("round", color=INK2)
    a.set_ylabel("fraction bidding high", color=INK2)
    a.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.02), handlelength=1.6)
    a.set_title("A  OLMo, 10 agents, full history", loc="left", color=INK, fontweight="bold")

    # (B) the persistent all-high event, 4 agents.
    ex = four["exact_P_F1"]
    co = four["cohorts"]
    rows = [("actions-only audit", ex["actions_only"], None, COLOURS["actions_only"]),
            ("signal-aware audit", ex["signal_aware"], None, COLOURS["signal_aware"]),
            ("live, original 50 runs", co["original"]["live_F1"] / co["original"]["runs"],
             co["original"]["live_F1_ci_clopper_pearson"], INK),
            ("live, fresh 50 runs", co["fresh"]["live_F1"] / co["fresh"]["runs"],
             co["fresh"]["live_F1_ci_clopper_pearson"], INK)]
    for i, (lab, p, ci, col) in enumerate(rows):
        if ci is None:   # exact probabilities on the cached tables: bars, no interval
            b.barh(i, p, height=0.42, color=col, edgecolor=SURFACE, linewidth=1)
            txt = f"{100 * p:.4f}%".rstrip("0") if p < 0.001 else f"{100 * p:.2f}%"
            b.text(max(p, 0) + 0.008, i, txt, va="center", fontsize=8, color=INK)
        else:            # live rates: point and 95% Clopper-Pearson interval
            b.plot(ci, [i, i], color=col, linewidth=2)
            b.plot(p, i, marker="o", markersize=6, color=col, markeredgecolor=SURFACE,
                   markeredgewidth=1)
            k = co["original" if "original" in lab else "fresh"]["live_F1"]
            b.text(ci[1] + 0.008, i, f"{k}/50", va="center", fontsize=8, color=INK)
    b.set_yticks(range(len(rows)), [r[0] for r in rows])
    b.invert_yaxis()
    b.set_xlim(0, 0.42)
    b.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    b.set_xlabel("probability all agents bid high in every round 16–20", color=INK2)
    ro, rf = (co[c]["roundwise"]["actions_only"] for c in ("original", "fresh"))
    b.text(0.415, 0.0, f"actions-only roundwise check:\nmax TV {ro['max_delta']:.3f} (original), "
           f"{rf['max_delta']:.3f} (fresh),\nboth at round {ro['round_of_max']}, below the "
           f"cutoff {ro['original_cutoff']:.3f}: not flagged", fontsize=7.5, color=INK,
           ha="right", va="center")
    b.set_title("B  OLMo, 4 agents, last-round history", loc="left", color=INK,
                fontweight="bold")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig1_olmo_lead.pdf", facecolor=SURFACE, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)


def llama_figures() -> None:
    """Re-lay out the Llama follow-up figures at the report's width."""
    spec = importlib.util.spec_from_file_location("fcf", REPO / "scripts/followup_core_figures.py")
    fcf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fcf)
    names = {"fig1_primary_temporal": "figA_llama_endpoint", "fig3_clock_audit": "figA_llama_clock"}

    def subplots(nrows, ncols, **kw):     # 1 x 3 -> one wide panel over two
        fig = plt.figure(figsize=(WIDTH, 5.4), facecolor=SURFACE)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.45, wspace=0.35)
        axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
        fig.suptitle = lambda *a, **k: None   # the report's caption replaces the title
        return fig, axes

    def save(fig, name):
        for txt in fig.findobj(matplotlib.text.Text):
            if txt.get_text() == "rounds 2-7 (primary)":
                txt.set_text("rounds 2-7 (early)")
        for ax in fig.axes:
            for lab in ax.get_yticklabels():
                if lab.get_text() == "rounds 2-7 (primary)":
                    ax.set_yticks(ax.get_yticks(), [t.get_text().replace("(primary)", "(early)")
                                                    for t in ax.get_yticklabels()])
                    break
        fig.savefig(OUT / f"{names[name]}.pdf", facecolor=SURFACE, bbox_inches="tight",
                    pad_inches=0.02)
        plt.close(fig)

    original = plt.subplots
    fcf.plt.subplots = subplots           # fcf.plt is this module's pyplot: restore it after
    fcf.save = save
    try:
        fcf.figure1()
        fcf.figure3()
    finally:
        plt.subplots = original


def olmo_mechanism_figure() -> None:
    """Redraw the four-agent mechanism figure at the report's width with legible text."""
    spec = importlib.util.spec_from_file_location("eff", REPO / "scripts/extra_followup_figures.py")
    eff = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eff)
    plt.rcParams["font.family"] = "Carlito"   # the module sets its own font on import

    def save(fig, name):
        for txt in fig.findobj(matplotlib.text.Text):
            if txt.get_text():
                txt.set_fontsize(min(9.0, max(7.0, round(txt.get_fontsize() * 1.2, 1))))
        for ax in fig.axes:
            ax.tick_params(labelsize=7.5)
        fig.set_size_inches(WIDTH, 2.8)
        fig.tight_layout(w_pad=0.6)
        fig.savefig(OUT / "figA_olmo_mechanism.pdf", facecolor=SURFACE, bbox_inches="tight",
                    pad_inches=0.02)
        plt.close(fig)

    eff.save = save
    F = eff.F
    eff.figure2(json.loads((F / "audit_ablations.json").read_text()),
                json.loads((F / "observation_interventions.json").read_text()))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    lead_figure()
    llama_figures()
    olmo_mechanism_figure()
    print(f"figures in {OUT.relative_to(REPO)}")
