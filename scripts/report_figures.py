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
- figA_contraction.pdf: the primary Llama setting's current-round TV bound estimates from
  results/followup/dobrushin/ (Theorem 1, contraction with the same per-agent defects,
  contraction with joint one-step defects) against the observed roundwise TV and its
  simultaneous sampling cutoff.

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
CONTRACTION = REPO / "results/followup/dobrushin/dobrushin_results.json"
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
    """Figure 1: one takeaway per panel, text at about 9 pt at print size."""
    ten = json.loads(OLMO10.read_text())
    four = json.loads(FRESH.read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.35), facecolor=SURFACE,
                               gridspec_kw={"width_ratios": [1.0, 1.0]})
    for ax in (a, b):
        style(ax)
        ax.tick_params(labelsize=9)
    blue, orange = COLOURS["actions_only"], COLOURS["signal_aware"]
    percent = matplotlib.ticker.PercentFormatter(1.0, decimals=0)

    # (A) timing, 10 agents: the round-3 gap is the first thing the eye sees.
    X = ten["X"]
    t = np.arange(1, len(X["swarm"]["mean"]) + 1)
    live, ao, sa = (np.asarray(X[k]["mean"]) for k in ("swarm", "actions_only", "signal_aware"))
    a.fill_between(t, X["swarm"]["lo"], X["swarm"]["hi"], color=INK, alpha=0.12, linewidth=0)
    a.plot(t, ao, color=blue, linewidth=2.2, label="actions-only audit")
    a.plot(t, sa, color=orange, linewidth=2.2, linestyle=(0, (4, 2)), label="signal-aware audit")
    a.plot(t, live, color=INK, linewidth=2, marker="o", markersize=3.5, label="live games")
    a.annotate("", xy=(3, live[2] + 0.012), xytext=(3, ao[2] - 0.012),
               arrowprops={"arrowstyle": "<->", "color": INK, "linewidth": 1.3,
                           "shrinkA": 0, "shrinkB": 0})
    a.annotate(f"round 3: {100 * live[2]:.0f}% live\nvs {100 * ao[2]:.0f}–{100 * sa[2]:.0f}% forecast",
               xy=(3.15, (live[2] + ao[2]) / 2), xytext=(6.2, 0.37), fontsize=9, color=INK,
               va="center", ha="left",
               arrowprops={"arrowstyle": "-", "color": INK2, "linewidth": 0.8})
    a.set_xlim(1, 20)
    a.set_ylim(0, 0.7)
    a.set_xticks([1, 3, 5, 10, 15, 20])
    a.yaxis.set_major_formatter(percent)
    a.set_xlabel("round", color=INK2, fontsize=9)
    a.set_ylabel("agents bidding high", color=INK2, fontsize=9)
    a.legend(frameon=False, loc="upper right", fontsize=9, handlelength=2.2)
    a.set_title("A  10 agents, full history\nexpected ending, reached faster than forecast",
                loc="left", color=INK, fontsize=9.5, fontweight="bold", linespacing=1.3)

    # (B) the final-five-round all-high event, 4 agents.
    ex = four["exact_P_F1"]
    co = four["cohorts"]
    rows = [("actions-only audit", ex["actions_only"], None, blue),
            ("signal-aware audit", ex["signal_aware"], None, orange),
            ("live, original games", co["original"], co["original"]["live_F1_ci_clopper_pearson"], INK),
            ("live, fresh games", co["fresh"], co["fresh"]["live_F1_ci_clopper_pearson"], INK)]
    for i, (lab, v, ci, col) in enumerate(rows):
        if ci is None:   # exact audit forecasts on the cached tables: bars, no interval
            b.barh(i, v, height=0.46, color=col, edgecolor=SURFACE, linewidth=1)
            txt = "0.0076%" if v < 0.001 else f"{100 * v:.1f}%"
            b.text(v + 0.01, i, txt, va="center", fontsize=9, color=INK,
                   fontweight="bold" if v < 0.001 else "normal")
        else:            # live rates: point and 95% Clopper-Pearson interval
            k, n = v["live_F1"], v["runs"]
            b.plot(ci, [i, i], color=col, linewidth=2)
            b.plot(k / n, i, marker="o", markersize=7, color=col, markeredgecolor=SURFACE,
                   markeredgewidth=1)
            b.text(ci[1] + 0.01, i, f"{100 * k / n:.0f}% ({k}/{n})", va="center", fontsize=9,
                   color=INK)
    b.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
    b.invert_yaxis()
    b.set_xlim(0, 0.45)
    b.xaxis.set_major_formatter(percent)
    b.set_xlabel("chance of the final-five-round all-high event", color=INK2, fontsize=9)
    b.text(0.44, 0.5, "roundwise check: not flagged\nin either live cohort", fontsize=9,
           color=INK, ha="right", va="center")
    b.set_title("B  4 agents, last round only\na persistent outcome the audit called rare",
                loc="left", color=INK, fontsize=9.5, fontweight="bold", linespacing=1.3)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "fig1_olmo_lead.pdf", facecolor=SURFACE, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)


def _pp(v, _pos=None) -> str:
    """Tick label for a probability difference, in percentage points."""
    return f"{100 * v:.0f}"


def _relabel(fig, mapping: dict[str, str]) -> None:
    """Replace whole text strings (titles, tick labels, legend entries) in a figure."""
    for txt in fig.findobj(matplotlib.text.Text):
        new = mapping.get(txt.get_text())
        if new is not None:
            txt.set_text(new)
    for ax in fig.axes:
        labels = [t.get_text() for t in ax.get_yticklabels()]
        if any(lab in mapping for lab in labels):
            ax.set_yticks(ax.get_yticks(), [mapping.get(lab, lab) for lab in labels])


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
        _relabel(fig, {"rounds 2-7 (primary)": "rounds 2-7 (early)",
                       "endpoint y\n(rounds 16-20)": "endpoint F\n(rounds 16-20)",
                       "endpoint y": "endpoint F", "swarm (50 runs)": "live games",
                       "pooled audit (L = 1-19)": "pooled tables (lengths 1-19)",
                       "clock-indexed audit": "history-length tables",
                       "pooled table": "pooled tables", "clock-indexed table": "history-length tables",
                       "CRPS(pooled) - CRPS(clock)  [x 1e-3]\n> 0: clock-indexed audit better":
                       "CRPS(pooled) - CRPS(history-length)  [x 1e-3]\n> 0: history-length tables better"})
        for ax in fig.axes:
            ylab = ax.get_ylabel()
            if "fraction bidding high" in ylab:
                ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
                ax.set_ylabel("agents bidding high", color=INK2, fontsize=9)
            elif "kappa" in ylab:
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_pp))
                ax.set_ylabel("one-step error (percentage points)", color=INK2, fontsize=9)
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
        def as_pct(text):
            try:
                v = float(text)
            except ValueError:
                return None
            return f"{100 * v:.4f}%".rstrip("0") if v < 0.001 else f"{100 * v:.1f}%"
        mapping = {"full table": "signal-aware", "own msg only": "own message only",
                   "nbr msgs only": "neighbours only", "actions only": "actions-only",
                   "nbr msgs only (on top of\nactions only)": "neighbours only\n(on top of actions-only)",
                   "live 8/50 (95% CI shaded)": "live 16% (8/50), 95% interval shaded",
                   "P(all high in rounds 16–20)": "final-five-round all-high event",
                   "P(still all-high)": "still all-high",
                   "A  Audit keeps which messages? (exact)": "A  Audit keeps which messages?",
                   "B  Agents shown which? (exact)": "B  Agents shown which messages?",
                   "C  Staying all-high": "C  Staying all-high"}
        for txt in fig.findobj(matplotlib.text.Text):
            t = txt.get_text()
            if t in mapping:
                txt.set_text(mapping[t])
            elif as_pct(t) is not None and not isinstance(txt, matplotlib.text.Annotation) \
                    and txt.axes is not None and txt not in txt.axes.get_xticklabels() \
                    and txt not in txt.axes.get_yticklabels():
                txt.set_text(as_pct(t))
        for ax in fig.axes:
            labels = [x.get_text() for x in ax.get_yticklabels()]
            if any(lab in mapping for lab in labels):
                ax.set_yticks(ax.get_yticks(), [mapping.get(lab, lab) for lab in labels])
            if ax.get_xlabel() == "final-five-round all-high event":
                ax.set_xticks([0, 0.1, 0.2])
                ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
            if ax.get_ylabel() == "still all-high":
                ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        for txt in fig.findobj(matplotlib.text.Text):
            if txt.get_text():
                txt.set_fontsize(min(9.0, max(7.5, round(txt.get_fontsize() * 1.2, 1))))
        for ax in fig.axes:
            ax.tick_params(labelsize=7.5)
        fig.set_size_inches(WIDTH, 3.0)
        fig.tight_layout(w_pad=0.6)
        fig.savefig(OUT / "figA_olmo_mechanism.pdf", facecolor=SURFACE, bbox_inches="tight",
                    pad_inches=0.02)
        plt.close(fig)

    eff.save = save
    F = eff.F
    eff.figure2(json.loads((F / "audit_ablations.json").read_text()),
                json.loads((F / "observation_interventions.json").read_text()))


def contraction_figure() -> None:
    """Primary Llama setting: bound estimates for the current-round bid-profile TV."""
    res = json.loads(CONTRACTION.read_text())["results"]
    r = next(x for x in res if x["condition"]["channel"] == "cheap_talk"
             and x["condition"]["n"] == 10 and x["condition"]["graph"] == "cycle"
             and x["condition"]["last_k"] is None)
    t = np.asarray(r["rounds"])
    cap = lambda v: np.minimum(np.asarray(v, dtype=float), 1.0)  # noqa: E731
    th1 = cap(r["theorem1"])
    same = cap(r["tv_contraction_same_local_defects"])
    joint = cap(r["tv_contraction_joint_defect"])
    lo, hi = (cap(v) for v in r["tv_contraction_joint_defect_ci"])
    obs = np.asarray(r["observed_binned_tv"])
    cut = r["observed_tv_null"]["max_over_rounds"]
    blue = COLOURS["actions_only"]   # every curve concerns the actions-only audit

    fig, ax = plt.subplots(figsize=(WIDTH, 3.3), facecolor=SURFACE)
    style(ax)
    ax.plot(t, th1, color=INK2, linewidth=1.6, linestyle=(0, (5, 3)),
            label="Theorem 1, full trajectory (capped at 1)")
    ax.plot(t, same, color=blue, linewidth=1.6, linestyle=(0, (1.5, 2)),
            label="contraction, same per-agent defects")
    ax.fill_between(t, lo, hi, color=blue, alpha=0.18, linewidth=0)
    ax.plot(t, joint, color=blue, linewidth=2.2,
            label="contraction, joint one-step defects (95% band)")
    ax.axhline(cut, color=INK2, linewidth=0.9, linestyle=":",
               label=f"simultaneous cutoff under the audit model ({cut:.3f})")
    ax.plot(t, obs, color=INK, linewidth=1.4, marker="o", markersize=3.5,
            label="observed roundwise TV, 50 live games")
    for y, txt in ((th1[-1], f"{th1[-1]:.0f}"), (same[-1], f"{same[-1]:.3f}"),
                   (joint[-1], f"{joint[-1]:.3f}")):
        ax.text(t[-1] + 0.3, y, txt, va="center", ha="left", fontsize=9, color=INK)
    ax.set_xlim(1, t[-1])
    ax.set_ylim(0, 1.05)
    ax.set_xticks([1, 3, 5, 10, 15, 20])
    ax.set_xlabel("round", color=INK2, fontsize=9)
    ax.set_ylabel("TV distance (0–1)", color=INK2, fontsize=9)
    ax.tick_params(labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2),
              ncol=2, handlelength=2.4, columnspacing=1.5)
    fig.tight_layout()
    fig.savefig(OUT / "figA_contraction.pdf", facecolor=SURFACE, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    lead_figure()
    contraction_figure()       # before the imported modules change matplotlib's defaults
    llama_figures()
    olmo_mechanism_figure()
    print(f"figures in {OUT.relative_to(REPO)}")
