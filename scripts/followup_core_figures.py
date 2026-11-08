#!/usr/bin/env python3
"""Figures for the follow-up evidence package (results/followup/core/figures/), PDF + PNG.

Figure 1: the primary endpoint contrast with every uncertainty layer, observed and
predicted X_t, and the signed one-step error by round (Llama, cheap talk, n = 10).
Figure 3 candidate: the clock-indexed audit against the pooled audit (exploratory).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from swarm_audit.plots import COLOURS, INK, INK2, LABELS, SURFACE, _style, plt  # noqa: E402

FU = REPO / "results" / "followup" / "core"
FIG = FU / "figures"
PRIMARY = "core_cheap_talk__cheap_talk_n10_cycle_full"
CLOCK_COLOUR = "#6250d6"     # reference palette slot 7 (violet); slots 1-5 are taken in plots.py


# Uncertainty layers are not entities: they are drawn in ink shades and told apart by marker.
LAYER3, LAYER4 = "#7f7e7a", "#a9a8a3"


def _int_rounds(*axes) -> None:
    import matplotlib.ticker as mt

    for ax in axes:
        ax.xaxis.set_major_locator(mt.MultipleLocator(2))


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", facecolor=SURFACE, dpi=200)
    plt.close(fig)


def _lab(ax, x, y):
    ax.set_xlabel(x, color=INK2, fontsize=9)
    ax.set_ylabel(y, color=INK2, fontsize=9)


def figure1() -> None:
    r = json.loads((FU / "temporal" / f"{PRIMARY}.json").read_text())
    key = "actions_only - signal_aware"
    orig = r["original_simulation"]["paired"][key]
    prec = r["monte_carlo"].get("precision_run", r["monte_carlo"]["pooled"])["paired"][key]
    tab = r["table_resampling"]["summary"]["paired"][key]
    nest = r["table_resampling"]["summary"]["paired_nested"][key]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.25, 1.35, 1.1]})
    for ax in axes:
        _style(ax)

    # (a) paired D per block, four uncertainty layers.
    a = axes[0]
    blocks = [("endpoint", "endpoint y\n(rounds 16-20)"), ("2-7", "rounds 2-7"),
              ("8-13", "rounds 8-13"), ("14-20", "rounds 14-20")]
    layers = [("original sims, runs resampled", orig, INK, "o"),
              (f"{r['monte_carlo'].get('precision_run', {}).get('n_model', 200000):,} sims, runs resampled",
               prec, INK2, "s"),
              ("tables resampled (+ sims)", tab, LAYER3, "D"),
              ("runs x tables x sims (nested)", nest, LAYER4, "^")]
    for i, (bk, _) in enumerate(blocks):
        for j, (lab, v, c, mk) in enumerate(layers):
            y = i + (j - 1.5) * 0.17
            a.plot([v[bk]["lo"] * 1e4, v[bk]["hi"] * 1e4], [y, y], color=c, linewidth=2)
            a.plot(v[bk]["mean"] * 1e4, y, marker=mk, color=c, markersize=6, linestyle="none",
                   markeredgecolor=SURFACE, markeredgewidth=1, label=lab if i == 0 else None)
    a.axvline(0, color=INK2, linewidth=1)
    a.set_yticks(range(len(blocks)), [b[1] for b in blocks], fontsize=8)
    a.invert_yaxis()
    _lab(a, "D = CRPS(actions-only) - CRPS(signal-aware)  [x 1e-4]\n< 0: actions-only forecast better", "")
    a.legend(frameon=False, fontsize=7, loc="lower left")
    a.set_title("(a) paired contrast, 95% intervals", fontsize=9, color=INK, loc="left")

    # (b) observed and predicted X_t, with the table-resampling band.
    b = axes[1]
    T = len(r["X"]["swarm"]["mean"])
    t = np.arange(1, T + 1)
    sw = r["X"]["swarm"]
    b.fill_between(t, sw["lo"], sw["hi"], color=INK, alpha=0.12, linewidth=0)
    b.plot(t, sw["mean"], color=INK, linewidth=2, marker="o", markersize=4, label="swarm (50 runs)")
    for m in ("actions_only", "signal_aware", "order0"):
        band = r["table_resampling"]["x_band"].get(m)
        if band is not None and m != "order0":
            b.fill_between(t, band["lo"], band["hi"], color=COLOURS[m], alpha=0.18, linewidth=0)
        b.plot(t, r["X"][m]["mean"], color=COLOURS[m], linewidth=2,
               linestyle="--" if m == "order0" else "-", label=LABELS[m])
    b.set_xlim(1, T)
    b.set_ylim(0, max(0.6, float(np.max(sw["hi"])) + 0.05))
    _lab(b, "round t", "mean fraction bidding high, X_t")
    b.legend(frameon=False, fontsize=7, loc="upper right")
    b.set_title("(b) observed vs autonomous forecasts", fontsize=9, color=INK, loc="left")

    # (c) signed one-step error by round.
    c = axes[2]
    for m in ("actions_only", "signal_aware"):
        e = r["one_step"][m]["by_round"]["signed"]
        mean, lo, hi = (np.asarray(e[k], float) for k in ("mean", "lo", "hi"))
        c.fill_between(t[1:], lo[1:], hi[1:], color=COLOURS[m], alpha=0.18, linewidth=0)
        c.plot(t[1:], mean[1:], color=COLOURS[m], linewidth=2, label=LABELS[m])
    c.axhline(0, color=INK2, linewidth=1)
    c.set_xlim(2, T)
    _lab(c, "round t", "mean p_H - kappa(H | s)  (swarm histories)")
    c.legend(frameon=False, fontsize=7, loc="lower right")
    c.set_title("(c) signed one-step error, teacher-forced", fontsize=9, color=INK, loc="left")
    _int_rounds(b, c)
    fig.suptitle("Llama 3.1 8B, cheap talk, cycle n = 10, full history: the endpoint null hides "
                 "an early-round bias", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig1_primary_temporal")


def figure3() -> None:
    p = FU / "clock" / f"{PRIMARY}.json"
    mp = FU / "mechanism" / f"{PRIMARY}.json"
    if not p.exists() or not mp.exists():
        print("no clock or mechanism result; Figure 3 skipped")
        return
    r = json.loads(p.read_text())
    mech = json.loads(mp.read_text())
    tr = json.loads((FU / "temporal" / f"{PRIMARY}.json").read_text())
    m = "signal_aware"
    comp = r["comparison"][m]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.2, 1.2, 1.0]})
    for ax in axes:
        _style(ax)
    sw = tr["X"]["swarm"]
    T = len(sw["mean"])
    t = np.arange(1, T + 1)

    a = axes[0]
    a.fill_between(t, sw["lo"], sw["hi"], color=INK, alpha=0.12, linewidth=0)
    a.plot(t, sw["mean"], color=INK, linewidth=2, marker="o", markersize=4, label="swarm (50 runs)")
    a.plot(t, comp["mean_X_pooled"], color=COLOURS[m], linewidth=2, label="pooled audit (L = 1-19)")
    a.plot(t, comp["mean_X_clock"], color=CLOCK_COLOUR, linewidth=2, label="clock-indexed audit")
    a.set_xlim(1, T)
    a.set_ylim(0, max(0.6, float(np.max(sw["hi"])) + 0.05))
    _lab(a, "round t", "mean fraction bidding high, X_t")
    a.legend(frameon=False, fontsize=7, loc="upper right")
    a.set_title("(a) signal-aware forecasts", fontsize=9, color=INK, loc="left")

    b = axes[1]
    for key, c, lab in (("signal_aware_pooled", COLOURS[m], "pooled table"),
                        ("signal_aware_clock", CLOCK_COLOUR, "clock-indexed table")):
        e = mech["errors"][key]["by_round"]
        mean, lo, hi = (np.asarray(e[k], float) for k in ("mean", "lo", "hi"))
        b.fill_between(t[1:], lo[1:], hi[1:], color=c, alpha=0.18, linewidth=0)
        b.plot(t[1:], mean[1:], color=c, linewidth=2, label=lab)
    b.axhline(0, color=INK2, linewidth=1)
    b.set_xlim(2, T)
    _lab(b, "round t", "mean p_H - kappa(H | s)  (swarm histories)")
    b.legend(frameon=False, fontsize=7, loc="lower left")
    b.set_title("(b) signed one-step error, teacher-forced", fontsize=9, color=INK, loc="left")

    c_ = axes[2]
    keys = [("2-7", "rounds 2-7 (primary)"), ("8-13", "rounds 8-13"), ("14-20", "rounds 14-20"),
            ("endpoint", "endpoint y")]
    cond = comp["improvement_pooled_minus_clock"]
    nest = r["table_sensitivity"]["results"][m]
    for i, (k, _) in enumerate(keys):
        for j, (lab, v, col, mk) in enumerate([("runs resampled", cond[k], INK, "o"),
                                                ("runs x tables x sims", nest[k]["nested_runs_tables_simulation"],
                                                 LAYER3, "^")]):
            y = i + (j - 0.5) * 0.25
            c_.plot([v["lo"] * 1e3, v["hi"] * 1e3], [y, y], color=col, linewidth=2)
            c_.plot(v["mean"] * 1e3, y, marker=mk, color=col, markersize=6, linestyle="none",
                    markeredgecolor=SURFACE, markeredgewidth=1, label=lab if i == 0 else None)
    c_.axvline(0, color=INK2, linewidth=1)
    c_.set_yticks(range(len(keys)), [k[1] for k in keys], fontsize=8)
    c_.invert_yaxis()
    _lab(c_, "CRPS(pooled) - CRPS(clock)  [x 1e-3]\n> 0: clock-indexed audit better", "")
    c_.legend(frameon=False, fontsize=7, loc="lower right")
    c_.set_title("(c) paired improvement, 95% intervals", fontsize=9, color=INK, loc="left")
    _int_rounds(a, b)
    fig.suptitle("Exploratory repair: matching history length fixes the early bias and exposes a "
                 "content shift later (same 50 runs; not an independent validation)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig3_clock_audit")


if __name__ == "__main__":
    figure1()
    figure3()
    print(f"figures in {FIG}")
