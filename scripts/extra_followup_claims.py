#!/usr/bin/env python3
"""claims.csv for the overnight follow-up: every number read from the result files.

    python scripts/extra_followup_claims.py

Columns: claim, value, interval_or_precision, condition, provenance (file and key),
evidence_type, allowed_wording.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
F = REPO / "results" / "extra" / "followup"
EXACT = "exact chain law on the cached enumerated kernel (numerical tolerance 1e-10, independently checked)"


def load(name):
    p = F / name
    return json.loads(p.read_text()) if p.exists() else None


def f(x, d=4):
    return f"{x:.{d}f}" if abs(x) >= 10 ** -d or x == 0 else f"{x:.2e}"


def main() -> None:
    abl, obs = load("audit_ablations.json"), load("observation_interventions.json")
    live, sm = load("observation_live.json"), load("size_memory.json")
    o = abl["models"]["olmo-3-7b"]
    v = o["variants"]
    mech = o["mechanism"]
    r1 = mech["round1_all_H"]
    e2 = obs["models"]["olmo-3-7b"]
    ll = abl["models"]["llama-3.1-8b"]["variants"]
    cond = "OLMo-3-7B, n = 4 cycle, cheap talk, last-1 history, T = 20"
    rows = []

    def add(claim, value, prec, condition, prov, ev, wording):
        rows.append([claim, value, prec, condition, prov, ev, wording])

    for k, lab in (("both", "full signal-aware table"), ("own_only", "own last message retained, neighbour messages averaged"),
                   ("neighbours_only", "neighbour messages retained, own message averaged"),
                   ("neither", "both averaged (= derived actions-only audit)")):
        add(f"P(F=1) under the {lab}", f(v[k]["P_F1"], 6), EXACT, cond,
            f"audit_ablations.json models.olmo-3-7b.variants.{k}.P_F1", "exact",
            "audit variant forecast; not a live measurement")
    add("Smallest tested retained information that recovers the full-table endpoint", o["smallest_recovering_variant"],
        "tolerance |dP(F=1)| <= 0.01 and expected excess CRPS <= 0.001 (frozen in plan.json)", cond,
        "audit_ablations.json models.olmo-3-7b.smallest_recovering_variant", "exact",
        "Among the four tested variants, keeping the agent's own last message suffices to recover the "
        "persistent-high forecast; this is not universal sufficiency")
    add("Expected excess CRPS of the actions-only (neither) forecast under the full law", f(v["neither"]["expected_excess_crps_vs_both"], 5),
        EXACT + "; identity sum (Q-G)^2/K (standard, not novel)", cond,
        "audit_ablations.json models.olmo-3-7b.variants.neither.expected_excess_crps_vs_both", "exact",
        "expected CRPS advantage of the full table if the full chain were true")
    add("Expected excess CRPS of the own-message-only forecast under the full law", f(v["own_only"]["expected_excess_crps_vs_both"], 6),
        EXACT, cond, "audit_ablations.json models.olmo-3-7b.variants.own_only.expected_excess_crps_vs_both", "exact", "as above")
    for k in ("own_only", "neither"):
        d = v[k]["live_crps_minus_both"]
        add(f"Live CRPS({k}) - CRPS(full table), 50 arm 9a runs", f(d["mean"], 5),
            f"95% whole-run bootstrap [{f(d['lo'], 5)}, {f(d['hi'], 5)}], B = 2000", cond,
            f"audit_ablations.json models.olmo-3-7b.variants.{k}.live_crps_minus_both", "live",
            "empirical, conditional on the tables; the interval includes 0" if d["lo"] <= 0 <= d["hi"] else "empirical")
    add("Equal-weight sensitivity: P(F=1) own-message-only", f(v["own_only_equal_w"]["P_F1"], 6), EXACT, cond,
        "audit_ablations.json models.olmo-3-7b.variants.own_only_equal_w.P_F1", "exact", "sensitivity; same conclusion")
    sh = mech["own_message_share_within_all_H_rounds_2_20"]
    add("Share of own last message 'low' inside all-high profiles (occupancy-weighted, rounds 2-20)", f(sh["low"], 6),
        EXACT, cond, "audit_ablations.json models.olmo-3-7b.mechanism.own_message_share_within_all_H_rounds_2_20.low",
        "exact (oracle diagnostic)", "the audit gives each own message weight 1/3")
    add("Shares 'high' / 'none' inside all-high (rounds 2-20)", f"{f(sh['high'])} / {f(sh['none'])}", EXACT, cond,
        "audit_ablations.json models.olmo-3-7b.mechanism.own_message_share_within_all_H_rounds_2_20", "exact (oracle diagnostic)", "")
    lv = mech["live_own_message_share_in_all_H_rounds_2_20"]
    add("Live share of 'low' messages sent in all-high rounds 2-20", f(lv["low"]), f"{lv['agent_rounds']} agent-rounds",
        cond, "audit_ablations.json models.olmo-3-7b.mechanism.live_own_message_share_in_all_H_rounds_2_20", "live",
        "descriptive count from the 50 arm 9a runs")
    g = o["kernel_grid_P_H"]["H-2"]
    add("K(H | own H, 2 high neighbours, any neighbour messages, own 'low')",
        f"{min(x['low'] for x in g.values()):.3f}-{max(x['low'] for x in g.values()):.3f}", "enumerated table, exact", cond,
        "audit_ablations.json models.olmo-3-7b.kernel_grid_P_H.H-2", "table",
        "dependence on own-message history; not intention, commitment or collusion")
    add("K(H | own H, 2 high neighbours, any neighbour messages, own 'high' or 'none')",
        f"{min(min(x['high'], x['none']) for x in g.values()):.4f}-{max(max(x['high'], x['none']) for x in g.values()):.4f}",
        "enumerated table, exact", cond, "audit_ablations.json models.olmo-3-7b.kernel_grid_P_H.H-2", "table", "")
    ex = mech["exit_one_step_oracle"]
    add("One-step exit from all-high: full table vs actions-only (occupancy-weighted, rounds 2-20)",
        f"{f(ex['both']['occupancy_weighted_rounds_2_20'])} vs {f(ex['neither']['occupancy_weighted_rounds_2_20'])}",
        EXACT, cond, "audit_ablations.json models.olmo-3-7b.mechanism.exit_one_step_oracle", "exact (oracle diagnostic)",
        "weights are the full chain's occupancy, not an audit")
    sv = mech["survival_from_both_entry_distribution"]
    add("Finite-horizon persistence in all-high after 19 rounds: full / own-only / actions-only",
        f"{f(sv['both'][19])} / {f(sv['own_only'][19])} / {f(sv['neither'][19])}", EXACT, cond,
        "audit_ablations.json models.olmo-3-7b.mechanism.survival_from_both_entry_distribution", "exact",
        "finite-horizon persistence from a common entry distribution; not an absorbing equilibrium")
    add("Share of P(F=1) from runs all-high in round 1", f(r1["exact_share_of_F1_from_round1_all_H"]), EXACT, cond,
        "audit_ablations.json models.olmo-3-7b.mechanism.round1_all_H", "exact", "")
    add("Live: runs all-high in round 1 / of which F=1 / F=1 otherwise",
        f"{r1['live_runs_round1_all_H']} / {r1['live_F1_among_round1_all_H']} / {r1['live_F1_among_round1_not_all_H']}",
        "counts of 50 runs", cond, "audit_ablations.json models.olmo-3-7b.mechanism.round1_all_H", "live", "")
    for p, lab in (("unmodified", "as logged"), ("own_null", "own message shown as none"),
                   ("neighbour_null", "neighbour messages shown as none"), ("both_null", "both shown as none")):
        add(f"P(F=1) when agents are {lab}", f(e2["processes"][p]["P_F1"], 6),
            EXACT.replace("chain law", "counterfactual law"), cond,
            f"observation_interventions.json models.olmo-3-7b.processes.{p}.P_F1",
            "exact model-based counterfactual", "counterfactual observation intervention on the cached kernel; not a new live measurement")
    add("Factorial interaction on P(F=1) (both present - own only - neighbours only + neither)",
        f(e2["interaction"]["P_F1"], 5), EXACT, cond, "observation_interventions.json models.olmo-3-7b.interaction.P_F1",
        "exact model-based counterfactual", "channels available to agents")
    add("Uniform replacement of hidden messages has the same law as the averaged audits",
        f"max |dP| = {max(x['max_abs_diff_transition_vs_E1'] for x in e2['uniform_replacement_sensitivity'].values()):.1e}",
        "identity by construction, checked numerically", cond,
        "observation_interventions.json models.olmo-3-7b.uniform_replacement_sensitivity", "exact", "")
    add("Llama control: P(F=1) under all four variants", f"<= {max(ll[k]['P_F1'] for k in ('both','own_only','neighbours_only','neither')):.1e}",
        EXACT, cond.replace("OLMo-3-7B", "Llama-3.1-8B"), "audit_ablations.json models.llama-3.1-8b.variants", "exact",
        "no persistent-high outcome to lose; the OLMo effect is model-dependent")
    add("Llama control: max expected excess CRPS of any variant",
        f(max(ll[k]["expected_excess_crps_vs_both"] for k in ("own_only", "neighbours_only", "neither")), 6),
        EXACT, cond.replace("OLMo-3-7B", "Llama-3.1-8B"), "audit_ablations.json models.llama-3.1-8b.variants", "exact", "")
    if live:
        for p, c in live["cells"].items():
            if "per_run" not in c:
                add(f"E2 live cell {p}", "not run", "", cond, "observation_live.json", "", c.get("status", ""))
                continue
            add(f"E2 live {p}: F=1 runs (exact law in brackets)", f"{c['live_F1_count']}/50 ({f(c['exact_P_F1'])})",
                f"Clopper-Pearson [{f(c['live_P_F1_ci'][0])}, {f(c['live_P_F1_ci'][1])}]; per-run p = {f(c['per_run']['p_value'])} "
                f"(alpha {c['per_run']['alpha']}); per-decision max diff {c['per_decision']['max_abs_diff_p_H']}",
                cond, f"observation_live.json cells.{p}", "live (implementation check)",
                "checks the counterfactual implementation against the served LLM; adds no kernel information")
    if sm:
        for k, c in sm["cells"].items():
            add(f"Size x memory {k}: P(F=1)", f"{c['F1_count']}/{c['runs']}",
                f"Clopper-Pearson [{f(c['P_F1_ci'][0])}, {f(c['P_F1_ci'][1])}]", f"OLMo-3-7B, n = {c['n']} cycle, {c['memory']} history",
                f"size_memory.json cells.{k}", "live", c["source"])
        for k, c in sm["contrasts"].items():
            d = c["P_F1_a_minus_b"]
            add(f"Contrast {k} ({c['a']} - {c['b']}): P(F=1)", f(d["diff"]), f"Newcombe nominal 95% [{f(d['lo'])}, {f(d['hi'])}]",
                "OLMo-3-7B", f"size_memory.json contrasts.{k}", "live",
                "exploratory, nominal (unadjusted) interval; cells collected in different sessions, not jointly randomised")
        ms = sm.get("multiplicity_sensitivity")
        if ms:
            for fam in ("two_size_contrasts", "all_four_contrasts"):
                for k, d in ms[fam]["intervals"].items():
                    add(f"Multiplicity sensitivity ({fam}) {k}: P(F=1) difference", f(d["diff"]),
                        f"Bonferroni-adjusted Newcombe [{f(d['lo'])}, {f(d['hi'])}] (alpha {ms[fam]['alpha_per_contrast']:g} each)",
                        "OLMo-3-7B", f"size_memory.json multiplicity_sensitivity.{fam}.intervals.{k}", "live (post hoc sensitivity)",
                        "say 'size matters' only for last-round memory; under full history the adjusted interval includes 0")
    with open(F / "claims.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["claim", "value", "interval_or_precision", "condition", "provenance", "evidence_type", "allowed_wording"])
        w.writerows(rows)
    print(f"wrote {F / 'claims.csv'} ({len(rows)} claims)")


if __name__ == "__main__":
    main()
