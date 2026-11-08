#!/usr/bin/env python3
"""E1 (audit information variants) and E2 (observation interventions), exact at n = 4.

Frozen specification: results/extra/followup/plan.json (docs/overnight-extra-plan.md).
Reads the enumerated last-1 tables and the logged runs only; no model query.

    python scripts/extra_followup_exact.py

Outputs (results/extra/followup/): audit_ablations.json, observation_interventions.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from swarm_audit import build, followup as fu, stats  # noqa: E402
from swarm_audit.audit import AuditTable  # noqa: E402
from swarm_audit.game import SIGNALS  # noqa: E402
from swarm_audit.prediction import ActionsChain, SignalChain, exact_law  # noqa: E402
from swarm_audit.states import sig_compositions  # noqa: E402
from swarm_audit.swarm import load_runs  # noqa: E402
from swarm_audit.topology import make  # noqa: E402

OUT = REPO / "results" / "extra" / "followup"
T, N, D, B, SEED = 20, 4, 2, 2000, 20261108
TOL = 1e-12
MODELS = {
    "olmo-3-7b": {"tables": REPO / "results/extra/audit/olmo-3-7b", "role": "primary",
                  "runs": REPO / "results/extra/runs/olmo_calibration_cheap_talk/cheap_talk_n4_cycle_last1",
                  "run_range": None},
    "llama-3.1-8b": {"tables": REPO / "results/audit/llama-3.1-8b", "role": "control (fixed comparison)",
                     "runs": REPO / "results/runs/calibration_cheap_talk/cheap_talk_n4_cycle_last1",
                     "run_range": (0, 49)},
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def paired(a, b, seed: int) -> dict:
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return {"mean": float(d.mean()), "lo": float(np.quantile(bs, 0.025)),
            "hi": float(np.quantile(bs, 0.975)), "n": len(d), "B": B,
            "evidence": "live runs, whole-run bootstrap"}


def lookup_from_array(Jv: np.ndarray, start: np.ndarray) -> dict:
    """A table lookup (state key -> 2 x 3) from a variant array, for prediction.SignalChain."""
    from swarm_audit.states import key
    lk = {key(("START",)): start.reshape(2, 3)}
    for ai, a in enumerate("HL"):
        for k in range(D + 1):
            for (sh, sl, sn) in sig_compositions(D):
                for oi, o in enumerate(SIGNALS):
                    lk[key((a, k, sh, sl, sn, o))] = Jv[ai, k, sh, sl, oi].reshape(2, 3)
    return lk


def independent_row(table: dict, x: tuple, g, transform=None) -> np.ndarray:
    """Separately written check: P(x -> y) for every y, from the raw JSON entries and the
    graph's neighbour lists, one agent at a time (no kernel arrays)."""
    per = []
    for i, nb in enumerate(g.in_neighbours):
        a_i, m_i = divmod(x[i], 3)
        k = sum(x[j] // 3 == 0 for j in nb)
        ms = [SIGNALS[x[j] % 3] for j in nb]
        c = (ms.count("high"), ms.count("low"), ms.count("none"))
        o = SIGNALS[m_i]
        if transform == "own_null":
            o = "none"
        elif transform == "neighbour_null":
            c = (0, 0, len(nb))
        elif transform == "both_null":
            o, c = "none", (0, 0, len(nb))
        k_ = f"{'HL'[a_i]}-{k}-{c[0]}-{c[1]}-{c[2]}-{o}"
        per.append(np.asarray(table["entries"][k_]["kappa"], float).reshape(6))
    out = np.ones(6 ** len(x))
    for idx, y in enumerate(fu.profiles(len(x))):
        out[idx] = np.prod([per[i][y[i]] for i in range(len(x))])
    return out


def diagnostics(P_both: np.ndarray, v: np.ndarray, Ps: dict, AH: np.ndarray, Pr: np.ndarray,
                J: np.ndarray, aw: dict) -> dict:
    """Occupancy, own-message composition, entry, oracle exit and survival in all-H."""
    occ = v[:, AH].sum(1)                                            # (T,)
    m = Pr % 3
    comp = np.stack([(v[:, AH] @ (m[AH] == j).mean(1)) for j in range(3)], 1)   # (T, 3)
    comp_rel = comp / np.where(occ[:, None] > 0, occ[:, None], np.nan)
    entry = np.zeros(T)
    entry[0] = occ[0]
    entry_prof = np.zeros((T, AH.sum()))
    entry_prof[0] = v[0, AH]
    for t in range(1, T):
        flow = v[t - 1, ~AH] @ P_both[np.ix_(~AH, AH)]
        entry_prof[t] = flow
        entry[t] = flow.sum()
    pi_entry = entry_prof.sum(0) / entry_prof.sum()
    w_r = v[1:, AH].sum(0) / v[1:, AH].sum()                         # occupancy over rounds 2-20
    pooled_comp = [float(w_r @ (m[AH] == j).mean(1)) for j in range(3)]
    exit_by_variant, survival = {}, {}
    for name, P in Ps.items():
        stay = P[np.ix_(AH, AH)].sum(1)
        exit_by_variant[name] = {
            "per_round": [float(v[t, AH] @ (1 - stay) / occ[t]) if occ[t] > 0 else None for t in range(T)],
            "occupancy_weighted_rounds_2_20": float(w_r @ (1 - stay))}
        Q = P[np.ix_(AH, AH)]
        s, out = pi_entry.copy(), []
        for _ in range(T):
            out.append(float(s.sum()))
            s = s @ Q
        survival[name] = out
    # occupancy-weighted distribution of the agent-level (c, o) slice in all-H profiles
    a_idx = Pr // 3
    slice_w = {}
    for i in range(N):
        nb = [(i - 1) % N, (i + 1) % N]
        sh = (m[AH][:, nb] == 0).sum(1)
        sl = (m[AH][:, nb] == 1).sum(1)
        for r, (h_, l_, o_) in enumerate(zip(sh, sl, m[AH][:, i])):
            kk = (int(h_), int(l_), int(o_))
            slice_w[kk] = slice_w.get(kk, 0.0) + w_r[r] / N
    assert np.all(a_idx[AH] == 0)
    p_h_slice = sum(w * float(J[0, 2, h_, l_, o_][:3].sum()) for (h_, l_, o_), w in slice_w.items())
    p_h_ptest = sum(aw[c] / 3 * float(J[0, 2, c[0], c[1], o_][:3].sum())
                    for c in aw for o_ in range(3))
    return {
        "occupancy_all_H": occ.tolist(),
        "own_message_share_within_all_H_per_round": {SIGNALS[j]: comp_rel[:, j].tolist() for j in range(3)},
        "own_message_share_within_all_H_rounds_2_20": dict(zip(SIGNALS, pooled_comp)),
        "audit_weight_on_each_own_message": 1 / 3,
        "entry_probability_per_round": entry.tolist(),
        "exit_one_step_oracle": exit_by_variant,
        "survival_from_both_entry_distribution": survival,
        "slice_H2_occupancy_weighted": {
            "distribution_over_(s_high,s_low,own)": {f"{a}-{b}-{SIGNALS[c]}": w for (a, b, c), w in
                                                     sorted(slice_w.items(), key=lambda z: -z[1])},
            "P_H_occupancy_weighted": p_h_slice, "P_H_ptest_weighted": p_h_ptest,
            "label": "ORACLE diagnostic: weights are the full chain's own occupancy, not an audit"},
        "labels": {"exit": "occupancy-weighted oracle diagnostic (both chain's all-H distribution at t)",
                   "survival": "finite-horizon persistence in all-H from the both chain's entry distribution; not absorption"},
    }


def run_model(name: str, spec: dict) -> tuple[dict, dict]:
    g = make("cycle", N)
    tpath = spec["tables"] / "cheap_talk_d2_enumerated_signal.json"
    apath = spec["tables"] / "cheap_talk_d2_enumerated_actions.json"
    raw = json.loads(tpath.read_text())
    sig = AuditTable.load(tpath)
    act = AuditTable.load(apath)
    J, start = fu.kernel_array(sig.lookup(), D)
    Pr = fu.profiles(N)
    AH = fu.all_high(N)
    v1 = fu.start_law(start, N)
    aw = fu.comp_weights(D, "primary")

    # ---- E1 variants
    arrays = {v: fu.variant_array(J, D, v) for v in fu.VARIANTS}
    arrays["own_only_equal_w"] = fu.variant_array(J, D, "own_only", "equal")
    arrays["neither_equal_w"] = fu.variant_array(J, D, "neither", "equal")
    Ps, laws, checks = {}, {}, {}
    for v, Jv in arrays.items():
        P = fu.transition_matrix(fu.agent_vectors(Jv, g))
        Ps[v] = P
        checks[v] = {"min_entry": float(P.min()), "max_row_sum_error": float(np.abs(P.sum(1) - 1).max()),
                     "kernel_min": float(np.nanmin(Jv))}
        laws[v] = fu.propagate(P, v1, N, T)
        ex = exact_law(SignalChain(lookup_from_array(Jv, start), D, True), g, T)   # prediction.py
        checks[v]["exact_law_agreement_y"] = float(np.abs(ex["y"] - laws[v]["y"]).max())
    # reproduction of the existing chains
    ex_full = exact_law(SignalChain(sig.lookup(), D, True), g, T)
    ex_act = exact_law(ActionsChain(act.lookup(), D), g, T)
    act_eq = AuditTable.load(spec["tables"] / "cheap_talk_d2_enumerated_actions_equal.json")
    ex_act_eq = exact_law(ActionsChain(act_eq.lookup(), D), g, T)
    checks["both_reproduces_signal_aware_y"] = float(np.abs(ex_full["y"] - laws["both"]["y"]).max())
    checks["neither_action_law_equals_actions_only_y"] = float(np.abs(ex_act["y"] - laws["neither"]["y"]).max())
    checks["neither_action_law_equals_actions_only_X"] = float(np.abs(
        ex_act["X"] @ np.arange(N + 1) / N - laws["neither"]["EX"]).max())
    checks["neither_equal_w_equals_actions_only_equal_w_y"] = float(np.abs(
        ex_act_eq["y"] - laws["neither_equal_w"]["y"]).max())
    checks["start_identical"] = "by construction: every variant and process uses the one START vector of the table"
    rng = np.random.default_rng(SEED)
    rows = rng.choice(len(Pr), 12, replace=False)
    checks["independent_rows_max_diff"] = float(max(
        np.abs(independent_row(raw, tuple(Pr[x]), g) - Ps["both"][x]).max() for x in rows))
    # independent averaging check for the neither variant (H-2 marginal = derived actions table)
    checks["neither_H2_equals_derived_actions_kappa"] = float(abs(
        arrays["neither"][0, 2, 0, 0, 0][:3].sum() - act.kappa_H("H-2")))
    for k in ("H-0", "H-1", "H-2", "L-0", "L-1", "L-2"):
        ai, kk = "HL".index(k[0]), int(k[2])
        checks[f"neither_{k}_minus_derived"] = float(arrays["neither"][ai, kk, 0, 0, 0][:3].sum() - act.kappa_H(k))
    G_both = np.cumsum(laws["both"]["y"])
    variants = {}
    data = load_runs(spec["runs"])
    if spec["run_range"]:
        lo, hi = spec["run_range"]
        data = data.subset(np.nonzero((data.runs >= lo) & (data.runs <= hi))[0])
    y_obs = stats.high_count(data.actions)
    crps_runs = {v: stats.crps(np.cumsum(laws[v]["y"]), y_obs) for v in laws}
    for v in laws:
        Q = np.cumsum(laws[v]["y"])
        s = fu.law_summary(laws[v]["y"], laws[v]["EX"], N)
        s["expected_excess_crps_vs_both"] = fu.expected_excess_crps(Q, G_both)
        s["dP_F1_vs_both"] = s["P_F1"] - float(laws["both"]["y"][-1])
        s["recovers_reference"] = bool(abs(s["dP_F1_vs_both"]) <= 0.01 and s["expected_excess_crps_vs_both"] <= 0.001)
        s["live_crps"] = stats.bootstrap_mean(crps_runs[v], B, np.random.default_rng(SEED))
        s["live_crps_minus_both"] = paired(crps_runs[v], crps_runs["both"], SEED + 1)
        s["evidence"] = "exact chain law (cached kernel); live_* from logged runs"
        variants[v] = s
    retained = {"both": 2, "own_only": 1, "neighbours_only": 1, "neither": 0}
    ok = [v for v in fu.VARIANTS if variants[v]["recovers_reference"]]
    smallest = min(ok, key=lambda v: retained[v]) if ok else None
    swarm_F1 = float((y_obs == 5 * N).mean())
    grid = {f"{'HL'[ai]}-{k}": {f"{sh}-{sl}-{D - sh - sl}": {SIGNALS[o]: float(J[ai, k, sh, sl, o][:3].sum())
                                                            for o in range(3)}
                                for (sh, sl, _) in sig_compositions(D)}
            for ai in range(2) for k in range(D + 1)}
    diag = diagnostics(Ps["both"], laws["both"]["v"], Ps, AH, Pr, J, aw)
    # where the high outcome comes from: the round-1 profile (exact and live)
    v_ah = np.where(AH, v1, 0.0)
    L_ah = fu.propagate(Ps["both"], v_ah / v_ah.sum(), N, T)
    L_not = fu.propagate(Ps["both"], np.where(AH, 0.0, v1) / (1 - v_ah.sum()), N, T)
    r1_ah = data.actions[:, 0, :].all(1)
    F1 = y_obs == 5 * N
    allh = data.actions[:, 1:, :].all(2)                              # (R, T-1): rounds 2-20
    live_msgs = data.signals[:, 1:, :][allh]                          # messages sent in all-H rounds
    diag["round1_all_H"] = {
        "exact_P_round1_all_H": float(v_ah.sum()),
        "exact_P_F1_given_round1_all_H": float(L_ah["y"][-1]),
        "exact_P_F1_given_round1_not_all_H": float(L_not["y"][-1]),
        "exact_share_of_F1_from_round1_all_H": float(v_ah.sum() * L_ah["y"][-1] / laws["both"]["y"][-1]),
        "live_runs_round1_all_H": int(r1_ah.sum()),
        "live_F1_among_round1_all_H": int((F1 & r1_ah).sum()),
        "live_F1_among_round1_not_all_H": int((F1 & ~r1_ah).sum()),
        "evidence": "exact (cached kernel) and live counts"}
    diag["live_own_message_share_in_all_H_rounds_2_20"] = {
        "agent_rounds": int(live_msgs.size),
        **{SIGNALS[j]: (float((live_msgs == j).mean()) if live_msgs.size else None) for j in range(3)},
        "evidence": "live runs (messages sent in rounds 2-20 in which all agents bid H)"}
    e1 = {
        "model": name, "role": spec["role"],
        "table": {"path": str(tpath.relative_to(REPO)), "sha256": sha(tpath)},
        "runs": {"path": str(spec["runs"].relative_to(REPO)), "n_runs": int(data.R),
                 "run_range": spec["run_range"], "live_P_F1": swarm_F1,
                 "live_P_F1_count": int((y_obs == 5 * N).sum()),
                 "live_mean_F": float((y_obs / (5 * N)).mean())},
        "variants": variants,
        "smallest_recovering_variant": smallest,
        "recovery_tolerance": "|dP(F=1)| <= 0.01 and expected excess CRPS <= 0.001 (plan.json)",
        "checks": checks,
        "kernel_grid_P_H": grid,
        "mechanism": diag,
    }

    # ---- E2 observation interventions
    procs, pchecks = {}, {}
    Pp = {}
    for p in fu.PROCESSES:
        P = fu.transition_matrix(fu.agent_vectors(J, g, fu.observation(p)))
        Pp[p] = P
        pchecks[p] = {"min_entry": float(P.min()), "max_row_sum_error": float(np.abs(P.sum(1) - 1).max())}
        pchecks[p]["independent_rows_max_diff"] = float(max(
            np.abs(independent_row(raw, tuple(Pr[x]), g, None if p == "unmodified" else p) - P[x]).max()
            for x in rows))
        L = fu.propagate(P, v1, N, T)
        s = fu.law_summary(L["y"], L["EX"], N)
        # survival in all-H from this process's own entry distribution
        v = L["v"]
        ent = np.zeros((T, AH.sum()))
        ent[0] = v[0, AH]
        for t in range(1, T):
            ent[t] = v[t - 1, ~AH] @ P[np.ix_(~AH, AH)]
        pi = ent.sum(0) / ent.sum()
        Q = P[np.ix_(AH, AH)]
        sv, out = pi.copy(), []
        for _ in range(T):
            out.append(float(sv.sum()))
            sv = sv @ Q
        s["survival_own_entry"] = out
        s["occupancy_all_H"] = v[:, AH].sum(1).tolist()
        s["evidence"] = "exact, model-based counterfactual on the cached kernel (not live)"
        procs[p] = s
    pchecks["unmodified_equals_both_variant"] = float(np.abs(Pp["unmodified"] - Ps["both"]).max())
    pF = {p: procs[p]["P_F1"] for p in procs}
    eF = {p: procs[p]["E_F"] for p in procs}
    inter = {"P_F1": pF["unmodified"] - pF["neighbour_null"] - pF["own_null"] + pF["both_null"],
             "E_F": eF["unmodified"] - eF["neighbour_null"] - eF["own_null"] + eF["both_null"],
             "definition": "both channels present - own feedback only (neighbour_null) - neighbour "
                           "feedback only (own_null) + neither (both_null)"}
    # uniform-replacement sensitivity: identical law to the E1 averaged audits
    uni = {}
    for label, own, nbr, e1v in (("own_randomised", True, False, "neighbours_only"),
                                 ("neighbours_randomised", False, True, "own_only"),
                                 ("both_randomised", True, True, "neither")):
        P = fu.transition_matrix(fu.agent_vectors(J, g, fu.uniform_observation(D, own, nbr)))
        L = fu.propagate(P, v1, N, T)
        uni[label] = {**{k: v for k, v in fu.law_summary(L["y"], L["EX"], N).items() if k != "y_law"},
                      "equals_E1_variant": e1v,
                      "max_abs_diff_transition_vs_E1": float(np.abs(P - Ps[e1v]).max())}
    e2 = {"model": name, "processes": procs, "interaction": inter, "checks": pchecks,
          "uniform_replacement_sensitivity": uni,
          "labels": {"what": "model-based counterfactual observation interventions on the validated "
                             "cached kernel; not new live measurements",
                     "none_semantics": "'none' is meaningful content in the prompt (\"sent no message\"), "
                                       "not a neutral token",
                     "precision": f"exact; numerical tolerance {TOL}"}}
    return e1, e2


def main() -> None:
    e1s, e2s = {}, {}
    for name, spec in MODELS.items():
        e1s[name], e2s[name] = run_model(name, spec)
        v = e1s[name]["variants"]
        print(name, {k: round(v[k]["P_F1"], 6) for k in v}, "smallest:", e1s[name]["smallest_recovering_variant"])
        print("  E2", {p: round(e2s[name]["processes"][p]["P_F1"], 6) for p in fu.PROCESSES},
              "interaction", round(e2s[name]["interaction"]["P_F1"], 6))
    meta = {"plan": "results/extra/followup/plan.json", "script": "scripts/extra_followup_exact.py",
            "T": T, "n": N, "graph": "cycle", "event": "F = 1: all agents high in rounds 16-20"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_ablations.json").write_text(json.dumps({**meta, "models": e1s}, indent=1) + "\n")
    (OUT / "observation_interventions.json").write_text(json.dumps({**meta, "models": e2s}, indent=1) + "\n")


if __name__ == "__main__":
    main()
