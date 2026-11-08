#!/usr/bin/env python3
"""Independent check of the key follow-up numbers (plan.json; deliverable 7).

Written separately from swarm_audit/followup.py and scripts/extra_followup_exact.py: it
reads the raw JSON table by state-key strings, forms each variant's kernel by explicit
weighted sums over keys (weights from math.comb), builds transition rows agent by agent
with dictionaries, and computes P(F=1), P(F=0) and E[F] by forward recursion on
(profile, running count). It also checks the expected-CRPS identity by brute force.

    python scripts/extra_followup_check.py

Output: results/extra/followup/independent_check.json (and a nonzero exit on mismatch).
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "extra" / "followup"
SIG = ("high", "low", "none")
N, T, W = 4, 20, 5
NB = {i: ((i - 1) % N, (i + 1) % N) for i in range(N)}      # the 4-cycle
TOL = 1e-10


def weight_c(c):
    return math.comb(2, c[0]) * math.comb(2 - c[0], c[1]) / 9.0


COMPS = [(h, l, 2 - h - l) for h in range(3) for l in range(3 - h)]


def kernel(table, variant):
    """state (a, k, c, o) -> 6-tuple, by explicit sums over state keys."""
    def raw(a, k, c, o):
        kap = table["entries"][f"{a}-{k}-{c[0]}-{c[1]}-{c[2]}-{o}"]["kappa"]
        return [kap[0][0], kap[0][1], kap[0][2], kap[1][0], kap[1][1], kap[1][2]]
    out = {}
    for a in "HL":
        for k in range(3):
            for c in COMPS:
                for o in SIG:
                    if variant == "both":
                        cs, os_ = [(c, 1.0)], [(o, 1.0)]
                    elif variant == "own_only":
                        cs, os_ = [(cc, weight_c(cc)) for cc in COMPS], [(o, 1.0)]
                    elif variant == "neighbours_only":
                        cs, os_ = [(c, 1.0)], [(oo, 1 / 3) for oo in SIG]
                    else:
                        cs, os_ = [(cc, weight_c(cc)) for cc in COMPS], [(oo, 1 / 3) for oo in SIG]
                    vec = [0.0] * 6
                    for cc, wc in cs:
                        for oo, wo in os_:
                            r = raw(a, k, cc, oo)
                            vec = [v + wc * wo * x for v, x in zip(vec, r)]
                    out[(a, k, c, o)] = vec
    return out


def observe(c, o, process):
    if process in ("own_null", "both_null"):
        o = "none"
    if process in ("neighbour_null", "both_null"):
        c = (0, 0, 2)
    return c, o


def law(table, variant="both", process="unmodified"):
    K = kernel(table, variant)
    profs = list(itertools.product(range(6), repeat=N))
    index = {p: i for i, p in enumerate(profs)}
    st = table["entries"]["START"]["kappa"]
    start = [st[0][0], st[0][1], st[0][2], st[1][0], st[1][1], st[1][2]]
    P = np.zeros((len(profs), len(profs)))
    for p in profs:
        vecs = []
        for i in range(N):
            a = "HL"[p[i] // 3]
            o = SIG[p[i] % 3]
            k = sum(1 for j in NB[i] if p[j] // 3 == 0)
            ms = [SIG[p[j] % 3] for j in NB[i]]
            c = (ms.count("high"), ms.count("low"), ms.count("none"))
            c, o = observe(c, o, process)
            vecs.append(K[(a, k, c, o)])
        for q in profs:
            P[index[p], index[q]] = vecs[0][q[0]] * vecs[1][q[1]] * vecs[2][q[2]] * vecs[3][q[3]]
    v = np.array([start[p[0]] * start[p[1]] * start[p[2]] * start[p[3]] for p in profs])
    highs = np.array([sum(1 for x in p if x // 3 == 0) for p in profs])
    for _ in range(T - W):                       # to round 16
        v = v @ P
    # joint (profile, count) as a dict of count -> vector
    joint = {}
    for h in range(N + 1):
        joint[h] = np.where(highs == h, v, 0.0)
    for _ in range(W - 1):
        new = {}
        for cnt, vec in joint.items():
            nxt = vec @ P
            for h in range(N + 1):
                part = np.where(highs == h, nxt, 0.0)
                new[cnt + h] = new.get(cnt + h, 0) + part
        joint = new
    y = np.zeros(W * N + 1)
    for cnt, vec in joint.items():
        y[cnt] += vec.sum()
    return y


def main() -> int:
    abl = json.loads((OUT / "audit_ablations.json").read_text())
    obs = json.loads((OUT / "observation_interventions.json").read_text())
    tables = {"olmo-3-7b": REPO / "results/extra/audit/olmo-3-7b/cheap_talk_d2_enumerated_signal.json",
              "llama-3.1-8b": REPO / "results/audit/llama-3.1-8b/cheap_talk_d2_enumerated_signal.json"}
    res, bad = {}, []
    for m, path in tables.items():
        table = json.loads(path.read_text())
        res[m] = {}
        ys = {}
        for v in ("both", "own_only", "neighbours_only", "neither"):
            y = law(table, variant=v)
            ys[v] = y
            ref = abl["models"][m]["variants"][v]
            got = {"P_F1": y[-1], "P_F0": y[0], "E_F": float((np.arange(21) * y).sum() / 20)}
            diff = max(abs(got[k] - ref[k]) for k in got)
            res[m][f"E1_{v}"] = {**got, "max_abs_diff_vs_main": diff}
            if diff > TOL:
                bad.append(f"{m} E1 {v} {diff}")
        for p in ("unmodified", "own_null", "neighbour_null", "both_null"):
            y = law(table, process=p)
            ref = obs["models"][m]["processes"][p]
            got = {"P_F1": y[-1], "P_F0": y[0], "E_F": float((np.arange(21) * y).sum() / 20)}
            diff = max(abs(got[k] - ref[k]) for k in got)
            res[m][f"E2_{p}"] = {**got, "max_abs_diff_vs_main": diff}
            if diff > TOL:
                bad.append(f"{m} E2 {p} {diff}")
        # CRPS identity by brute force: sum_y G(y) [CRPS(Q, y) - CRPS(G, y)]
        G, Q = np.cumsum(ys["both"]), np.cumsum(ys["neither"])
        brute = 0.0
        for yy in range(21):
            ind = np.array([1.0 if yy <= k else 0.0 for k in range(21)])
            brute += ys["both"][yy] * (((Q - ind) ** 2).sum() - ((G - ind) ** 2).sum()) / 20
        ref = abl["models"][m]["variants"]["neither"]["expected_excess_crps_vs_both"]
        res[m]["crps_identity"] = {"brute_force": brute, "identity": float(((Q - G) ** 2).sum() / 20),
                                   "main": ref}
        if abs(brute - ref) > TOL:
            bad.append(f"{m} crps identity {brute} vs {ref}")
    res["tolerance"] = TOL
    res["pass"] = not bad
    res["failures"] = bad
    (OUT / "independent_check.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({m: {k: round(v["P_F1"], 8) for k, v in r.items() if isinstance(v, dict) and "P_F1" in v}
                      for m, r in res.items() if isinstance(r, dict)}, indent=1))
    print("PASS" if not bad else f"FAIL: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
