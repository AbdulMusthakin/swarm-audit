"""Signal responsiveness on the final cheap-talk prompt (Abdul, 7 Nov 2026; run in G4).

The screening metric (results/screening/summary.md): with the action counts fixed, the
spread max - min of P(H) over four neighbour-message variants, for each own last bid and
history (length 1, and 3 random fillers at length 5). The "signal" bar is the mean of
the length-1 values (0.05).

- d = 4: the screening grid itself (2 high and 2 low; all "high", all "low", all silent,
  2 + 2), so the number compares directly with screening's 0.051 for Llama 3.1 8B.
- d = 2: the same construction on the cycle's degree (1 high and 1 low; both "high",
  both "low", both silent, 1 + 1), which is what the experiment uses.
"""

from __future__ import annotations

import copy

import numpy as np

from .policies import Backend, Query
from .screening import build_grid, load_config

D2_GRID = {"d": 2, "n_high": 1, "own_last": ["H", "L"], "own_last_signal": "none",
           "signal_variants": {"all_high": [2, 0, 0], "all_low": [0, 2, 0],
                               "all_silent": [0, 0, 2], "one_one": [1, 1, 0]}}


def grids(seed: int | None = None) -> dict[int, list]:
    cfg = load_config()
    if seed is not None:
        cfg["seed"] = seed
    out = {4: [s for s in build_grid(cfg) if s.step == "cheap_talk"]}
    c2 = copy.deepcopy(cfg)
    c2["grid"]["cheap_talk"] = D2_GRID
    out[2] = [s for s in build_grid(c2) if s.step == "cheap_talk"]
    return out


def measure(backend: Backend) -> dict:
    res = {}
    for d, states in grids().items():
        outs = backend.query([Query("cheap_talk", s.history, s.d) for s in states])
        groups: dict[tuple, dict[str, float]] = {}
        rows = []
        for s, o in zip(states, outs):
            groups.setdefault((s.own_last, s.length, s.filler), {})[s.signal_variant] = o.p_H
            rows.append({"prompt_id": s.prompt_id, "p_H": o.p_H, "p_valid": o.p_valid,
                         "sig_after_H": o.sig["H"], "sig_after_L": o.sig["L"]})
        spreads = [{"own_last": k[0], "length": k[1], "filler": k[2],
                    "spread": max(v.values()) - min(v.values()), "p_H": v}
                   for k, v in sorted(groups.items(), key=str)]
        l1 = [g["spread"] for g in spreads if g["length"] == 1]
        l5 = [g["spread"] for g in spreads if g["length"] > 1]
        res[f"d{d}"] = {"signal_responsiveness_length1_mean": float(np.mean(l1)),
                        "signal_responsiveness_length5_mean": float(np.mean(l5)),
                        "by_own_last_length1": {g["own_last"]: g["spread"] for g in spreads
                                                if g["length"] == 1},
                        "groups": spreads, "prompts": rows, "bar": 0.05}
    res["screening_reference"] = {"d4_length1_mean": 0.051, "prompt": "draft of 6 Nov (messages "
                                  "'I'll bid high' without a stated round)"}
    return res
