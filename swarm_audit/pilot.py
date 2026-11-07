"""Gate G6: Abdul's pilot rule (configs/pilot.yaml), applied mechanically.

Fixed on 7 Nov 2026 before the pilot: if at least `min_runs` of the pilot runs reach
all-low consensus by round `by_round`, apply `change` to the payoffs (uLH = 11) and
freeze it for both channels; otherwise change nothing.

The primary reading of "all-low consensus by round 5" uses the paper's consensus time
tau (first round of three consecutive all-low rounds): tau <= 5. Two other readings are
evaluated too; if any would give a different decision, the rule is ambiguous for this
pilot and Abdul decides.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import stats
from .game import PayoffParams

REPO_ROOT = Path(__file__).resolve().parent.parent
PAYOFF_CONFIGS = ("core.yaml", "calibration.yaml", "secondary.yaml", "harness.yaml")


def decide(A: np.ndarray, rule: dict) -> dict:
    """A: pilot actions (R, T, n), 1 = H."""
    side, by, k = rule["consensus_side"], int(rule["by_round"]), int(rule["min_runs"])
    v = 1.0 if side == "H" else 0.0
    X = stats.frac_high(A)
    con = stats.consensus(A)
    readings = {
        "tau_le_by_round": (con.side == side) & (np.nan_to_num(con.tau, nan=np.inf) <= by),
        "unanimous_in_some_round_le_by_round": (X[:, :by] == v).any(axis=1),
        "three_rounds_completed_by_round": (con.side == side)
        & (np.nan_to_num(con.tau, nan=np.inf) <= by - 2),
    }
    counts = {name: int(x.sum()) for name, x in readings.items()}
    trig = {name: c >= k for name, c in counts.items()}
    primary = trig["tau_le_by_round"]
    return {"runs": int(A.shape[0]), "runs_meeting_condition": counts, "triggered": trig,
            "apply_change": bool(primary), "ambiguous": len(set(trig.values())) > 1,
            "tau": [None if np.isnan(t) else int(t) for t in con.tau],
            "side": con.side.tolist(), "X_t_mean": X.mean(0).tolist()}


def new_payoffs(payoffs: dict, change: dict) -> dict:
    out = {**payoffs, **change}
    p = PayoffParams(**{k: float(v) for k, v in out.items()})
    if not p.is_prisoners_dilemma():
        raise ValueError(f"the changed payoffs {out} break the prisoner's-dilemma ordering")
    return out


def freeze_payoffs(payoffs: dict, configs=PAYOFF_CONFIGS,
                   folder: Path = REPO_ROOT / "configs") -> list[str]:
    """Rewrite the `payoffs:` line of every config (both channels use them)."""
    line = "payoffs: {" + ", ".join(f"{k}: {v:g}" for k, v in payoffs.items()) + "}"
    changed = []
    for name in configs:
        p = Path(folder) / name
        s = p.read_text()
        s2, n = re.subn(r"^payoffs: \{[^}]*\}", line, s, count=1, flags=re.M)
        if n != 1:
            raise ValueError(f"no payoffs line in {p}")
        p.write_text(s2)
        changed.append(name)
    return changed
