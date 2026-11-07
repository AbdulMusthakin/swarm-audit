"""Gate G5: the calibration arms (n = 4, cycle, last round only) against their tables.

Per decision: every logged probability must equal the enumerated table entry for its
state, up to the numerical floor from G3, and prompts with the same state must be
byte-identical (one prompt sha256 per state, equal to the table's).

Per run (the teammates' test): y = high bids in rounds 16-20 (out of 20 decisions).
The gap between the empirical CDF of y over the runs and the exact CDF from the chain
is compared with the 99th percentile of the same gap over 20,000 simulated sets of
runs from that chain. Arm 4b is gated on the signal-aware chain; the actions-only
chain's result there is a finding, not a gate.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from . import stats
from .audit import AuditTable
from .prediction import ActionsChain, SignalChain, exact_law
from .swarm import RunData
from .topology import Graph

REPO_ROOT = Path(__file__).resolve().parent.parent


def per_decision(data: RunData, table: AuditTable, floor: float = 0.0) -> dict:
    """Max difference between logged and table probabilities, and prompt identity."""
    sig = table.kind == "signal"
    st = data.state_sig if sig else data.state
    max_pH, max_sig, n_dec = 0.0, 0.0, 0
    worst = None
    for idx in np.ndindex(st.shape):
        k = st[idx]
        n_dec += 1
        raw = table.entries[k].get("raw")   # enumerated tables keep the raw kernel output
        dH = abs(data.p_H[idx] - (raw["p_H"] if raw else table.kappa_H(k)))
        if dH > max_pH:
            max_pH, worst = dH, {"state": k, "index": list(map(int, idx))}
        if sig:
            if raw:
                conds = [np.asarray(raw["sig"]["H"]), np.asarray(raw["sig"]["L"])]
            else:
                J = table.kappa_joint(k)
                conds = [J[ai] / J[ai].sum() if J[ai].sum() > 0 else None for ai in range(2)]
            for ai, cond in enumerate(conds):
                if cond is not None:
                    max_sig = max(max_sig, float(np.abs(data.sig[idx][ai] - cond).max()))
    shas: dict[str, set] = defaultdict(set)
    for idx in np.ndindex(st.shape):
        shas[st[idx]].add(data.sha[idx])
    multi = {k: len(v) for k, v in shas.items() if len(v) > 1}
    table_mismatch = [k for k, v in shas.items()
                      if table.entries[k].get("prompt_sha256") is not None
                      and v != {table.entries[k]["prompt_sha256"]}]
    ok = max_pH <= floor and max_sig <= floor and not multi and not table_mismatch
    return {"pass": bool(ok), "decisions": n_dec, "floor": floor,
            "max_abs_diff_p_H": max_pH, "max_abs_diff_signal": max_sig, "worst": worst,
            "states_seen": len(shas), "states_with_several_prompts": multi,
            "states_whose_prompt_differs_from_table": table_mismatch}


ALPHA = 0.01


def per_run(data: RunData, chain, g: Graph, sims: int = 20000, alpha: float = ALPHA,
            seed: int = 5) -> dict:
    """The per-run test as a discrete Monte Carlo p-value (review C5): pass if
    p = (1 + #{simulated gap >= observed gap}) / (1 + sims) > alpha. Fixed before any
    live calibration data existed (7 Nov 2026)."""
    ex = exact_law(chain, g, data.T)
    y = stats.high_count(data.actions)
    G = stats.cdf_from_law(ex["y"])
    gap = stats.ks_gap(y, G)
    null = stats.ks_null_gaps(ex["y"], data.R, sims, np.random.default_rng(seed))
    pval = stats.mc_pvalue(gap, null)
    K = len(G) - 1
    return {"pass": bool(pval > alpha), "gap": gap, "p_value": pval, "alpha": alpha,
            "rule": "discrete Monte Carlo p-value (1 + #{null >= gap}) / (1 + sims) > alpha",
            "threshold_q99": float(np.quantile(null, 1 - alpha)), "sims": sims,
            "runs": data.R,
            "chain": getattr(chain, "name", type(chain).__name__),
            "exact_mean_y": float((np.arange(K + 1) * ex["y"]).sum()),
            "swarm_mean_y": float(y.mean()), "n_profiles": ex["n_profiles"]}


def check_arm(data: RunData, tables: dict[str, AuditTable], g: Graph, floor: float = 0.0,
              sims: int = 20000) -> dict:
    d = g.regular_degree
    if data.channel == "counts":
        t = tables["actions"]
        return {"per_decision": per_decision(data, t, floor),
                "per_run": per_run(data, ActionsChain(t.lookup(), d), g, sims)}
    t = tables["signal"]
    res = {"per_decision": per_decision(data, t, floor),
           "per_run": per_run(data, SignalChain(t.lookup(), d, t.own_signal), g, sims)}
    res["finding_actions_only_chain"] = per_run(
        data, ActionsChain(tables["actions"].lookup(), d), g, sims)
    return res


# ------------------------------------------------------ gate records (review C3)

GATES_DIR = REPO_ROOT / "results" / "gates"
CHANNELS = ("counts", "cheap_talk")


def gate_record(res: dict, data: RunData, tables: dict[str, AuditTable], floor: float) -> dict:
    """The G5 record of one calibration arm. Required checks: per decision and per run
    (on cheap talk, the signal-aware chain). The actions-only chain on cheap talk is a
    finding, not part of the gate."""
    import datetime as dt

    from .swarm import git_commit

    required = {"per_decision": res["per_decision"]["pass"], "per_run": res["per_run"]["pass"]}
    return {"gate": "G5", "channel": data.channel, "pass": all(required.values()),
            "required": required, "floor": floor, "runs": data.R,
            "kernel_fingerprint": data.meta.get("kernel_fingerprint"),
            "run_fingerprint": data.meta.get("fingerprint"),
            "table_fingerprints": {k: t.meta.get("provenance", {}).get("fingerprint")
                                   for k, t in tables.items() if t is not None},
            "finding_not_gate": ("actions-only chain on cheap talk"
                                 if "finding_actions_only_chain" in res else None),
            "folder": data.meta.get("folder"), "git_commit": git_commit(),
            "time": dt.datetime.now(dt.timezone.utc).isoformat(), "result": res}


def write_gate(rec: dict, folder: Path | None = None) -> Path:
    """Write results/gates/G5_<channel>.json; every record is also appended to
    history.jsonl, so a superseded record is never lost."""
    import json

    folder = Path(folder or GATES_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps(rec, default=_jsonable)
    with open(folder / "history.jsonl", "a") as fh:
        fh.write(text + "\n")
    path = folder / f"G5_{rec['channel']}.json"
    path.write_text(json.dumps(json.loads(text), indent=1) + "\n")
    return path


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return str(x)


def _load(path: Path) -> dict | None:
    import json

    return json.loads(path.read_text()) if path.exists() else None


def write_exception(channel: str, gate: dict, approved_by: str, date: str, decision: str,
                    evidence: list[str], folder: Path | None = None) -> Path:
    """Record an approved exception to a failed G5 (Abdul's decision), bound to the exact
    runs and kernel of the failed record. The failed record itself is left unchanged."""
    import json

    folder = Path(folder or GATES_DIR)
    if gate.get("pass"):
        raise ValueError("G5 passed: no exception needed")
    if not gate["required"].get("per_decision"):
        raise ValueError("a per-decision failure cannot be excepted")
    rec = {"gate": "G5", "channel": channel, "exception": True, "approved_by": approved_by,
           "date": date, "decision": decision, "evidence": evidence,
           "applies_to": {"kernel_fingerprint": gate["kernel_fingerprint"],
                          "run_fingerprint": gate["run_fingerprint"], "runs": gate["runs"]},
           "requires": "a passing confirmation (G5_<channel>_confirmation.json) before the "
                       f"{channel} core launches"}
    path = folder / f"G5_{channel}_exception.json"
    path.write_text(json.dumps(rec, indent=1) + "\n")
    with open(folder / "history.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return path


def _channel_ok(ch: str, kernel_fingerprint: str, folder: Path, problems: list) -> None:
    rec = _load(folder / f"G5_{ch}.json")
    if rec is None:
        problems.append(f"{ch}: no G5 record")
        return
    if rec.get("kernel_fingerprint") != kernel_fingerprint:
        problems.append(f"{ch}: G5 ran under a different kernel (stale)")
        return
    if rec.get("pass"):
        return
    exc = _load(folder / f"G5_{ch}_exception.json")
    conf = _load(folder / f"G5_{ch}_confirmation.json")
    if exc is None or exc["applies_to"]["kernel_fingerprint"] != kernel_fingerprint \
            or exc["applies_to"]["run_fingerprint"] != rec.get("run_fingerprint"):
        problems.append(f"{ch}: G5 failed ({rec.get('required')}) and no matching approved "
                        "exception")
    elif conf is None:
        problems.append(f"{ch}: G5 excepted by {exc['approved_by']}, but the confirmation "
                        "has not run")
    elif not conf.get("pass") or conf.get("kernel_fingerprint") != kernel_fingerprint:
        problems.append(f"{ch}: the confirmation failed or is stale: paused (Abdul, rule 3)")


def require_g5(kernel_fingerprint: str, arm_channel: str = "cheap_talk",
               folder: Path | None = None) -> None:
    """Block a core launch unless G5 allows it (review C3; Abdul's decision, 7 Nov 2026).

    counts core: counts G5 passed under the same kernel.
    cheap-talk core: counts G5 passed, and cheap-talk G5 passed or was excepted by an
    approved exception bound to the same runs and kernel together with a passing
    confirmation. A failed confirmation pauses the cheap-talk core only."""
    folder = Path(folder or GATES_DIR)
    problems: list[str] = []
    _channel_ok("counts", kernel_fingerprint, folder, problems)
    if arm_channel == "cheap_talk":
        _channel_ok("cheap_talk", kernel_fingerprint, folder, problems)
    if problems:
        raise SystemExit("Stopping before the core launch (gate G5): " + "; ".join(problems))
