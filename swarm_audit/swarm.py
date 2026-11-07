"""Swarm runs: all R runs of a condition in lockstep, one backend call per round.

For agent i in round t of run r (hand-off section 3):
1. Its history h_i^t (records of rounds 1..t-1) becomes a `Query`; every agent of every
   unfinished run goes into one `backend.query` call per round.
2. The action is sampled in Python: a = H if u < p_H, with u from the run's own stream
   Generator(PCG64(SeedSequence([base_seed, arm_id, condition_id, run_index]))).
   The agents draw in index order; on cheap talk each agent draws its action and then
   its signal (from P(m | h, a) for the sampled a) before the next agent draws.
3. Once every agent has acted, each agent's record for round t is built from its
   in-neighbours' outputs (rho_i). Nothing from round t is visible in round t.

Every decision is logged, one JSON line each, to
results/runs/<arm>/<condition>/run_<r>.jsonl.gz. A run's file is written once the run
is complete, so a condition resumes by skipping runs whose file exists. The logged p_H
is the kernel K_i(H | h); later steps use the logged numbers and never re-query.
"""

from __future__ import annotations

import datetime as _dt
import gzip
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .game import ACTIONS, DEFAULT_PARAMS, SIGNALS, PayoffParams
from .ladder import History, Record
from .policies import Backend, KernelOut, Query
from .provenance import bind_run_provenance
from .states import actions_state, key, signal_state
from .topology import Graph, make

REPO_ROOT = Path(__file__).resolve().parent.parent
P_VALID_FLAG = 0.9


@dataclass(frozen=True)
class Condition:
    """One cell of the run manifest."""

    arm: str
    arm_id: int
    condition_id: int
    channel: str
    n: int
    graph: str = "cycle"
    T: int = 20
    last_k: int | None = None
    runs: int = 50
    base_seed: int = 20261107
    params: PayoffParams = DEFAULT_PARAMS
    own_signal_in_state: bool = True

    @property
    def name(self) -> str:
        mem = "full" if self.last_k is None else f"last{self.last_k}"
        return f"{self.channel}_n{self.n}_{self.graph}_{mem}"

    def make_graph(self) -> Graph:
        return make(self.graph, self.n)

    def rng(self, run: int) -> np.random.Generator:
        ss = np.random.SeedSequence([self.base_seed, self.arm_id, self.condition_id, run])
        return np.random.Generator(np.random.PCG64(ss))


def sample_signal(u: float, probs: Sequence[float]) -> str:
    c = np.cumsum(probs)
    c = c / c[-1]  # guard against rounding: the probabilities sum to 1 up to ~1e-16
    return SIGNALS[int(np.searchsorted(c, u, side="right"))]


def build_records(graph: Graph, actions: Sequence[str],
                  signals: Sequence[str] | None) -> list[Record]:
    """rho_i for every agent: own output plus in-neighbour counts."""
    recs = []
    for i, nb in enumerate(graph.in_neighbours):
        n_high = sum(actions[j] == "H" for j in nb)
        if signals is None:
            recs.append(Record(actions[i], n_high, len(nb) - n_high))
        else:
            ms = [signals[j] for j in nb]
            recs.append(Record(actions[i], n_high, len(nb) - n_high, signal=signals[i],
                               sig_high=ms.count("high"), sig_low=ms.count("low"),
                               sig_none=ms.count("none")))
    return recs


def _rec_dict(r: Record) -> dict:
    d = {"action": r.action, "n_high": r.n_high, "n_low": r.n_low}
    if r.has_signals:
        d.update(signal=r.signal, sig_high=r.sig_high, sig_low=r.sig_low, sig_none=r.sig_none)
    return d


def run_file(out_dir: Path, run: int) -> Path:
    return out_dir / f"run_{run:03d}.jsonl.gz"


def simulate(cond: Condition, backend: Backend, runs: Sequence[int]) -> dict[int, list[dict]]:
    """Run the given runs of a condition in lockstep. Returns run -> decision lines."""
    g = cond.make_graph()
    cheap = cond.channel == "cheap_talk"
    hist: dict[int, list[History]] = {r: [() for _ in range(g.n)] for r in runs}
    rngs = {r: cond.rng(r) for r in runs}
    lines: dict[int, list[dict]] = {r: [] for r in runs}
    for t in range(1, cond.T + 1):
        queries = [Query(cond.channel, hist[r][i], g.in_degree(i), cond.last_k)
                   for r in runs for i in range(g.n)]
        outs: list[KernelOut] = backend.query(queries)
        if len(outs) != len(queries):
            raise RuntimeError("backend returned the wrong number of outputs")
        k = 0
        for r in runs:
            acts, sigs, rows = [], [], []
            for i in range(g.n):
                o, h = outs[k], hist[r][i]
                k += 1
                u_a = float(rngs[r].random())
                a = "H" if u_a < o.p_H else "L"
                row = {"arm": cond.arm, "channel": cond.channel, "n": g.n, "graph": cond.graph,
                       "run": r, "round": t, "agent": i, "d": g.in_degree(i),
                       "state": key(actions_state(h)),
                       "prompt_sha256": o.prompt_sha256, "n_tokens": o.n_tokens,
                       "p_H": o.p_H, "logprob_H": o.logprob_H, "logprob_L": o.logprob_L,
                       "p_valid": o.p_valid, "u_action": u_a, "action": a,
                       "replayed": o.replayed}
                if cheap:
                    u_m = float(rngs[r].random())
                    m = sample_signal(u_m, o.sig[a])
                    row.update(state_sig=key(signal_state(h, cond.own_signal_in_state)),
                               sig_after_H=list(o.sig["H"]), sig_after_L=list(o.sig["L"]),
                               sig_p_valid_H=o.sig_p_valid["H"], sig_p_valid_L=o.sig_p_valid["L"],
                               sig_logprobs=o.sig_logprobs, u_signal=u_m, signal=m)
                    sigs.append(m)
                acts.append(a)
                rows.append(row)
            recs = build_records(g, acts, sigs if cheap else None)
            for i, (row, rec) in enumerate(zip(rows, recs)):
                row["record"] = _rec_dict(rec)
                hist[r][i] = hist[r][i] + (rec,)
            lines[r].extend(rows)
    return lines


def write_run(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as fh:
        for ln in lines:
            fh.write(json.dumps(ln) + "\n")
    os.replace(tmp, path)


def read_run(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as fh:
        return [json.loads(x) for x in fh]


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def update_manifest(arm_dir: Path, cond: Condition, provenance: dict, session: dict,
                    summary: dict) -> None:
    """One entry per condition: its original provenance (never rewritten), every session
    that added runs (commit, backend, timings), and the current summary."""
    path = arm_dir / "manifest.json"
    man = json.loads(path.read_text()) if path.exists() else {"arm": cond.arm, "conditions": {}}
    old = man["conditions"].get(cond.name, {})
    man["conditions"][cond.name] = {
        "provenance": old.get("provenance", provenance),
        "sessions": old.get("sessions", []) + [session],
        "summary": summary}
    man["updated"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(man, indent=2, default=str) + "\n")


def run_condition(cond: Condition, backend: Backend, runs_root: Path, chunk: int | None = None,
                  extra_manifest: dict | None = None, log=print) -> Path:
    """Run every unfinished run of `cond`, `chunk` runs at a time in lockstep (all at
    once by default), writing each run's log as it completes. Returns the folder."""
    out = runs_root / cond.arm / cond.name
    # Review C2: bind the configuration before computing; a resume must match it.
    prov = bind_run_provenance(out, cond, backend, git_commit())
    todo = [r for r in range(cond.runs) if not run_file(out, r).exists()]
    if len(todo) < cond.runs:
        log(f"{cond.name}: {cond.runs - len(todo)} runs already logged, {len(todo)} to run")
    chunk = chunk or max(len(todo), 1)
    t0 = time.time()
    n_dec = 0
    for c in range(0, len(todo), chunk):
        part = todo[c:c + chunk]
        lines = simulate(cond, backend, part)
        for r in part:
            write_run(run_file(out, r), lines[r])
            n_dec += len(lines[r])
        log(f"{cond.name}: runs {part[0]}..{part[-1]} done ({time.time() - t0:.1f}s)")
    summary = summarise_condition(out, cond)
    update_manifest(runs_root / cond.arm, cond, prov, {
        "condition": {**asdict(cond), "name": cond.name},
        "backend": backend.describe(), "git_commit": git_commit(),
        "time": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "runs_added": len(todo), "seconds": round(time.time() - t0, 2), "decisions": n_dec,
        **(extra_manifest or {})}, summary)
    return out


def summarise_condition(out: Path, cond: Condition) -> dict:
    """Counts of logged runs and decisions, and the p_valid flag (< 0.9)."""
    files = sorted(out.glob("run_*.jsonl.gz"))
    n_dec, low = 0, []
    for f in files:
        for ln in read_run(f):
            n_dec += 1
            pv = [ln["p_valid"]] + ([ln["sig_p_valid_H"], ln["sig_p_valid_L"]]
                                    if "sig_p_valid_H" in ln else [])
            if min(pv) < P_VALID_FLAG:
                low.append({"run": ln["run"], "round": ln["round"], "agent": ln["agent"],
                            "p_valid": pv})
    return {"runs_logged": len(files), "decisions": n_dec,
            "p_valid_below_0.9": len(low), "p_valid_flagged_examples": low[:20]}


# -------------------------------------------------------------------- reading runs


@dataclass
class RunData:
    """Logged runs of one condition as arrays (run, round, agent); rounds are 1..T at
    index 0..T-1. Actions: 1 = H. Signals: index into game.SIGNALS, -1 on counts."""

    channel: str
    n: int
    graph: str
    T: int
    runs: np.ndarray
    actions: np.ndarray
    p_H: np.ndarray
    p_valid: np.ndarray
    state: np.ndarray
    sha: np.ndarray
    signals: np.ndarray | None = None
    sig: np.ndarray | None = None          # (R, T, n, 2, 3): P(m | h, a) for a = H, L
    state_sig: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def R(self) -> int:
        return len(self.runs)

    def subset(self, idx) -> "RunData":
        idx = np.asarray(idx)
        kw = {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and k != "runs" else v)
              for k, v in self.__dict__.items()}
        kw["runs"] = self.runs[idx]
        return RunData(**kw)

    def joint(self) -> np.ndarray:
        """K(a, m | h) for every decision, shape (R, T, n, 2, 3)."""
        p = self.p_H[..., None]
        return np.stack([p * self.sig[..., 0, :], (1 - p) * self.sig[..., 1, :]], axis=-2)


def load_runs(folder: Path) -> RunData:
    files = sorted(Path(folder).glob("run_*.jsonl.gz"))
    if not files:
        raise FileNotFoundError(f"no runs in {folder}")
    all_lines = [read_run(f) for f in files]
    first = all_lines[0][0]
    n, cheap = first["n"], first["channel"] == "cheap_talk"
    T = max(ln["round"] for ln in all_lines[0])
    R = len(files)
    acts = np.zeros((R, T, n), np.int8)
    p_h = np.zeros((R, T, n))
    p_v = np.zeros((R, T, n))
    st = np.empty((R, T, n), object)
    sha = np.empty((R, T, n), object)
    sigs = np.full((R, T, n), -1, np.int8)
    sig = np.zeros((R, T, n, 2, 3)) if cheap else None
    st_sig = np.empty((R, T, n), object) if cheap else None
    runs = np.zeros(R, int)
    for ri, lines in enumerate(all_lines):
        if len(lines) != T * n:
            raise ValueError(f"{files[ri]} has {len(lines)} lines, expected {T * n}")
        runs[ri] = lines[0]["run"]
        for ln in lines:
            t, i = ln["round"] - 1, ln["agent"]
            acts[ri, t, i] = ln["action"] == "H"
            p_h[ri, t, i] = ln["p_H"]
            p_v[ri, t, i] = ln["p_valid"]
            st[ri, t, i] = ln["state"]
            sha[ri, t, i] = ln["prompt_sha256"]
            if cheap:
                sigs[ri, t, i] = SIGNALS.index(ln["signal"])
                sig[ri, t, i, 0] = ln["sig_after_H"]
                sig[ri, t, i, 1] = ln["sig_after_L"]
                st_sig[ri, t, i] = ln["state_sig"]
    prov_path = Path(folder) / "provenance.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    return RunData(first["channel"], n, first["graph"], T, runs, acts, p_h, p_v, st, sha,
                   sigs if cheap else None, sig, st_sig,
                   meta={"folder": str(folder), "fingerprint": prov.get("fingerprint"),
                         "kernel_fingerprint": prov.get("kernel_fingerprint")})


assert ACTIONS == ("H", "L")
