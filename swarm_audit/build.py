"""Config loading, conditions, backends and the input gate for the build scripts.

GPU work goes through `require_inputs`, which stops unless Abdul's inputs (build
hand-off section 1) are recorded in the config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .audit import AuditTable, PTest, derive_actions_table, enumerated_table, fit_table
from .game import PayoffParams
from .ladder import Labels
from .policies import Backend, LLMBackend, Query, ScriptedBackend
from .states import START, all_actions_states, all_signal_states, record_for_state
from .swarm import Condition

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"


def load(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / "configs" / p
    with open(p) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_path"] = str(p)
    return cfg


def params(cfg: dict) -> PayoffParams:
    return PayoffParams(**{k: float(v) for k, v in cfg["payoffs"].items()})


def ptest(cfg: dict) -> PTest:
    return PTest.from_config({**cfg["p_test"], "T": cfg["T"]})


def conditions(cfg: dict, arm: str, ns: list[int] | None = None) -> list[Condition]:
    a = cfg["arms"][arm]
    out = []
    for n in a["n"]:
        if ns and n not in ns:
            continue
        out.append(Condition(arm=arm, arm_id=int(a["arm_id"]), condition_id=int(n),
                             channel=a["channel"], n=int(n), graph=a["graph"], T=int(cfg["T"]),
                             last_k=a.get("last_k"), runs=int(a["runs"]),
                             base_seed=int(cfg["seeds"]["base_seed"]), params=params(cfg),
                             own_signal_in_state=bool(cfg["own_signal_in_state"])))
    return out


# ---------------------------------------------------------------------- the gate

INPUT_TEXT = {
    "model_confirmed": "Abdul has not confirmed the main model (inputs.model_confirmed)",
    "prompts_approved": "Abdul has not approved the prompt templates (inputs.prompts_approved)",
    "modal_account": "no Modal account chosen (inputs.modal_account)",
    "pilot_rule": "no pilot rule ('none' or a rule) recorded (inputs.pilot_rule)",
}


def missing_inputs(cfg: dict, need: tuple[str, ...]) -> list[str]:
    inp = cfg.get("inputs", {})
    return [INPUT_TEXT[k] for k in need if not inp.get(k)]


def require_inputs(cfg: dict, need: tuple[str, ...]) -> None:
    miss = missing_inputs(cfg, need)
    if miss:
        raise SystemExit("Stopping before any GPU run (build hand-off section 1):\n  - "
                         + "\n  - ".join(miss) + f"\nSet them in {cfg['_path']} once Abdul decides.")


# ---------------------------------------------------------------------- backends


def labels(cfg: dict) -> Labels:
    return Labels(**cfg["model"]["labels"])


def scripted(policy: str, cfg: dict) -> ScriptedBackend:
    return ScriptedBackend(policy, params=params(cfg))


def modal_backend(cfg: dict, log_name: str) -> tuple[LLMBackend, Any]:
    """The LLM backend on Modal: Server.label_logprobs (method logprob_token_ids), each
    call logged to logs/serving/. Must be used inside `serving.app.run()`."""
    from . import serving
    from .screening import load_tokenizer

    m, srv = cfg["model"], cfg["serving"]
    eng = {"max_model_len": srv["max_model_len"],
           "gpu_memory_utilization": srv["gpu_memory_utilization"],
           "enforce_eager": srv["enforce_eager"], **m.get("engine_kwargs", {})}
    server = serving.Server(hf_id=m["hf_id"], revision=m["revision"],
                            engine_kwargs_json=json.dumps(eng))
    meta = {"model": m["name"], "hf_id": m["hf_id"], "revision": m["revision"]}

    def call(prompts: list[list[int]], ids: list[int], kind: str) -> list[dict]:
        return serving.call_logged(server.label_logprobs, prompts, ids, method=srv["method"],
                                   log_name=log_name, prompts=prompts,
                                   prompt_ids=[str(i) for i in range(len(prompts))],
                                   meta={**meta, "call": kind, "allowed_token_ids": ids})

    tok = load_tokenizer(m)
    be = LLMBackend(tok, labels(cfg), tuple(m["label_ids"]["actions"]),
                    tuple(m["label_ids"]["signals"]), call, params(cfg), m.get("chat_kwargs"),
                    model_info={**meta, "vllm": srv["vllm_version"],
                                "batch_invariant": serving.BATCH_INVARIANT,
                                "engine_kwargs": eng})
    return be, server


# ------------------------------------------------------------------------ tables


def build_tables(backend: Backend, cfg: dict, channel: str, d: int, enumerated: bool = False,
                 actions_only: bool = False) -> dict[str, AuditTable]:
    """The audit tables of one channel and degree.

    counts: {"actions"}. cheap talk: {"signal", "actions" (derived, primary weights),
    "actions_equal" (derived, equal weights)}; with actions_only (K_10, arm 7) a directly
    fitted {"actions"} table instead. enumerated: the calibration tables (last round only).
    """
    own = bool(cfg["own_signal_in_state"])
    a, s = cfg["audit"]["actions"], cfg["audit"]["signal"]
    pt = ptest(cfg)
    if enumerated:
        if channel == "counts":
            return {"actions": enumerated_table(backend, channel, d, "actions", own)}
        sig = enumerated_table(backend, channel, d, "signal", own)
    elif channel == "counts" or actions_only:
        return {"actions": fit_table(backend, channel, d, "actions", a["n_fit"], a["n_score"],
                                     pt, own)}
    else:
        sig = fit_table(backend, channel, d, "signal", s["n_fit"], s["n_score"], pt, own)
    return {"signal": sig, "actions": derive_actions_table(sig, "primary"),
            "actions_equal": derive_actions_table(sig, "equal")}


def table_dir(cfg: dict, backend_name: str) -> Path:
    return RESULTS / "audit" / backend_name.replace(":", "_")


def save_tables(tables: dict[str, AuditTable], folder: Path, tag: str) -> None:
    for k, t in tables.items():
        t.save(folder / f"{tag}_{k}.json")


def load_tables(folder: Path, tag: str):
    from .analysis import Tables

    def get(k):
        p = folder / f"{tag}_{k}.json"
        return AuditTable.load(p) if p.exists() else None
    return Tables(get("actions"), get("signal"), get("actions_equal"))


# ------------------------------------------------------------------- drift panel


def drift_panel(cfg: dict, n_core: int = 10, seed: int = 99) -> list[Query]:
    """Every calibration state (both channels, last round only), plus `n_core` core audit
    prompts per channel drawn from p_test (full history, d = 2)."""
    from .audit import synthetic_history

    own = bool(cfg["own_signal_in_state"])
    out = []
    for s in all_actions_states(2):
        out.append(Query("counts", () if s == START else (record_for_state(s, 2),), 2, 1))
    for s in all_signal_states(2, own):
        out.append(Query("cheap_talk", () if s == START else (record_for_state(s, 2),), 2, 1))
    rng = np.random.default_rng(seed)
    pt = ptest(cfg)
    for ch in ("counts", "cheap_talk"):
        sts = all_actions_states(2, start=False)
        for j in range(n_core):
            s = sts[j % len(sts)]
            out.append(Query(ch, synthetic_history(rng, ch, 2, s, pt), 2))
    return out


def panel_outputs(backend: Backend, panel: list[Query]) -> list[dict]:
    outs = backend.query(panel)
    return [{"p_H": o.p_H, "sig": o.sig, "sha": o.prompt_sha256} for o in outs]


def panel_max_change(a: list[dict], b: list[dict]) -> float:
    m = 0.0
    for x, y in zip(a, b):
        m = max(m, abs(x["p_H"] - y["p_H"]))
        if x["sig"]:
            for act in "HL":
                m = max(m, max(abs(p - q) for p, q in zip(x["sig"][act], y["sig"][act])))
    return m


class PanelMixer:
    """Wraps a backend: the first call whose batch has at least `min_batch` queries also
    carries the drift panel (in shuffled positions); its outputs are kept in `mixed`."""

    def __init__(self, inner: Backend, panel: list[Query], min_batch: int = 1, seed: int = 0):
        self.inner, self.panel, self.min_batch = inner, panel, min_batch
        self.mixed: list[dict] | None = None
        self.rng = np.random.default_rng(seed)
        self.name = inner.name

    def query(self, queries):
        if self.mixed is not None or len(queries) < self.min_batch:
            return self.inner.query(queries)
        allq = list(queries) + self.panel
        perm = self.rng.permutation(len(allq))
        outs = self.inner.query([allq[i] for i in perm])
        back = [None] * len(allq)
        for pos, i in enumerate(perm):
            back[i] = outs[pos]
        self.mixed = [{"p_H": o.p_H, "sig": o.sig, "sha": o.prompt_sha256}
                      for o in back[len(queries):]]
        return back[:len(queries)]

    def describe(self):
        return self.inner.describe()


# ------------------------------------------------------------------- GPU sessions

SPEND_FILE = RESULTS / "spend.json"


def spent_usd() -> float:
    """GPU spend recorded so far by the build scripts (results/spend.json)."""
    if not SPEND_FILE.exists():
        return 0.0
    return float(sum(e["usd"] for e in json.loads(SPEND_FILE.read_text())))


def record_spend(what: str, seconds: float, cfg: dict, containers: int = 1) -> float:
    import datetime as dt

    usd = seconds / 3600 * cfg["serving"]["l4_usd_per_hour"] * containers
    log = json.loads(SPEND_FILE.read_text()) if SPEND_FILE.exists() else []
    log.append({"what": what, "seconds": round(seconds, 1), "containers": containers,
                "usd": round(usd, 4), "time": dt.datetime.now(dt.timezone.utc).isoformat(),
                "note": "wall-clock estimate at the list price; check the Modal dashboard"})
    SPEND_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPEND_FILE.write_text(json.dumps(log, indent=1) + "\n")
    return usd


def check_cost(cfg: dict, worst_minutes: float) -> None:
    """Stop if the worst case (every container to its time limit) would pass the cap."""
    srv = cfg["serving"]
    worst = worst_minutes / 60 * srv["l4_usd_per_hour"] * srv["max_containers"]
    left = srv["cost_cap_usd"] - spent_usd()
    print(f"worst-case GPU cost ${worst:.2f}; ${left:.2f} left of the ${srv['cost_cap_usd']:.2f} cap")
    if worst > left:
        raise SystemExit("Stopping: the worst-case cost would pass the cap. Ask Abdul.")


def gpu_session(cfg: dict, need: tuple[str, ...]):
    """Context for Modal GPU work: checks the inputs and that the batch-invariance flag
    and container count in the environment match the config, then runs the app."""
    import os
    from contextlib import contextmanager

    require_inputs(cfg, need)
    srv = cfg["serving"]
    env_bi = os.environ.get("SWARM_AUDIT_BATCH_INVARIANT", "0") == "1"
    if env_bi != bool(srv["batch_invariant"]):
        raise SystemExit(f"config serving.batch_invariant = {srv['batch_invariant']} but "
                         f"SWARM_AUDIT_BATCH_INVARIANT={'1' if env_bi else '0'}: set them alike")
    if int(os.environ.get("SWARM_AUDIT_MAX_CONTAINERS", "1")) != int(srv["max_containers"]):
        raise SystemExit("set SWARM_AUDIT_MAX_CONTAINERS to serving.max_containers")
    os.environ.setdefault("SWARM_AUDIT_TIMEOUT_MIN", str(srv["timeout_minutes"]))

    @contextmanager
    def ctx():
        import modal

        from .serving import app
        with modal.enable_output(), app.run():
            yield
    return ctx()
