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
    out = []
    for k in need:
        v = inp.get(k)
        if isinstance(v, dict):   # e.g. prompts_approved: one flag per template
            out += [f"{INPUT_TEXT[k]}: {name}" for name, ok in v.items() if not ok]
        elif not v:
            out.append(INPUT_TEXT[k])
    return out


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
    if use_deployed(cfg):
        import modal

        server = modal.Cls.from_name(serving.APP_NAME, "Server")(
            hf_id=m["hf_id"], revision=m["revision"], engine_kwargs_json=json.dumps(eng))
    else:
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

    import fcntl

    usd = seconds / 3600 * cfg["serving"]["l4_usd_per_hour"] * containers
    SPEND_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPEND_FILE.with_suffix(".lock"), "w") as lock:   # GPU scripts may run at once
        fcntl.flock(lock, fcntl.LOCK_EX)
        log = json.loads(SPEND_FILE.read_text()) if SPEND_FILE.exists() else []
        log.append({"what": what, "seconds": round(seconds, 1), "containers": containers,
                    "usd": round(usd, 4), "time": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "note": "wall-clock estimate at the list price; check the Modal dashboard"})
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


def current_workspace() -> str:
    """The Modal workspace of the active credentials (from `modal token info`)."""
    import re
    import subprocess

    r = subprocess.run(["modal", "token", "info"], capture_output=True, text=True)
    m = re.search(r"Workspace:\s*(\S+)", r.stdout)
    if r.returncode != 0 or not m:
        raise SystemExit("Modal is not signed in (or `modal token info` failed). A teammate "
                         "signs in to their own workspace with `python scripts/modal_login.py`.")
    return m.group(1)


def check_workspace(cfg: dict) -> str:
    ws = current_workspace()
    if ws in cfg.get("inputs", {}).get("forbidden_workspaces", []):
        raise SystemExit(f"Stopping: signed in to Modal workspace {ws!r}, which Abdul ruled "
                         "out for GPU runs. Sign in to a teammate's workspace instead.")
    print(f"Modal workspace: {ws}")
    return ws


def gpu_session(cfg: dict, need: tuple[str, ...]):
    """Context for Modal GPU work: checks the inputs, the Modal workspace, and that the
    batch-invariance flag and container count in the environment match the config, then
    runs the app."""
    import os
    from contextlib import contextmanager

    require_inputs(cfg, need)
    check_workspace(cfg)
    srv = cfg["serving"]
    env_bi = os.environ.get("SWARM_AUDIT_BATCH_INVARIANT", "0") == "1"
    if env_bi != bool(srv["batch_invariant"]):
        raise SystemExit(f"config serving.batch_invariant = {srv['batch_invariant']} but "
                         f"SWARM_AUDIT_BATCH_INVARIANT={'1' if env_bi else '0'}: set them alike")
    if int(os.environ.get("SWARM_AUDIT_MAX_CONTAINERS", "1")) != int(srv["max_containers"]):
        raise SystemExit("set SWARM_AUDIT_MAX_CONTAINERS to serving.max_containers")
    os.environ.setdefault("SWARM_AUDIT_SCALEDOWN_S", str(srv.get("scaledown_window_s", 60)))
    os.environ.setdefault("SWARM_AUDIT_TIMEOUT_MIN", str(srv["timeout_minutes"]))

    if use_deployed(cfg):
        check_deployed(cfg)

    @contextmanager
    def ctx():
        import modal

        if use_deployed(cfg):       # the deployed app outlives this script
            try:
                yield
            finally:                  # its GPU idles up to scaledown_window_s afterwards
                record_spend("idle after script (upper bound)",
                             float(srv["scaledown_window_s"]), cfg)
            return
        from .serving import app
        with modal.enable_output(), app.run():
            yield
    return ctx()


# ------------------------------------------------------------ the deployed server

DEPLOY_FILE = RESULTS / "serving" / "deployed.json"
SERVER_CODE = ("serving.py", "tokens.py", "ladder.py", "game.py", "__init__.py")


def server_code_sha() -> str:
    """sha256 over the package files the server container imports."""
    import hashlib

    h = hashlib.sha256()
    for name in SERVER_CODE:
        h.update(name.encode() + b"\0" + (Path(__file__).parent / name).read_bytes())
    return h.hexdigest()


def deploy_settings(cfg: dict) -> dict:
    srv = cfg["serving"]
    return {"batch_invariant": bool(srv["batch_invariant"]),
            "scaledown_window_s": int(srv["scaledown_window_s"]),
            "max_containers": int(srv["max_containers"]),
            "timeout_minutes": int(srv["timeout_minutes"]),
            "vllm_version": srv["vllm_version"], "gpu": srv["gpu"]}


def check_deployed(cfg: dict) -> None:
    """The deployed server must run this repository's serving code and settings, in the
    workspace that is signed in; otherwise redeploy (scripts/deploy_server.py)."""
    if not DEPLOY_FILE.exists():
        raise SystemExit("serving.deployed is true but nothing is deployed: run "
                         "scripts/deploy_server.py")
    d = json.loads(DEPLOY_FILE.read_text())
    problems = []
    if d.get("code_sha256") != server_code_sha():
        problems.append("the serving code changed since the deploy")
    if d.get("settings") != deploy_settings(cfg):
        problems.append(f"settings differ: deployed {d.get('settings')}, config {deploy_settings(cfg)}")
    if d.get("workspace") != current_workspace():
        problems.append(f"deployed in workspace {d.get('workspace')!r}, signed in elsewhere")
    if problems:
        raise SystemExit("The deployed server does not match: " + "; ".join(problems)
                         + ". Redeploy with scripts/deploy_server.py.")


# --------------------------------------------------------- label check, final prompts


def label_check_queries(cfg: dict, seed: int = 11) -> list[Query]:
    """The drift panel (every calibration state, both channels) plus round-20 p_test
    histories on both channels: the prompts whose label tokenisation must hold."""
    from .audit import synthetic_history

    pt = ptest(cfg)
    rng = np.random.default_rng(seed)
    long_pt = PTest(**{**pt.__dict__, "L_min": pt.L_max})   # the longest histories
    out = drift_panel(cfg)
    for ch in ("counts", "cheap_talk"):
        for s in all_actions_states(2, start=False):
            out.append(Query(ch, synthetic_history(rng, ch, 2, s, long_pt), 2))
    return out


def check_final_labels(tok: Any, cfg: dict) -> dict:
    """Every action label is one token with the configured id after every prompt, and
    every signal label likewise after each action label; plus prompt lengths."""
    from .tokens import chat_prefix, check_labels, encode

    lab = labels(cfg)
    m = cfg["model"]
    act_texts, sig_texts, n_tok = [], [], {"counts": 0, "cheap_talk": 0}
    for qy in label_check_queries(cfg):
        from .ladder import render

        r = render(qy.channel, qy.history, qy.d, lab, params(cfg), qy.last_k)
        text = chat_prefix(tok, r, m.get("chat_kwargs"))
        act_texts.append(text)
        n_tok[qy.channel] = max(n_tok[qy.channel], len(encode(tok, text)))
        if qy.channel == "cheap_talk":
            base = text[: -len(r.assistant)]
            sig_texts += [base + r.signal_prefix(a) for a in lab.actions]
    a = check_labels(tok, act_texts, lab.actions)
    s = check_labels(tok, sig_texts, lab.signals)
    ok = (a.ok and s.ok and list(a.ids) == m["label_ids"]["actions"]
          and list(s.ids) == m["label_ids"]["signals"])
    return {"pass": bool(ok), "action_ids": a.ids, "signal_ids": s.ids,
            "configured": m["label_ids"], "problems": a.problems + s.problems,
            "prompts_checked": len(act_texts), "signal_positions_checked": len(sig_texts),
            "max_prompt_tokens": n_tok, "max_model_len": cfg["serving"]["max_model_len"]}


# ------------------------------------------------------------- settled decisions

BUILD_CONFIGS = ("core.yaml", "calibration.yaml", "secondary.yaml")


def set_config_line(key: str, value: str, folder: Path = REPO_ROOT / "configs",
                    configs=BUILD_CONFIGS, note: str = "") -> None:
    """Rewrite the first line `  <key>: ...` (any indentation) in every build config,
    keeping the rest of the file and its comments. `value` is YAML text."""
    import re

    for name in configs:
        p = Path(folder) / name
        s = p.read_text()
        pat = re.compile(rf"^(\s*){re.escape(key)}:[^\n]*$", re.M)
        m = pat.search(s)
        if not m:
            raise ValueError(f"no `{key}:` line in {p}")
        line = f"{m.group(1)}{key}: {value}" + (f"   # {note}" if note else "")
        p.write_text(s[:m.start()] + line + s[m.end():])


def use_deployed(cfg: dict) -> bool:
    """serving.deployed, unless SWARM_AUDIT_DEPLOYED=0/1 overrides it for one script."""
    import os

    env = os.environ.get("SWARM_AUDIT_DEPLOYED")
    return bool(cfg["serving"].get("deployed")) if env is None else env == "1"


def numerical_floor(cfg: dict) -> float:
    f = cfg["serving"].get("numerical_floor")
    if f is None:
        raise SystemExit("G3 has not set the numerical floor: run scripts/check_serving.py "
                         "(and --decide)")
    return float(f)


PLAN_FILE = RESULTS / "serving" / "g4_plan.json"


def check_in_plan(arm: str, cond_names: list[str]) -> None:
    """Core and second-topology arms run only as the G4 plan allows."""
    if not PLAN_FILE.exists():
        raise SystemExit("no G4 plan (results/serving/g4_plan.json): run scripts/benchmark.py")
    plan = json.loads(PLAN_FILE.read_text())
    allowed = {(p["arm"], p["condition"]) for p in plan["run"]}
    cut = [c for c in cond_names if (arm, c) not in allowed]
    if cut:
        raise SystemExit(f"the G4 plan does not include {arm}: {cut} (cut to fit the budget; "
                         f"see {PLAN_FILE})")
