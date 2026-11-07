"""Model screening: the state grid, label choice, prompt building and summary metrics.

Used by scripts/screen_tokenizers.py and scripts/screen_models.py; settings are in
configs/screening.yaml. See docs/handoff-model-screening.md.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .game import ACTIONS, SIGNALS, PayoffParams
from .ladder import History, Labels, Record, render
from .tokens import LabelCheck, chat_prefix, check_labels, encode

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "screening.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def payoff_params(cfg: dict) -> PayoffParams:
    return PayoffParams(**{k: float(v) for k, v in cfg["payoffs"].items()})


# --------------------------------------------------------------------------- grid


@dataclass(frozen=True)
class State:
    """One screening prompt: a state (the latest round) at a given history length."""

    step: str
    d: int
    f: float
    own_last: str
    signal_variant: str | None
    filler: int | None  # None: history length 1; else the index of the random filler
    history: History

    @property
    def length(self) -> int:
        return len(self.history)

    @property
    def state_id(self) -> str:
        sv = f"_{self.signal_variant}" if self.signal_variant else ""
        return f"{self.step}_d{self.d}_f{self.f:g}_own{self.own_last}{sv}"

    @property
    def prompt_id(self) -> str:
        return f"{self.state_id}_len{self.length}" + (
            f"_fill{self.filler}" if self.filler is not None else ""
        )


def _random_record(rng: random.Random, d: int, cheap: bool) -> Record:
    """A random earlier round: own action uniform, number of high neighbours uniform on 0..d,
    and on the cheap-talk step own signal uniform and each neighbour's signal uniform."""
    n_high = rng.randint(0, d)
    action = rng.choice(ACTIONS)
    if not cheap:
        return Record(action, n_high, d - n_high)
    sigs = [rng.choice(SIGNALS) for _ in range(d)]
    return Record(action, n_high, d - n_high, signal=rng.choice(SIGNALS),
                  sig_high=sigs.count("high"), sig_low=sigs.count("low"),
                  sig_none=sigs.count("none"))


def build_grid(cfg: dict) -> list[State]:
    """All screening states. Fillers (random earlier rounds) are shared by every state
    with the same step and d, so differences between states isolate the latest round."""
    g = cfg["grid"]
    rng = random.Random(cfg["seed"])
    n_fill, n_earlier = g["fillers"], g["filler_length"] - 1
    states: list[State] = []

    def fillers(d: int, cheap: bool) -> list[list[Record]]:
        return [[_random_record(rng, d, cheap) for _ in range(n_earlier)] for _ in range(n_fill)]

    c = g["counts"]
    for d in c["d"]:
        fill = fillers(d, cheap=False)
        for f in c["f"]:
            n_high = round(f * d)
            if abs(n_high / d - f) > 1e-9:
                raise ValueError(f"f = {f} is not a multiple of 1/{d}")
            for a in c["own_last"]:
                last = Record(a, n_high, d - n_high)
                states.append(State("counts", d, f, a, None, None, (last,)))
                for j, earlier in enumerate(fill):
                    states.append(State("counts", d, f, a, None, j, (*earlier, last)))

    ct = g["cheap_talk"]
    d, n_high = ct["d"], ct["n_high"]
    fill = fillers(d, cheap=True)
    for variant, (sh, sl, sn) in ct["signal_variants"].items():
        for a in ct["own_last"]:
            last = Record(a, n_high, d - n_high, signal=ct["own_last_signal"],
                          sig_high=sh, sig_low=sl, sig_none=sn)
            states.append(State("cheap_talk", d, n_high / d, a, variant, None, (last,)))
            for j, earlier in enumerate(fill):
                states.append(State("cheap_talk", d, n_high / d, a, variant, j, (*earlier, last)))
    return states


# ----------------------------------------------------------------------- labels


def load_tokenizer(model_cfg: dict) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_cfg["hf_id"], revision=model_cfg.get("revision"),
        **model_cfg.get("tokenizer_kwargs", {}),
    )


def labels_from_sets(cfg: dict, action_set: str, signal_set: str) -> Labels:
    h, l = cfg["label_sets"]["actions"][action_set]
    sh, sl, sn = cfg["label_sets"]["signals"][signal_set]
    return Labels(h, l, sh, sl, sn)


@dataclass
class Prompt:
    state: State
    text: str            # chat-templated, ends with the reply prefix: the action is next
    ids: list[int]


def build_prompts(tokenizer: Any, model_cfg: dict, states: list[State], labels: Labels,
                  params: PayoffParams) -> list[Prompt]:
    out = []
    for s in states:
        text = chat_prefix(tokenizer, render(s.step, s.history, s.d, labels, params),
                           model_cfg.get("chat_kwargs"))
        out.append(Prompt(s, text, encode(tokenizer, text)))
    return out


def signal_prefix_text(prompt: Prompt, labels: Labels, params: PayoffParams,
                       action_label: str) -> str:
    """Prompt text before the signal label, after the agent has emitted `action_label`."""
    s = prompt.state
    rendered = render(s.step, s.history, s.d, labels, params)
    return prompt.text[: -len(rendered.assistant)] + rendered.signal_prefix(action_label)


def screen_label_sets(tokenizer: Any, model_cfg: dict, cfg: dict, states: list[State],
                      params: PayoffParams) -> dict[str, dict[str, LabelCheck]]:
    """Run the single-token check for every label set on every grid prompt.

    Action sets are checked at the action position of all prompts. Signal sets are
    checked at the signal position of the cheap-talk prompts, after each action label of
    the preferred passing action set (or of the first action set, if none passes)."""
    pref = cfg["label_preference"]
    sig_sets = cfg["label_sets"]["signals"]
    act_sets = cfg["label_sets"]["actions"]
    default_sig = pref["signals"][0]

    actions: dict[str, LabelCheck] = {}
    for name in act_sets:
        labels = labels_from_sets(cfg, name, default_sig)
        prompts = build_prompts(tokenizer, model_cfg, states, labels, params)
        actions[name] = check_labels(tokenizer, (p.text for p in prompts), labels.actions)

    act_choice = next((n for n in pref["actions"] if actions[n].ok), pref["actions"][0])
    cheap = [s for s in states if s.step == "cheap_talk"]
    signals: dict[str, LabelCheck] = {}
    for name in sig_sets:
        labels = labels_from_sets(cfg, act_choice, name)
        prompts = build_prompts(tokenizer, model_cfg, cheap, labels, params)
        texts = [signal_prefix_text(p, labels, params, a) for p in prompts for a in labels.actions]
        signals[name] = check_labels(tokenizer, texts, labels.signals)
    return {"actions": actions, "signals": signals, "signal_checked_after": act_choice}


def choose_labels(cfg: dict, model_cfg: dict, screen: dict) -> tuple[str, str] | None:
    """(action set, signal set): the model's explicit choice, else the first passing set
    in the preference order. None if no set passes."""
    if "labels" in model_cfg:
        return model_cfg["labels"]["actions"], model_cfg["labels"]["signals"]
    pref = cfg["label_preference"]
    a = next((n for n in pref["actions"] if screen["actions"][n].ok), None)
    s = next((n for n in pref["signals"] if screen["signals"][n].ok), None)
    if a is None or s is None or a != screen["signal_checked_after"]:
        return None
    return a, s


# ---------------------------------------------------------------------- metrics


UNSAT_LO, UNSAT_HI = 0.02, 0.98


def summarise_model(rows: list[dict]) -> dict:
    """Screening metrics for one model from its screening.csv rows (dicts)."""
    import numpy as np

    def p(r: dict) -> float:
        return float(r["p_H"])

    by_id = {r["prompt_id"]: r for r in rows}
    out: dict[str, Any] = {}

    for step in ("counts", "cheap_talk", "all"):
        sel = [p(r) for r in rows if step == "all" or r["step"] == step]
        if sel:
            arr = np.array(sel)
            out[f"unsaturated_{step}"] = float(np.mean((arr >= UNSAT_LO) & (arr <= UNSAT_HI)))
            out[f"n_{step}"] = len(sel)

    # Neighbour responsiveness: |P(H | f=1) - P(H | f=0)| for each own action and d.
    neigh = []
    for r in rows:
        if r["step"] != "counts" or float(r["f"]) != 1.0:
            continue
        twin_id = r["prompt_id"].replace(f"_f{float(r['f']):g}_", "_f0_")
        twin = by_id.get(twin_id)
        if twin is not None:
            neigh.append({"d": int(r["d"]), "own_last": r["own_last"],
                          "length": int(r["length"]), "filler": r["filler"],
                          "delta": abs(p(r) - p(twin))})
    out["neighbour"] = neigh

    # Signal responsiveness: max - min of P(H) over the signal variants, fixed action counts.
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        if r["step"] == "cheap_talk":
            groups.setdefault((r["own_last"], int(r["length"]), r["filler"]), []).append(p(r))
    out["signal"] = [{"own_last": k[0], "length": k[1], "filler": k[2],
                      "spread": max(v) - min(v), "n_variants": len(v)}
                     for k, v in sorted(groups.items(), key=str)]

    # History sensitivity: spread of P(H) across the random fillers of each state.
    fills: dict[str, list[float]] = {}
    for r in rows:
        if r["filler"] not in ("", None):
            fills.setdefault(r["state_id"], []).append(p(r))
    out["history"] = {sid: max(v) - min(v) for sid, v in fills.items()}

    pv = np.array([float(r["p_valid"]) for r in rows])
    out["p_valid_mean"], out["p_valid_min"] = float(pv.mean()), float(pv.min())
    sig_pv = [float(r[c]) for r in rows for c in ("p_valid_sig_after_H", "p_valid_sig_after_L")
              if r.get(c) not in ("", None)]
    if sig_pv:
        out["p_valid_sig_mean"], out["p_valid_sig_min"] = float(np.mean(sig_pv)), float(np.min(sig_pv))
    return out
