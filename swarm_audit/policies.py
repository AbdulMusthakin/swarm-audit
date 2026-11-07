"""Kernel backends: the one interface the swarm, audit and analysis code talks to.

A backend takes a batch of `Query`s (channel, an agent's full history h, its in-degree d,
and the memory setting last_k) and returns one `KernelOut` per query: P(H | h) and, on
the cheap-talk channel, the signal distribution P(m | h, a) after *both* actions, so
the six-way output kernel K_i(a, m | h) = P(a | h) P(m | h, a) is known exactly.

Two backends:
- `ScriptedBackend`: a hand-written policy computes the probabilities directly (hand-off
  section 6). Used for the harness check (gate G2) and the tests.
- `LLMBackend`: renders the prompt with `ladder.render`, chat-templates and tokenises it
  with `tokens`, and reads the label log-probabilities from a server call
  (`Server.label_logprobs`, method `logprob_token_ids`, on Modal; or any callable with the
  same signature). This is the screening code path.

The rest of the package never looks inside a backend: it only sees `KernelOut`s.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from .game import DEFAULT_PARAMS, SIGNALS, PayoffParams
from .ladder import History, Labels, Rendered, pi_k, render

CHANNELS = ("counts", "cheap_talk")
SIG_UNIFORM = (1 / 3, 1 / 3, 1 / 3)


@dataclass(frozen=True)
class Query:
    """One decision to score. `history` is the agent's full history; the backend shows
    the agent only its last `last_k` rounds when `last_k` is set (calibration arm)."""

    channel: str
    history: History
    d: int
    last_k: int | None = None

    @property
    def shown(self) -> History:
        return self.history if self.last_k is None else pi_k(self.history, self.last_k)


@dataclass
class KernelOut:
    """The exact output distribution for one query, plus what the logs need."""

    p_H: float
    p_valid: float = 1.0
    logprob_H: float | None = None
    logprob_L: float | None = None
    # Cheap talk: P(m | h, a) over game.SIGNALS, for a = "H" and a = "L".
    sig: dict[str, tuple[float, float, float]] | None = None
    sig_p_valid: dict[str, float] | None = None
    sig_logprobs: dict[str, tuple[float, float, float]] | None = None
    prompt_sha256: str | None = None
    n_tokens: int | None = None

    def joint(self) -> np.ndarray:
        """K(a, m | h) as a 2 x 3 array (rows H, L; columns game.SIGNALS)."""
        if self.sig is None:
            raise ValueError("no signal distribution (counts channel)")
        return np.array([[self.p_H * x for x in self.sig["H"]],
                         [(1 - self.p_H) * x for x in self.sig["L"]]])


class Backend(Protocol):
    name: str

    def query(self, queries: Sequence[Query]) -> list[KernelOut]: ...

    def describe(self) -> dict: ...


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ scripted policies

# A policy maps (channel, visible history, d) to (P(H), {a: P(m | a)} or None).
PolicyFn = Callable[[str, History, int], tuple[float, dict | None]]


def _copy_p(record) -> float:
    """Copy: 0.8 if all neighbours bid high last round, 0.5 if half did, 0.2 if none
    (on degree 2: both / one / none). In general 0.2 + 0.6 f."""
    return 0.2 + 0.6 * record.f


def coin(channel: str, shown: History, d: int):
    return 0.5, None


def copy(channel: str, shown: History, d: int):
    return (0.5 if not shown else _copy_p(shown[-1])), None


def two_round(channel: str, shown: History, d: int):
    """Like Copy, averaged over the last two rounds (the last one only in round 2)."""
    if not shown:
        return 0.5, None
    return float(np.mean([_copy_p(r) for r in shown[-2:]])), None


HIDDEN_SIG = {"H": (0.7, 0.15, 0.15), "L": (0.1, 0.45, 0.45)}


def hidden_signal(channel: str, shown: History, d: int):
    """Cheap talk: P(H) = 0.8 if any neighbour said "high" last round, else 0.2 (0.5 in
    round 1). Say "high" w.p. 0.7 after H and 0.1 after L; the rest splits evenly
    between "low" and silence."""
    if channel != "cheap_talk":
        raise ValueError("the hidden-signal policy needs the cheap-talk channel")
    if not shown:
        return 0.5, HIDDEN_SIG
    return (0.8 if shown[-1].sig_high > 0 else 0.2), HIDDEN_SIG


POLICIES: dict[str, PolicyFn] = {
    "coin": coin, "copy": copy, "two_round": two_round, "hidden_signal": hidden_signal,
}


@dataclass
class ScriptedBackend:
    """Computes the kernel from a scripted policy. With `render_prompts`, it also renders
    each prompt (no tokeniser) so its sha256 is logged as for the LLM."""

    policy: str
    labels: Labels = field(default_factory=lambda: Labels(" H", " L", " high", " low", " none"))
    params: PayoffParams = DEFAULT_PARAMS
    render_prompts: bool = True

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"unknown policy {self.policy!r}; choose from {sorted(POLICIES)}")
        self.fn = POLICIES[self.policy]
        self.name = f"scripted:{self.policy}"

    def query(self, queries: Sequence[Query]) -> list[KernelOut]:
        out = []
        for qy in queries:
            p_h, sig = self.fn(qy.channel, qy.shown, qy.d)
            if qy.channel == "cheap_talk" and sig is None:
                sig = {"H": SIG_UNIFORM, "L": SIG_UNIFORM}
            sha = None
            if self.render_prompts:
                r = render(qy.channel, qy.history, qy.d, self.labels, self.params, qy.last_k)
                sha = prompt_sha256(r.system + "\x00" + r.user + "\x00" + r.assistant)
            out.append(KernelOut(
                p_H=float(p_h), p_valid=1.0,
                sig=None if qy.channel == "counts" else {a: tuple(map(float, sig[a])) for a in "HL"},
                sig_p_valid=None if qy.channel == "counts" else {"H": 1.0, "L": 1.0},
                prompt_sha256=sha))
        return out

    def describe(self) -> dict:
        return {"backend": "scripted", "policy": self.policy}


# ------------------------------------------------------------------------ LLM backend

# call(prompts as token-id lists, allowed token ids, kind) -> one dict per prompt with
# "logprobs" (raw, one per allowed id), "probs" (renormalised) and "p_valid".
ServerCall = Callable[[list[list[int]], list[int], str], list[dict]]


@dataclass
class LLMBackend:
    """Render, chat-template, tokenise, and read exact label probabilities from a server.

    `action_ids` and `signal_ids` are the label token ids (from the tokenizer screen).
    With `verify_labels`, every prompt is checked so that each label is exactly that one
    token where it is generated (`encode(prompt + label) == encode(prompt) + [id]`)."""

    tokenizer: Any
    labels: Labels
    action_ids: tuple[int, int]
    signal_ids: tuple[int, int, int] | None
    call: ServerCall
    params: PayoffParams = DEFAULT_PARAMS
    chat_kwargs: dict | None = None
    verify_labels: bool = True
    model_info: dict = field(default_factory=dict)
    name: str = "llm"

    def _texts(self, qy: Query) -> tuple[str, list[str]]:
        from .tokens import chat_prefix

        r: Rendered = render(qy.channel, qy.history, qy.d, self.labels, self.params, qy.last_k)
        text = chat_prefix(self.tokenizer, r, self.chat_kwargs)
        if qy.channel == "counts":
            return text, []
        base = text[: -len(r.assistant)] if r.assistant else text
        return text, [base + r.signal_prefix(lab) for lab in self.labels.actions]

    def _encode(self, text: str, labels: Sequence[str], ids: Sequence[int]) -> list[int]:
        from .tokens import encode

        toks = encode(self.tokenizer, text)
        if self.verify_labels:
            for lab, tid in zip(labels, ids):
                full = encode(self.tokenizer, text + lab)
                if full != toks + [tid]:
                    raise ValueError(f"label {lab!r} is not the single token {tid} after "
                                     f"this prompt (prompt ends {text[-40:]!r})")
        return toks

    def query(self, queries: Sequence[Query]) -> list[KernelOut]:
        texts, sig_texts = zip(*(self._texts(qy) for qy in queries)) if queries else ((), ())
        act_tok = [self._encode(t, self.labels.actions, self.action_ids) for t in texts]
        res = self.call(act_tok, list(self.action_ids), "action")
        outs = []
        for t, toks, r in zip(texts, act_tok, res):
            lp = r["logprobs"]
            outs.append(KernelOut(p_H=float(r["probs"][0]), p_valid=float(r["p_valid"]),
                                  logprob_H=float(lp[0]), logprob_L=float(lp[1]),
                                  prompt_sha256=prompt_sha256(t), n_tokens=len(toks)))
        cheap = [i for i, qy in enumerate(queries) if qy.channel == "cheap_talk"]
        if cheap:
            if self.signal_ids is None:
                raise ValueError("cheap-talk queries need signal label ids")
            flat = [self._encode(s, self.labels.signals, self.signal_ids)
                    for i in cheap for s in sig_texts[i]]
            sres = self.call(flat, list(self.signal_ids), "signal")
            for j, i in enumerate(cheap):
                rh, rl = sres[2 * j], sres[2 * j + 1]
                o = outs[i]
                o.sig = {"H": tuple(map(float, rh["probs"])), "L": tuple(map(float, rl["probs"]))}
                o.sig_p_valid = {"H": float(rh["p_valid"]), "L": float(rl["p_valid"])}
                o.sig_logprobs = {"H": tuple(map(float, rh["logprobs"])),
                                  "L": tuple(map(float, rl["logprobs"]))}
        return outs

    def describe(self) -> dict:
        return {"backend": "llm", "labels": [*self.labels.actions,
                                             *(self.labels.signals if self.signal_ids else ())],
                "action_ids": list(self.action_ids),
                "signal_ids": list(self.signal_ids) if self.signal_ids else None,
                "chat_kwargs": self.chat_kwargs, **self.model_info}


def check_signals_order() -> None:
    """KernelOut.sig is ordered as game.SIGNALS; this guards against a silent reorder."""
    assert SIGNALS == ("high", "low", "none")


check_signals_order()
