"""Information ladder: what agents see at each step, and the coarse-graining q.

A history h = (r^1, ..., r^{t-1}) is a tuple of `Record`s, oldest first; round 1 has
the empty history. `render` turns a history into the prompt pieces for a step, using
the templates in swarm_audit/prompts/. Chat templating and tokenisation are in
swarm_audit/tokens.py, because they depend on the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .game import ACTIONS, DEFAULT_PARAMS, SIGNALS, PayoffParams, frac_high, u

STEPS: tuple[str, str] = ("counts", "cheap_talk")
PROMPT_DIR = Path(__file__).parent / "prompts"
_TEMPLATE_FILES = {"counts": "counts.txt", "cheap_talk": "cheap_talk.txt"}
_SECTION_RE = re.compile(r"^\[([a-z_]+)\]\s*$")


@dataclass(frozen=True)
class Record:
    """r^s_i: agent i's own output in round s and its in-neighbours' counts.

    `signal` and the `sig_*` counts are set on the cheap-talk step only.
    """

    action: str
    n_high: int
    n_low: int
    signal: str | None = None
    sig_high: int | None = None
    sig_low: int | None = None
    sig_none: int | None = None

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}, got {self.action!r}")
        if self.n_high < 0 or self.n_low < 0 or self.n_high + self.n_low < 1:
            raise ValueError("action counts must be non-negative with d >= 1")
        sig = (self.sig_high, self.sig_low, self.sig_none)
        if self.has_signals:
            if self.signal not in SIGNALS:
                raise ValueError(f"signal must be one of {SIGNALS}, got {self.signal!r}")
            if any(c is None or c < 0 for c in sig) or sum(sig) != self.d:
                raise ValueError("signal counts must be non-negative and sum to d")
        elif any(c is not None for c in sig):
            raise ValueError("signal counts given without the agent's own signal")

    @property
    def d(self) -> int:
        return self.n_high + self.n_low

    @property
    def f(self) -> float:
        return frac_high(self.n_high, self.d)

    @property
    def has_signals(self) -> bool:
        return self.signal is not None


History = tuple[Record, ...]


def q(record: Record) -> Record:
    """Coarse-graining: keep the action and action counts; drop signals."""
    return Record(record.action, record.n_high, record.n_low)


def pi_k(coarse_history: History, k: int) -> History:
    """Keep the last k entries of a (coarse) history; all of it if shorter."""
    if k < 0:
        raise ValueError("k must be non-negative")
    return tuple(coarse_history[-k:]) if k > 0 else ()


@dataclass(frozen=True)
class Labels:
    """Output labels for one model. Each must be a single token where it is generated."""

    H: str
    L: str
    sig_high: str | None = None
    sig_low: str | None = None
    sig_none: str | None = None

    @property
    def actions(self) -> tuple[str, str]:
        """Labels in the order of game.ACTIONS."""
        return (self.H, self.L)

    @property
    def signals(self) -> tuple[str, str, str]:
        """Labels in the order of game.SIGNALS."""
        if self.sig_high is None or self.sig_low is None or self.sig_none is None:
            raise ValueError("these labels have no signal labels")
        return (self.sig_high, self.sig_low, self.sig_none)


@dataclass(frozen=True)
class Rendered:
    """The step's prompt, before chat templating.

    `assistant` is the reply prefix; the next token is the action label. On the cheap-talk
    step, `signal_prefix(action_label)` is the reply prefix before the signal label.
    """

    step: str
    system: str
    user: str
    assistant: str
    after_bid: str | None = None

    def signal_prefix(self, action_label: str) -> str:
        if self.after_bid is None:
            raise ValueError(f"step {self.step!r} has no signal")
        return f"{self.assistant}{action_label}\n{self.after_bid}"


@lru_cache(maxsize=None)
def load_template(step: str) -> dict[str, str]:
    """Parse a prompt file into its sections (see the comments at the top of the file)."""
    if step not in _TEMPLATE_FILES:
        raise ValueError(f"step must be one of {STEPS}, got {step!r}")
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in (PROMPT_DIR / _TEMPLATE_FILES[step]).read_text().splitlines():
        m = _SECTION_RE.match(line)
        if m:
            current = m.group(1)
            sections[current] = []
        elif current is None:
            if line.strip() and not line.startswith("#"):
                raise ValueError(f"text before the first section in {step}: {line!r}")
        else:
            sections[current].append(line)
    return {name: "\n".join(lines).strip("\n") for name, lines in sections.items()}


def fmt_num(x: float) -> str:
    """Print a payoff with at most two decimals and no trailing zeros."""
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


_BID_WORD = {"H": "high", "L": "low"}


def render(
    step: str,
    history: History,
    d: int,
    labels: Labels,
    params: PayoffParams = DEFAULT_PARAMS,
    last_k: int | None = None,
) -> Rendered:
    """Render an agent's history into the template for `step`.

    `d` is the agent's in-degree. `last_k`, if given, cuts the history to its last k
    rounds (the calibration arm); the last payoff is still that of the latest round.
    """
    tpl = load_template(step)
    cheap = step == "cheap_talk"
    shown = history if last_k is None else pi_k(history, last_k)

    for r in history:
        if r.d != d:
            raise ValueError(f"record has {r.d} neighbours but d = {d}")
        if r.has_signals != cheap:
            raise ValueError(f"record signals do not match step {step!r}")

    fill: dict[str, str] = {
        "d": str(d),
        "uLH": fmt_num(params.uLH),
        "uHH": fmt_num(params.uHH),
        "uLL": fmt_num(params.uLL),
        "uHL": fmt_num(params.uHL),
        "slope_H": fmt_num(params.uHH - params.uHL),
        "slope_L": fmt_num(params.uLH - params.uLL),
        "H": labels.H.strip(),
        "L": labels.L.strip(),
    }
    if cheap:
        s_high, s_low, s_none = labels.signals
        fill.update(S_HIGH=s_high.strip(), S_LOW=s_low.strip(), S_NONE=s_none.strip())

    if not shown:
        body = tpl["first_round"]
    else:
        lines = [tpl["history_header"]]
        n = len(shown)
        for i, r in enumerate(shown):
            k = n - i  # rounds ago
            when = tpl["when_last"] if k == 1 else tpl["when_ago"].format(k=k)
            row = dict(fill, when=when, own_bid=_BID_WORD[r.action],
                       n_high=str(r.n_high), n_low=str(r.n_low))
            if cheap:
                row.update(own_message=tpl[f"own_message_{r.signal}"],
                           s_high=str(r.sig_high), s_low=str(r.sig_low),
                           s_none=str(r.sig_none))
            lines.append(tpl["history_line"].format(**row))
        last = history[-1]
        lines.append("")
        lines.append(tpl["last_payoff"].format(payoff=fmt_num(u(last.action, last.f, params))))
        body = "\n".join(lines)

    return Rendered(
        step=step,
        system=tpl["system"].format(**fill),
        user=body + "\n\n" + tpl["instruction"].format(**fill),
        assistant=tpl["assistant"],
        after_bid=tpl.get("assistant_after_bid") if cheap else None,
    )
