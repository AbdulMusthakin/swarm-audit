"""Repeated procurement auction: binary bid, cheap-talk signals, payoffs.

Notation follows setup-paper/sections/2-setup.tex. Agent i bids H (high) or L (low).
Its payoff depends only on its own bid and on f, the fraction of its in-neighbours
who bid high (the agent itself is not counted):

    u(H, f) = uHL + f * (uHH - uHL)
    u(L, f) = uLL + f * (uLH - uLL)

Signals (cheap talk) never enter the payoff.
"""

from __future__ import annotations

from dataclasses import dataclass

ACTIONS: tuple[str, str] = ("H", "L")

# Cheap-talk vocabulary: "I'll bid high", "I'll bid low", silence.
SIGNALS: tuple[str, str, str] = ("high", "low", "none")


@dataclass(frozen=True)
class PayoffParams:
    """The four payoff values. First subscript: own bid; second: all neighbours' bids."""

    uLH: float = 12.0
    uHH: float = 10.0
    uLL: float = 4.0
    uHL: float = 0.0

    def is_prisoners_dilemma(self) -> bool:
        """The ordering imposed in the paper: uLH > uHH > uLL > uHL and 2 uHH > uLH + uHL."""
        return (
            self.uLH > self.uHH > self.uLL > self.uHL
            and 2 * self.uHH > self.uLH + self.uHL
        )


DEFAULT_PARAMS = PayoffParams()


def u(action: str, f: float, params: PayoffParams = DEFAULT_PARAMS) -> float:
    """Payoff of bidding `action` when a fraction `f` of in-neighbours bid high."""
    if not 0.0 <= f <= 1.0:
        raise ValueError(f"f must lie in [0, 1], got {f}")
    if action == "H":
        return params.uHL + f * (params.uHH - params.uHL)
    if action == "L":
        return params.uLL + f * (params.uLH - params.uLL)
    raise ValueError(f"action must be 'H' or 'L', got {action!r}")


def frac_high(n_high: int, d: int) -> float:
    """f = n_high / d for an agent with in-degree d >= 1."""
    if d < 1:
        raise ValueError(f"in-degree must be at least 1, got {d}")
    if not 0 <= n_high <= d:
        raise ValueError(f"n_high must lie in [0, d] = [0, {d}], got {n_high}")
    return n_high / d
