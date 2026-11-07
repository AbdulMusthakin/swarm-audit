# Audit tables and p_test

Tables are fitted on synthetic histories shown to one agent alone (`swarm_audit/audit.py`,
`scripts/run_audit.py`). Swarm data never fits them, except the fitted-from-swarm baseline,
which is labelled as such.

## p_test — FLAGGED FOR ABDUL: this is a design choice and he may change it

Parameters: `p_test` in `configs/core.yaml` (identical in every build config).

- **Length.** A history has L past rounds, L uniform on {1, …, 19}. START (round 1) is
  the single empty-history prompt. So every length 0, …, T − 1 has positive probability,
  as Definition 2 requires.
- **Earlier rounds.** The agent's own action is uniform on {H, L}. Each in-neighbour's
  action is independently uniform, so the number bidding high is Binomial(d, 1/2) (not
  uniform on 0..d, as in the screening fillers). Each round's payoff is computed from its
  own action and counts with `game.u`, so test prompts have the swarm's format exactly.
- **Messages (cheap talk).** Every message not fixed by the target state is uniform over
  the three signals, the agent's own included, so neighbour signal counts are
  Multinomial(d; 1/3, 1/3, 1/3). This also holds in the last round when the state is
  actions-only, so an actions-only table averages over last-round messages.
- **The last round** is set to the target state.

Equivalently, p_test is "L uniform on {0, …, 19}, every entry of every round uniform", and
each state's sets are draws from it conditioned on the state. As a distribution over
states it gives START weight 1/20 and state s = (a, k [, signal counts c, own signal m])
weight 19/20 × 1/2 × Binom(d, 1/2)(k) [× Multinom(d; 1/3)(c) × 1/3]. These weights are used
for ε(p_test) pooled over states, the order-0 baseline κ₀ and the defect shift.

## Sizes (two disjoint sets per state, from different seeds)

| table | states (cycle, d = 2) | fit | score | histories | label read-outs |
|---|---|---|---|---|---|
| actions-only, counts | 6 + START | 100 | 50 | 902 | 902 |
| signal-aware, cheap talk (own signal shown) | 108 + START | 50 | 25 | 8,102 | 24,306 |
| actions-only, K_10 (arm 7 only) | 20 + START | 100 | 50 | 3,002 | 3,002 or 9,006 |

START is one prompt in each set. On cheap talk every history costs three read-outs: the
action, and the message after each action.

## Tables

- **κ(H | s)**: the mean logged p_H over the fitting histories in s.
- **κ′(a, m | s′)**: the mean of P(a | h) · P(m | h, a).
- **Actions-only on cheap talk** is derived, not queried:
  κ(H | s) = Σ_{s′→s} w(s′ | s) κ′(H | s′), with w from p_test (primary: multinomial
  counts, uniform own signal) or equal on each signal configuration (sensitivity, the
  teammates' choice). Its ε(p_test) is scored on the signal-aware scoring set with the
  same weights.
- **Calibration (arms 4, 4b)**: last round only, so the prompt is a function of the state;
  each state is queried once (`--enumerated`).

Each JSON file stores per state: κ, n, its standard error, ε(p_test), and every per-history
probability on the fitting and scoring sets, with history lengths.

## Files

- `results/audit/<model>/<channel>_d<d>[_enumerated]_{actions,signal,actions_equal}.json`:
  the LLM tables (none yet: no GPU run has happened).
- `results/harness/audit/`: the scripted-policy tables of gate G2.
