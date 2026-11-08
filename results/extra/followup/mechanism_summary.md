# Why the actions-only audit erases OLMo's persistent high bidding

**Exploratory.** Chosen after seeing arm 9a; the specification was frozen in `plan.json`
(commit `d9ec5fc`) before anything was computed. Setting: OLMo-3-7B, n = 4 cycle, cheap
talk, last-round history, T = 20.

- **F = 1:** every agent bids high in every round 16–20.
- **Source keys:** [A] `audit_ablations.json`, [O] `observation_interventions.json`,
  [L] `observation_live.json`, [S] `size_memory.json`, all in `results/extra/followup/`.
- **"Exact":** chain laws on the cached, validated kernel. A separately written script
  reproduces them to 1e-10 (`independent_check.json`). They carry no sampling uncertainty.

## What changes

- **Exact laws** [A variants]: P(F = 1) is 0.1424 under the full signal-aware table and
  0.000076 under the derived actions-only audit. In the live runs, 8 of 50 had F = 1.
- **Cost of the gap:** if the full law is the truth, the actions-only forecast loses an
  expected 0.0205 in CRPS. This uses the standard identity Σ_k (Q(k) − G(k))²/20.
- **Live runs:** the paired difference is +0.0252, 95% whole-run interval
  [−0.0034, +0.0596]. Fifty runs cannot separate the two forecasts on their own.

## Which information the audit needs (E1)

All four audits emit the same six-outcome (bid, message) vector from the same START. They
differ only in which messages they average out, using p_test's weights:

| audit keeps | P(F = 1) | expected excess CRPS |
|---|---|---|
| own and neighbour messages (full table) | 0.1424 | 0 |
| own last message only | **0.1373** | 0.00002 |
| neighbour messages only | 0.00007 | 0.0205 |
| neither (= actions-only) | 0.00008 | 0.0205 |

- **Own message only is enough.** It is the smallest tested variant that recovers the
  full-table forecast within the frozen tolerance [A smallest_recovering_variant].
  - Equal weights give the same conclusion (0.1354).
  - On the live runs its CRPS differs from the full table's by only +0.0002
    [−0.0008, +0.0013].
- **Neighbour messages alone recover nothing.**
- This is the smallest of the four tested variants, not universal sufficiency.

## The mechanism, in three links

**1. The table depends on the own message** [A kernel_grid_P_H.H-2]. Take an agent that bid
high, with both neighbours high:

- After its own "high" or "none" message, K(H) is 0.9987–0.9996, whatever the neighbours
  said.
- After its own "low" message, K(H) is 0.68–0.73.
- The audit weights "low" at 1/3. That gives κ(H | H-2) = 0.905 and raises the one-step
  exit from all-high from 0.0027 to 0.329 [A mechanism.exit_one_step_oracle].

From a common first-entry distribution, the probability of still being all-high after 19
rounds [A mechanism.survival_from_both_entry_distribution] is:

| audit | still all-high after 19 rounds |
|---|---|
| full table | 0.948 |
| own message only | 0.914 |
| actions-only | 0.0005 |

This is finite-horizon persistence, not absorption.

**2. The real process almost never occupies that state.**

- **Exact chain:** own last messages inside all-high profiles in rounds 2–20 are "high"
  43.9%, "none" 56.1% and "low" 0.010% [A mechanism.own_message_share_within_all_H_rounds_2_20].
- **Live runs:** none of the 608 messages sent in all-high rounds was "low".
- **Oracle diagnostic, not an audit:** weighting the slice by the chain's own occupancy
  gives P(H) = 0.9993.

**3. The process does not need the channel (E2).** Same kernel, but message fields of what
agents are shown are replaced by "none" [O processes]:

| agents shown | P(F = 1) |
|---|---|
| everything, as logged | 0.1424 |
| own message replaced by "none" | 0.1412 |
| neighbour messages replaced by "none" | 0.1409 |
| both replaced by "none" | 0.1413 |

- **The interaction is small:** 0.0016.
- **These are model-based counterfactuals**, not new live measurements.
- **Random replacement is a different intervention.** A uniform random message reproduces
  the averaged audits' laws exactly, to 1e-16 [O uniform_replacement_sensitivity], because
  it shows "low" one time in three.

**Supported claim:** this is an audit-weighting failure, not a necessary self-feedback
channel.

- Averaging the own message uniformly injects an own-"low" state.
- The process essentially never visits that state inside all-high, and in that state
  retention is low.
- That injected state supplies the escape probability that erases the persistent-high
  outcome.

All three links hold: the table dependence, occupancy along the reference process, and
the intervention.

## Where the high outcome comes from

- **Round 1 settles it.** 99.9% of P(F = 1) comes from an all-high round 1
  (0.6225⁴ = 0.150) followed by persistence (0.948) [A mechanism.round1_all_H].
- **Live runs agree:** all 8 all-high starts ended F = 1, and no other run did.

## Size against memory (E3, live, 50 runs per cell)

P(F = 1) with Clopper–Pearson 95% intervals [S cells]:

| | last-round history | full history |
|---|---|---|
| n = 4 | 8/50 [0.07, 0.29] | 5/50 [0.03, 0.22] |
| n = 10 | 0/50 [0, 0.07] | 0/50 [0, 0.07] |

- **Size matters, clearly so only under last-round memory.** From n = 4 to n = 10, P(F = 1)
  falls under both memories, and the nominal (unadjusted, as planned) Newcombe 95%
  intervals exclude 0 [S contrasts]:
  - last-round history: −0.16 [−0.29, −0.06]
  - full history: −0.10 [−0.21, −0.01]
- **Correcting for multiple comparisons** (post hoc, after the independent review) changes
  only the full-history contrast [S multiplicity_sensitivity]:
  - Bonferroni over the two size contrasts gives [−0.31, −0.04] (last-round) and
    [−0.23, +0.01] (full history).
  - Over all four contrasts, last-round history stays below 0 ([−0.33, −0.02]).
- **Memory shows no detectable effect:** −0.06 [−0.20, +0.08] at n = 4, and 0 at n = 10.
- **The forecasts, committed before the runs, agree:** 0.008 for n = 10 last-round; 0.112
  for n = 4 full history, approximated from the 9b table.
- **Round 1 decides F = 1 in every cell.** All cells share the START prompt, so
  P(all high in round 1) = 0.6225ⁿ: 0.150 at n = 4 and 0.0087 at n = 10.
  - The size effect is consistent with an all-high start becoming rare as n grows.
  - No n = 10 run started all-high, so persistence at n = 10 is untested.
- **Not shown:**
  - a phase transition (only two sizes were tested);
  - mediation of any memory effect by the E1/E2 mechanism;
  - joint randomisation of the cells (they come from different sessions).

## Controls and scope

- **Llama control:** P(F = 1) ≤ 7e-11 under all four variants, and excess CRPS ≤ 4e-6.
  Llama has no persistent-high outcome to lose, so the effect depends on the model.
- **Live E2 validation** [L]: 4 × 50 fresh runs, all passing.
  - F = 1 counts were 9, 4, 7 and 8 against exact laws near 0.14.
  - Run-level p-values were 0.54, 0.23, 1.0 and 0.84, against a familywise α of 0.01.
  - All 16,000 logged probabilities equal the table entry for the transformed state.
  - This validates the implementation only. Every prompt is an enumerated one.
- **Limits:**
  - exploratory, with the analyses chosen after the results;
  - one game with frozen wording, one selected model, one topology and one horizon;
  - last-round history except in E3;
  - "none" is meaningful content, not a neutral token;
  - links 1 and 2 rest on the cached kernel and an oracle occupancy;
  - 50 runs per live cell.
- **Not claimed:** intention, commitment or collusion. The measured concept is dependence
  on own-message history.
