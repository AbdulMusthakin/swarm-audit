# Overnight core plan (8 Nov 2026): frozen before any follow-up result is computed

Source hand-off: `01_Core_Overnight_Handoff_2026-11-08.md` (independent, uploaded by Abdul at
about 00:50 UTC). Abdul's instructions with it: evaluate the hand-off and follow the evaluated
version; core work has no deadline, non-core work ends at about 10:00 UK (10:00 UTC); the
spending cap may be raised to $60 if suitable.

Everything below is **exploratory**: chosen after the original results were seen. This file
was written and committed before any number in it was computed, so the choices after this
point are prospective. It does not preregister the project retroactively.

## Evaluation of the hand-off

I checked its premises against the repository before adopting it.

| Hand-off statement | Check | Disposition |
|---|---|---|
| Primary D = -0.00008644, conditional 95% interval [-0.00022103, 0.00003532] | `results/analysis/core_cheap_talk/cheap_talk_n10_cycle_full.json`, `scores.paired_D`: identical | adopt |
| Maximum binned TV 0.26145 / 0.28855 against simultaneous null thresholds 0.23758 / 0.23536 | `scores.delta`, `scores.floor.*.max_over_rounds`: identical | adopt |
| Defect shift +0.01078 = +0.00985 within-state + 0.00093 state-frequency | `defect_shift.actions_only.pooled`: identical | adopt, as a decomposition only |
| Fitting histories `fit` / `fit_L` are stored | `cheap_talk_d2_signal.json`: 108 states x 50 fit (6-vector each) + 25 score, with lengths; START has 1 | adopt; C2 is feasible on existing data |
| "Existing three-consecutive-round consensus definition" for OLMo | `stats.consensus`: first t <= T-2 with X_t = X_{t+1} = X_{t+2} in {0, 1}; no consensus is censored at T | adopt unchanged |
| OLMo and n = 100 results exist | `results/extra/` (merged at `c171bec`) | read-only; numbers are imported with the extra arm's labels |
| $40 cap, workspace `aliyusufbelal` | Abdul now permits up to $60. About $6 recorded so far | keep $40 unless a projection needs more; never above $60 |
| 08:00 UK delivery target | Abdul set 10:00 for non-core work | use 10:00, keep the hand-off's order and reserve the last hour for the bundle |
| Related work (three arXiv papers) | not yet verified | verify each from its arXiv page; anything not retrievable is marked unverified, never paraphrased from the hand-off |

Additions and deviations:

1. **Core is unchanged and first.** Remaining core steps: cheap talk n = 20 runs 25-49, the
   n = 20 analyses, then arm 7 (second topology K_10, both channels, with its d = 9 audit
   tables) as planned in `results/serving/g4_plan.json`. The arm 7 GPU step is split into
   counts and cheap talk to stay under the 2-hour job limit. No kernel, prompt or serving
   change.
2. **Simulation-noise rule (precision target, fixed now).** See C1.3.
3. **C2 optional GPU top-up: decided below, with a fixed budget**, launched only if the stated
   conditions hold.
4. Secondary conditions (counts n = 10; n = 20 both channels once complete) get the same C1
   temporal scores with the same code; they are labelled secondary.

## C1: what endpoint scoring misses (primary follow-up condition)

**Data.** `results/runs/core_cheap_talk/cheap_talk_n10_cycle_full` (50 runs, Llama 3.1 8B,
cycle C_10, full history). Tables `results/audit/llama-3.1-8b/cheap_talk_d2_{signal,actions,
actions_equal}.json` (the original audit; never refitted on swarm data).

**Predictors.** Exactly those of `analysis.build_predictors`: order-zero, actions-only
(derived, primary weights), actions-only (equal weights), signal-aware. The reproduction uses
the original seed (`analysis.seed` 7, predictors seeded 8, 20,000 runs each), and must
reproduce the published D to machine precision before anything else is reported.

### C1.1 Metrics

- **Endpoint (unchanged):** CRPS of y (high bids in rounds 16-20, normalised by 5n), paired
  D = CRPS(actions-only) - CRPS(signal-aware).
- **Round-block CRPS of X_t.** For round t and predictor m, G_t is the CDF of the high count
  c_t = n X_t over {0, ..., n} from m's simulated runs. CRPS_{r,t} = (1/n) sum_{k=0}^{n}
  (G_t(k) - 1[c_{r,t} <= k])^2, the same normalisation as the endpoint score. Per run, average
  over the rounds of a block, then average over runs. Blocks: **2-7 (early; the follow-up's
  primary temporal score)**, 8-13 and 14-20; all three are reported. Paired differences per
  block: actions-only - signal-aware (both weightings), and each audit predictor minus
  order-zero.
- **Binned TV with its simultaneous null:** reproduced from the original analysis, not changed.
- **Signed and absolute one-step error.** Per logged decision, e = p_H - kappa(H | s): the
  actions-only (derived, primary) table on the actions-only state, and the signal-aware
  table's action marginal on the signal-aware state. Means of e and |e| by round (mean over
  agents), and by coarse actions-only state (own bid x neighbours' high count: 6 states) within
  each round block. These are teacher-forced (one step, the swarm's own histories), not group
  forecasts.
- **Intervals:** 95% percentile intervals from 2,000 resamples of whole runs (seed 20261108).
  Never decision-level resampling. Means and intervals are plotted for every round, not only
  the rounds that differ.

### C1.2 OLMo (read-only, descriptive)

From `results/extra/analysis/olmo_core_cheap_talk/cheap_talk_n10_cycle_full.json` (the extra
arm's own analysis): consensus survival curves of the swarm and each predictor under the
definition above, with censoring, and the round-by-round mean X_t. Descriptive only, labelled
"extra arm, imported". No new OLMo computation that the extra agent owns.

### C1.3 Monte Carlo sensitivity (fixed now)

- 10 independent simulation batches of 20,000 runs for each predictor, batch b seeded
  `SeedSequence([20261108, 101, b])`, b = 0..9. Report per-batch D (endpoint and early), their
  mean and SD.
- **Materiality rule:** simulation noise is material if the SD of the per-batch endpoint D
  exceeds 10% of the half-width of the original conditional interval
  ((hi - lo) / 2 = 0.000128; threshold 0.0000128). If material, the high-precision estimate is
  the D computed from all 10 batches pooled (200,000 runs per predictor), whose simulation SD
  is the batch SD / sqrt(10); if that still exceeds the threshold, one further run at
  n_model = 20,000 x ceil((batch SD / threshold)^2), capped at 2,000,000. The sign is reported
  whatever it is. No other reruns.

### C1.4 Audit-estimation (table) sensitivity

- 200 table resamples, resample j seeded `SeedSequence([20261108, 202, j])`. In each, for every
  non-START signal-aware state, resample its 50 fitting histories with replacement, keeping
  each six-outcome vector intact; kappa' is their mean. START (one deterministic query) is
  fixed. The actions-only tables (primary and equal weights) and order-zero are derived from
  that same resampled signal table, as in the original pipeline, so their dependence is kept.
  Scoring histories are not used.
- Each resample's predictors are simulated with 20,000 runs (seed tied to j) and scored on the
  same 50 swarm runs. Reported: the distribution of D (endpoint and early) over resamples
  (2.5% / 97.5% percentiles, SD). This contains simulation noise as well; the table-only SD is
  estimated as sqrt(SD_resample^2 - SD_MC^2) with SD_MC from C1.3, and labelled an
  approximation.
- **Nested interval** (if the timing benchmark allows): each table resample is paired with one
  whole-run bootstrap resample of the 50 swarm runs (seeded with j), giving an interval that
  includes run sampling, table estimation and simulation. It is labelled exactly so. It does
  not include kernel drift or prompt/model choice, and is not "total uncertainty".
- Enumerated last-round tables are not bootstrapped (they have no fitting uncertainty).

**Acceptance:** one script, `scripts/followup_core_temporal.py`, reproduces the primary D,
computes the temporal scores and both sensitivities, and logs any discrepancy with the
published numbers in its output before conclusions are drafted.

## C2: a clock-indexed audit (exploratory repair)

**Idea.** The original table averages synthetic histories of lengths L = 1..19. At round t a
full-history agent has L = t - 1. The clock-indexed audit uses the table for the matching
length block, a time-inhomogeneous composed chain. It is still an autonomous audit forecast:
no swarm history, observed fraction or swarm-fitted weight enters it.

**Fixed design.**

- Condition: Llama, cheap talk, n = 10 cycle, full history (the primary condition).
  Secondary control: counts n = 10 (its fitted counts table, actions-only), same code.
- Blocks: L in {1..6}, {7..12}, {13..19}; START separate (unchanged). At round t >= 2 the
  block containing L = t - 1 is used, so rounds 2-7, 8-13 and 14-20 use blocks 1, 2 and 3
  (they coincide with the C1 round blocks).
- Signal-aware clock table: for each state and block, kappa' is the mean of the `fit` vectors
  whose `fit_L` is in the block. Actions-only clock tables: derived from each block's signal
  table with the original p_test weights (primary and equal). Counts: mean of the counts table's
  `fit` p_H within the block.
- Empty cell (no fitting history of that state in that block): use the original pooled entry;
  count and report every use. Per state x block fitting counts are reported. No shrinkage, no
  tuning on swarm outcomes.
- Alignment check: `len(fit) == len(fit_L)` for every entry, else this path stops.
- Per-block eps(p_test) of the clock table and the pooled table on the scoring histories in
  that block (`score`, `score_L`), so the like-for-like test-distribution error is shown.
- **Primary C2 metric:** paired early-round CRPS improvement (pooled - clock; positive favours
  the clock audit), for the **signal-aware** predictor. Secondary: the same for actions-only
  (both weightings), the other blocks, the endpoint CRPS, and the counts control. Intervals as
  in C1 (whole runs, conditional on the tables), plus the C1.3 simulation batches and, if time
  allows, 200 table resamples within state x block.
- These are the same 50 runs the mismatch was seen on, so results are exploratory and not an
  independent validation. A null or a deterioration is reported as such. Matching the clock
  does not remove other differences between synthetic and swarm histories within a block.
- **Round-2 sensitivity (separately labelled):** at round 2 only, use the enumerated
  last-round table, if the L = 1 full-history prompt and the last-round prompt are shown to be
  byte-identical (checked by rendering both locally); otherwise not done.

**Optional GPU top-up (budget fixed now).** Only if all hold: arm 7 has finished by 04:30 UTC;
the projection (from the G4 benchmark rate) finishes by 07:30 UTC; the projected spend keeps
the ledger under $40. Design: for every signal-aware state and each block, 50 new fitting and
25 new scoring histories with L uniform within the block, from a new seed tag; the clock table
uses each block's new histories, and the pooled comparator is fitted from exactly the same new
histories with block weights 6/19, 6/19, 7/19 (the p_test length distribution). Both are scored
on the new scoring histories and on the same 50 runs. If the conditions fail, the existing-data
result stands with its limitations.

## C3: evidence package

`results/followup/core/`: `summary.md` (at most two pages), `numbers.json`, `claims.csv`
(value, interval type, source file and key, source commit, run set, status),
`temporal_scores.csv`, `uncertainty.json`, `clock_audit.json`, figures (PDF + PNG),
`report_insert.md` (600-900 words), `reproduce.md`, and `related_work.md` (at most 30 minutes;
verified sources only). Headline numbers are re-derived by a separately written calculation in
`scripts/followup_core_check.py` that does not import `swarm_audit.stats`, `analysis` or
`prediction`; agreement with reused pipeline functions is called a consistency check, not a
replication.

Labels kept in every combined table: original primary / secondary / exploratory /
exact-model calculation; live / simulated; fitted / enumerated; core / extra arm (imported).

Preserved and kept distinct: the original failed cheap-talk G5 per-run test (p = 0.0076), its
investigation, Abdul's exception, and the passing confirmation.
