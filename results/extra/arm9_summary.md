# Arm 9: OLMo-3-7B. EXPLORATORY, a conditional second-model extension

**This arm is exploratory.** Its pre-registered trigger was **not met**. It ran because Abdul
decided on 7 Nov at 20:15 to run it after seeing results. It is a second-model extension
under one prompt, one payoff set and one topology. It is **not a replication** of the core
and not a confirmatory test.

**Status: complete.** 9a and 9b are each 50 of 50 runs, analysed. 9a's G5 passed and the
lock-in stop did not apply (42 of 50 runs all-low).
- **9a:** paired D +0.025 [−0.003, +0.054].
- **9b:** every run locked into all-low. Both audits forecast that end state but not its
  speed. Paired D is −0.0001, and its interval is degenerate.
- **Spend for arm 9:** about $2.60, including the OLMo serving checks.

## Trigger and decision

- **Rule.** The rule was fixed in commit `a2657c5` (about 19:00 UTC on 7 Nov), before any
  result. Arm 9 would run only if Llama's final-prompt signal responsiveness on the d = 4
  screening grid (`d4.signal_responsiveness_length1_mean`) was strictly below 0.05.
- **Value.** It was **0.0553** (the core's `results/serving/signal_responsiveness.json`,
  commit `846ce77`), so the trigger was **not met**.
  - The other readings were reported but did not decide: d = 2 length 1 gave 0.038, d = 4
    length 5 gave 0.022, and d = 2 length 5 gave 0.038.
  - Record: `arm9_trigger.json`.
- **Decision.** Abdul decided at 20:15 to run 9a, then 9b subject to the lock-in stop. He
  decided after seeing Llama's d = 2 responsiveness (0.038) and the core's arm 4b CRPS
  null. Because the choice followed those results, the arm is exploratory.
- **Rules kept from the hand-off.** The design, the $25 cap and the stop rules are
  unchanged:
  - G5 must pass on 9a.
  - 9b does not run if at least 45 of the 50 9a runs have every agent bidding low in
    rounds 16–20.

## Model and serving checks

| check | result |
|---|---|
| model | `allenai/Olmo-3-7B-Instruct`, revision `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`; chat template with no extra arguments |
| X1, set-up | weights cached in this workspace (14.6 GB). On the final prompts every label is a single token with OLMo's own ids: actions 473/445, signals 1579/3428/7000. `serving/olmo-3-7b/workspace_setup.json` |
| X2, repeatability | batch invariance on. 48 prompts were scored alone, in a shuffled batch of 448 others, and warm. Max difference **0** in P(H) and in every signal probability, so the floor is 0. `serving/olmo-3-7b/g3_batchinv1.json` |
| drift, each session | start against end of each GPU session: max difference 0 (in each condition's `drift.json`) |

Everything else is the core's: prompts, payoffs (12, 10, 4, 0), T = 20, p_test, and the
analysis settings. A test checks that `configs/extra-olmo.yaml` differs from the arm 8
config only in the model and the arms.

## 9a: calibration, n = 4 cycle, cheap talk, last-1 history, 50 runs

The core's arm 4b design on OLMo (arm_id 9). The tables are enumerated: 109 states, exact
for a last-1 history.

### What OLMo did

- **The table shows strong lock-in after a low bid.**
  - After a low bid, κ(H | ·) is 0.000, 0.000 and 0.008 with 0, 1 and 2 high neighbours.
  - After a high bid it is 0.11, 0.19 and 0.91.
  - These are action marginals, averaged over messages.
- **The swarm locked in, both ways.** Every run reached consensus.
  - **42 of 50** runs were all-low in rounds 16–20 and **8 of 50** were all-high. No run
    was mixed.
  - X_t falls from 0.66 in round 1 to 0.16 by round 4 and stays there.

### G5 (`gates/olmo-3-7b/G5_cheap_talk.json`): **PASS**

- **Per decision:** the 4,000 logged probabilities match the table exactly: max difference
  0 in P(H) and in every signal probability, at floor 0.
- **Per run:** the signal-aware chain's gap is 0.018, with a discrete Monte Carlo
  p = 0.84 (α = 0.01).
- **A finding, not a gate:** the actions-only chain does not reproduce the run outcomes
  (gap 0.160, p = 0.00005). It predicts almost certain all-low lock-in (mean y 0.007
  against the swarm's 3.2), and misses the 8 all-high runs.

### Lock-in stop (`arm9a_lock_in.json`): **does not apply**

42 of 50 runs were all-low in rounds 16–20, below the stop threshold of 45. So 9b may run.

### Forecasts (y = high bids in rounds 16–20 / 5n; whole-run 95% intervals)

| predictor | CRPS | E_swarm F − E_model F | max Δ_t (null, max over rounds) |
|---|---|---|---|
| order-0 audit | 0.245 [0.185, 0.306] | −0.038 [−0.138, 0.062] | 0.762 (0.230) |
| actions-only audit | 0.160 [0.060, 0.260] | +0.160 [+0.060, +0.260] | 0.160 (0.191) |
| signal-aware audit | **0.135** [0.063, 0.206] | +0.016 [−0.084, +0.116] | 0.094 (0.196) |

- **Paired D** (actions-only CRPS minus signal-aware CRPS; > 0 favours signals) is
  **+0.025 [−0.003, +0.054]**. It points towards the signal-aware audit, but the interval
  includes 0.
- **Consensus probabilities:**

  | | all-high | all-low |
  |---|---|---|
  | swarm | 0.16 | 0.84 |
  | signal-aware chain | 0.147 | 0.853 |
  | actions-only chain | 0.068 | 0.932 |

  The actions-only chain reaches all-high sometimes but leaves it again, so its late-round
  X goes to 0.
- **Held-out runs 26–50** (25 runs):
  - Signal-aware audit: CRPS 0.163.
  - Fitted-from-swarm baseline: 0.163. It uses swarm data and is not an audit.
  - Actions-only audit: 0.200.
  - Paired D: +0.037 [−0.009, +0.083].

**A lock-in outcome is not "nothing left to predict."** Whether a run locks in high or low
is itself the forecast. In 9a, the split between the two is what separates the
signal-aware chain from the actions-only one.

## 9b: core extension, n = 10 cycle, cheap talk, full history, 50 runs

Arm_id 91. The tables are sampled from p_test histories at the full sizes: 50 fit and 25
score histories per state, 8,102 histories, 66 minutes, about $0.88.
`arm9b_projection.json` projected the cost before the run; the reduced 25/15 sizes were
not needed. 9b ran only after its launch checked two records in code: 9a's G5 record
(PASS) and `arm9a_lock_in.json` (stop: false).

### What OLMo did

- **Every one of the 50 runs locked into all-low.**
  - X_t falls from 0.632 in round 1 to 0.136 in round 3 and 0.032 in round 5. It
    reaches 0 by round 12 in every run.
  - y = 0 in all 50 runs: no high bid in rounds 16–20.
- **Neighbouring agents almost always agree:** edge disagreement is 0.031 over rounds
  2–20.

### Forecasts (y = high bids in rounds 16–20 / 5n)

| predictor | CRPS | E_swarm F − E_model F | max Δ_t, round (null, max over rounds) |
|---|---|---|---|
| order-0 audit | 0.324 | −0.362 | 0.926, round 10 (0.233) |
| actions-only audit | **0.0000** | −0.002 | 0.474, round 3 (0.209) |
| signal-aware audit | 0.0001 | −0.008 | 0.482, round 3 (0.208) |
| fitted from swarm (runs 26–50 only) | 0.0000 | −0.007 | n/a |

- **Paired D** (actions-only minus signal-aware) is −0.0001. Both audits forecast the
  outcome, so there is no measurable incremental value of the signals in y. The tiny
  difference slightly favours the actions-only audit.
- **The bootstrap intervals collapse to points** because y is 0 in every run. They do not
  measure forecast uncertainty here, and this is not evidence of high precision.
- **The two audits differ in timing, not in outcome.** Both predict the end state well:
  - all-low consensus probability: signal-aware 0.981, actions-only 0.988;
  - late-round high-bid fraction: under 0.01.
- **Both audits are too slow, though.** In round 3 the swarm is at 0.136, the
  actions-only chain at 0.340 and the signal-aware chain at 0.353.
  - Δ_t peaks at 0.47–0.48 in round 3. That is far above the null for the maximum over
    rounds (about 0.21).
  - It is above the pointwise null in rounds 2–7 for both audits.
  - So the audits forecast the outcome but not the path: the swarm locks in about two
    rounds faster than the audited chains.
- **The order-0 audit** has no state dependence, so it forecasts a stationary 0.36 and
  fails badly.

### Local defects

| table | ε(p_test) | ε(p_swarm) | shift, pooled (rounds 2–7 / 8–13 / 14–20) | bound saturates |
|---|---|---|---|---|
| actions-only (binary TV) | 0.099 | 0.015 | −0.084 (−0.056 / −0.104 / −0.104) | round 2 (B_20 = 2.96) |
| signal-aware (six-outcome joint TV) | 0.247 | 0.202 | −0.044 (−0.004 / −0.066 / −0.067) | round 2 (B_20 = 40.4) |

- **Like-for-like one-step comparison** (P(H) only):
  - signal-aware: |p_H − κ′| = 0.0149;
  - actions-only: |p_H − κ| = 0.0148.
- **The swarm's defects are far smaller than the audit's test defects.** That is because
  the swarm spends most decisions in the all-low states, where OLMo is nearly
  deterministic. The two measures are not comparable across tables: one is a binary TV,
  the other a six-outcome joint TV.

### Reading 9b

- **It is a degenerate, predictable outcome.** "Lock-in" here does not mean "nothing left
  to predict": both audits forecast the end state correctly, and both get its timing
  wrong.
- **It is uninformative about the incremental value of signals.** When every run ends
  all-low, y cannot separate a signal-aware forecast from an actions-only one.
- **Low screening responsiveness did not make this outcome inevitable.** 9a, with the
  same model, ended all-high in 8 of 50 runs.

## Limitations

- **Exploratory.** It was chosen after seeing results, so nothing here confirms or
  replicates the core.
- **One second model**, with one prompt, payoff set and cycle topology. Low screening
  responsiveness does not make a signal effect impossible: accumulation over rounds and
  untested histories can still matter.
- **Small samples.** There are 50 runs per condition. 9a's outcome is nearly binary,
  so its CRPS intervals are wide. 9b's outcome is constant (y = 0 in every run), so its
  intervals are degenerate.
- **One draw of each audit table.** The sampled 9b table is a single draw from p_test.
- **The intervals are conditional on the fitted tables.** They leave out audit-estimation
  uncertainty.

## Files

- **Config:** `configs/extra-olmo.yaml`.
- **Decision records:** `arm9_trigger.json`, `arm9a_lock_in.json`, `arm9b_projection.json`.
- **Serving checks:** `serving/olmo-3-7b/`.
- **Audit tables:** `audit/olmo-3-7b/`.
- **Runs:** `runs/olmo_calibration_cheap_talk/` and `runs/olmo_core_cheap_talk/`.
- **G5 record:** `gates/olmo-3-7b/G5_cheap_talk.json`.
- **Analyses:** `analysis/olmo_calibration_cheap_talk/` and `analysis/olmo_core_cheap_talk/`.
- **Figures:** `figures/fig1_olmo_calibration_cheap_talk_…png` and `figures/fig1_olmo_core_cheap_talk_…png` (X_t, the bound and Δ_t).
