# Follow-up summary: what the endpoint score misses, and a clock-indexed audit

8 Nov 2026, core session. **Everything below the primary result is exploratory.** It was
chosen after the original results were seen, under the plan frozen in
`docs/overnight-core-plan.md` (commit `d6ffc49`) before any follow-up number was computed.
Every number is listed with its exact source in `claims.csv` and `numbers.json`.

## Status

- **Core:** complete (03:45 UTC). Gates G1–G6 are done. Every planned arm ran, including
  n = 20 and arm 7 (K_10, both channels), and all of them are analysed. See the completion
  note in `results/README.md`.
- **G5 history is unchanged and kept distinct:** the cheap-talk per-run test failed
  (p = 0.0076); it was investigated (sampling, wiring and end to end all pass); Abdul accepted
  it as chance; the confirmation on fresh seeds passed. A later pass does not erase the
  failure (`results/gates/`).
- **Follow-up:** C1 (temporal diagnosis and uncertainty) and C2 (clock-indexed audit) are
  done for the primary condition, with secondary runs on counts and on the extra arm's
  existing OLMo runs. An independent re-derivation agrees (`independent_check.json`).

## 1. The primary result is a null, and it survives every uncertainty layer

Llama 3.1 8B, cheap talk, cycle n = 10, full history, 50 live runs. The primary contrast
D = CRPS(actions-only) − CRPS(signal-aware) of the late outcome y is
**−0.000086 [−0.000221, +0.000035]** (95%, whole-run bootstrap, conditional on the tables and
the forecast simulation). The new checks:

- **Simulation noise is material.** At the original 20,000 simulated runs per forecast, the
  SD of D over 10 independent batches is 7.8e-5, as large as D itself. At the pre-set precision
  (740,000 runs), D = **−0.000085 [−0.000225, +0.000054]**.
- **Audit-table uncertainty.** We resampled fitting histories within each state, 200 times.
  Combined with run resampling and simulation in a nested interval, D = −0.000106
  [−0.00047, +0.000086]. This excludes kernel drift (measured 0) and the choice of prompt,
  model and payoffs, so it is not total uncertainty.
- **Independent re-derivation.** Separate code with its own chains and an energy-form CRPS
  agrees within simulation error (z = −0.2).

**Conclusion:** at the endpoint, conditioning the audit on messages neither helps nor hurts
detectably. The sign of D is not established.

## 2. The endpoint null hides a transient forecast failure

- **The early rounds.** Both audits forecast X_t ≈ 0.38 from round 2 onward. The swarm
  drops to 0.32 on average over rounds 2–7 (0.29–0.38) and drifts up to about 0.36 later.
- **Binned TV.** The maximum binned TV exceeds its simultaneous null: 0.261 / 0.289 against
  0.238 / 0.235 (original analysis).
- **Early-round CRPS (rounds 2–7, X_t on {0, 1/n, ..., 1}).** The actions-only audit beats
  the signal-aware one: D_early = **−0.00074 [−0.00103, −0.00043]** (740,000 runs). The nested
  interval is [−0.00135, −0.00022], and none of the 200 table resamples gave D_early > 0.
- **OLMo.** On the extra arm's existing runs the same direction is larger:
  D_early = −0.0074 [−0.0086, −0.0060].
- **Teacher-forced signed error.** On the swarm's own histories, p_H − κ(H|s) is −0.058
  [−0.062, −0.054] in rounds 2–7 (signal-aware) and about −0.02 later. The audit
  over-predicts high bids early, and the bias does not cancel round by round. It is in the
  same direction for both audits.
- **The bias is within states, not a state-mix artefact.** In rounds 2–7 the signal-aware
  signed error is negative in all six coarse states, from −0.039 to −0.091
  (`signed_errors_by_state.csv`). Later, the sign splits: high-bid states are near 0 or
  positive, low-bid states negative. So pooled late summaries partly cancel.
- **OLMo endpoint.** Every run ends all-low, so the endpoint cannot discriminate. The swarm
  still converges faster than forecast. Under the existing three-round consensus
  definition, with non-consensus censored, no consensus run has started by round 5 in 20%
  of runs, against 71% forecast (actions-only). The median start is round 4 for the swarm,
  against round 7 for both forecasts (Figure A1).

- **Replication at n = 20 (secondary condition, fresh runs).**
  - Early D is −0.00097 [−0.00119, −0.00075] at 2,000,000 simulated runs; none of the 200
    table resamples is positive.
  - Endpoint D, conditional on the tables, is −0.00031 [−0.00045, −0.00017].
  - Including table uncertainty, endpoint D is [−0.00068, +0.00008] (nested
    [−0.00081, +0.00008]), so its sign is not established.
  - At 20,000 runs the published n = 20 D (−0.00004) sat within simulation noise: the
    batch SD is 1.7e-4, twice that at n = 10.

## 3. Mechanism: synthetic-history length, then content

- **The candidate cause.** The audit pools synthetic histories of lengths 1–19. At round t
  a full-history agent has t − 1 past rounds.
- **The clock-indexed audit** fits one table per length block (1–6, 7–12, 13–19) from the
  same stored fitting histories. Round t uses the block containing t − 1. It takes no input
  from the swarm.
- **Length explains most of the early bias.** The clock table cuts the early teacher-forced
  bias by about two-thirds in both models: Llama −0.058 → **−0.020**, OLMo −0.047 →
  **−0.016**.
- **Early forecasts improve.** Early CRPS improves (pooled − clock): Llama **+0.0060
  [+0.0033, +0.0086]** (nested [+0.0028, +0.0083]); OLMo **+0.041 [+0.029, +0.051]**. The
  clock audit's held-out error on synthetic histories, eps(p_test), is lower in every block.
- **Round-2 check (separately labelled).** The L = 1 full-history prompt is byte-identical
  to the last-round prompt in all 108 states, so the enumerated last-round table is the
  exact round-2 kernel. Its round-2 CRPS is 0.093, against 0.097 for the clock table and
  0.108 for the pooled table: the clock table gets most of the way to the exact kernel.
- **It fails later on Llama.** Rounds 8–13: −0.0014 [−0.0022, −0.0006]. Rounds 14–20:
  −0.0015 [−0.0028, −0.0001]. Endpoint: **−0.0036 [−0.0070, −0.0002]**. The late
  teacher-forced bias grows to −0.039, against −0.020 for the pooled table.
- **A candidate explanation: content shift (not tested).** Swarm histories before the last round contain about
  35% high bids (Llama, rounds 8–20). p_test uses 50% by construction. Long synthetic
  histories therefore push P(high) up more than real ones.
- **Reading.** The pooled audit's adequate late-round fit is consistent with two errors
  offsetting: short histories over-weighted early, high-bid-rich histories late. Matching
  the clock removes one and exposes the other. This is an interpretation of a decomposition,
  not a causal test.
- **Replication at n = 20 (cheap talk, secondary).** The same pattern holds:
  - early bias −0.060 → −0.021 with the clock table;
  - late bias −0.022 → −0.042;
  - early gain +0.0089 [+0.0066, +0.0111];
  - endpoint −0.0083 [−0.0109, −0.0058];
  - swarm histories again about 35% high.
- **Counts control (Llama n = 10).** The pooled audit is unbiased there (+0.002), the clock
  table changes no score detectably, and swarm histories contain about 59% high bids.

## 4. The extra arm's mechanism (imported; `extra_import.md`)

- **Setting:** OLMo, n = 4, last-round history; exact chain laws on the cached kernel.
- **The finding:** the actions-only audit destroys a persistent all-high outcome.
  P(F = 1) is 0.1424 under the full table against 0.00008 under the actions-only audit;
  live, 8 of 50 runs had F = 1.
- **Why:** it weights the agent's own "low" message at 1/3. The process almost never
  visits that state inside all-high, and in that state retention is low. Keeping only the
  own message recovers the full table's forecast (0.137).
- **Same design question as ours.** Both results say the isolated tests' choice of history
  and observation distribution is where the audit loses information.
- **Labels:** these are exact model-based results plus live checks, exploratory, and
  owned by the extra session.

## 5. Scale (imported from extra arm 8)

At n = 100 on a random 4-regular graph (counts), the actions-only audit improves endpoint
CRPS over order-zero by +0.0028 [+0.0016, +0.0039], while the estimated bound is vacuous from
round 2. Size, degree and topology all change at once, so this is not a size contrast.

## Limitations

- C1 and C2 were chosen after the original mismatch was seen and are scored on the same 50
  runs, so they are exploratory.
- One prompt pair, one payoff set and one cycle topology were used, with 50 runs per
  condition.
- The OLMo analyses reuse the extra arm's runs and tables. They are not an independent
  OLMo study.
- **The cause of the late error is not established.** Clock conditioning changes only
  history-length conditioning. The late deterioration is **consistent with** a
  history-content shift, but no controlled content intervention was run. A length-matched
  content intervention, or fresh predictive validation, would test it. The early
  improvement is supported as an operational result; the causal attribution of the
  remaining error is not (Codex review M4).
- No new GPU histories were collected for C2. More histories from the same p_test would
  reduce fitting variance, but they cannot correct a misspecified history population, if
  that is the problem. Fitting variance does not explain the late deterioration: it holds
  in every table resample (0 of 200 favour the clock audit in rounds 8–13 or at the
  endpoint). The early gain is also outside the table-resampling interval.
- The related-work check could not read full texts (`related_work.md`).
- The toy procurement game exposes an audit failure mode. It is not a detector of real-world
  collusion, and high bidding alone does not show intentional coordination.
- **Dual use.** Better forecasts of how coordinated agents behave could serve someone trying
  to steer or exploit a population of agents, as well as an overseer. We report a
  diagnostic of forecast failure, not a method for inducing coordination.

## Checks, spend and jobs

- **Independent re-derivation** (`independent_check.json`, code that imports nothing from
  the pipeline) agrees on every checked headline:
  - signed errors, pooled and clock: exact;
  - endpoint D: z = −0.2;
  - early D: z = 1.3;
  - clock early gain: +0.00601 against +0.00596, z = 0.8.

  It shares inputs and definitions with the pipeline, so it is a consistency check, not a
  replication.
- **Independent Codex review** (`codex/independent-audit-20261108`) reproduced the core
  headline numbers and found no critical error. Its material points concern the core:
  simulation error, already addressed by C1, and the primary flag on the OLMo arm, now fixed
  in `analysis.is_primary`. Both are handled; see `codex_review_response.md`. The table-only
  SD of the endpoint D is poorly determined (0 to 6e-5 across estimates), so table
  uncertainty is not called negligible.
- **Related work.** The reviewer read Flint et al. in full: cached LLM policies over memory
  states, validated against direct generations. The overlap is high, so no "first" claims
  (`related_work.md`).
- **Tests:** `tests/test_followup_core.py` covers CRPS normalisation, block indexing, the
  clock-table partition and fallback, the clock chain reducing to the original chain, and
  resampling integrity.
- **Spend:** the follow-up used no GPU. The core ledger (`results/spend.json`) records
  $5.80 before the remaining core runs, with a $40 cap.
- **Jobs:** see the completion note in `results/README.md` for the final state.
