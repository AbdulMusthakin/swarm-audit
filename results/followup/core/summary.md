# Follow-up summary: what the endpoint score misses, and a clock-indexed audit

8 Nov 2026, core session. **Everything below the primary result is exploratory.** It was
chosen after the original results were seen, under the plan frozen in
`docs/overnight-core-plan.md` (commit `d6ffc49`) before any follow-up number was computed.
Every number is listed with its exact source in `claims.csv` and `numbers.json`.

## Status

- **Core:** gates G1–G6 done. The counts and cheap-talk n = 10 arms are complete and
  analysed, and counts n = 20 is complete. Cheap talk n = 20 (runs 25–49) and arm 7 (K_10, both
  channels) were still running when this was written; see the completion note in
  `results/README.md` for their final state.
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
- **OLMo endpoint.** Every run ends all-low, so the endpoint cannot discriminate. The swarm
  still converges faster than forecast: no consensus by round 5 in 20% of runs, against 71%
  forecast (imported, extra arm).

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
- **It fails later on Llama.** Rounds 8–13: −0.0014 [−0.0022, −0.0006]. Rounds 14–20:
  −0.0015 [−0.0028, −0.0001]. Endpoint: **−0.0036 [−0.0070, −0.0002]**. The late
  teacher-forced bias grows to −0.039, against −0.020 for the pooled table.
- **Why, most likely: content shift.** Swarm histories before the last round contain about
  35% high bids (Llama, rounds 8–20). p_test uses 50% by construction. Long synthetic
  histories therefore push P(high) up more than real ones.
- **Reading.** The pooled audit's adequate late-round fit is consistent with two errors
  offsetting: short histories over-weighted early, high-bid-rich histories late. Matching
  the clock removes one and exposes the other. This is an interpretation of a decomposition,
  not a causal test.
- **Counts control (Llama n = 10).** The pooled audit is unbiased there (+0.002), the clock
  table changes no score detectably, and swarm histories contain about 59% high bids.

## 4. Scale (imported from extra arm 8)

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
- Clock conditioning does not address history-content shift. A content-matched audit (for
  example, synthetic histories whose bid rate follows the forecast) is the obvious next test.
- No new GPU histories were collected for C2. The late failure is a content effect, which
  more histories of the same p_test would not fix. The early gain is outside the
  table-resampling interval.
- The related-work check could not read full texts (`related_work.md`).
- The toy procurement game exposes an audit failure mode. It is not a detector of real-world
  collusion, and high bidding alone does not show intentional coordination.
