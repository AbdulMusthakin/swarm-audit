# Diagnostic sensitivity within the four-agent OLMo setting: pre-registered plan

Written 8 November 2026, before any simulation for this comparison was run. Exploratory,
CPU only, model-based. It asks whether a differently calibrated roundwise check, or a
persistence-event test, flags the actions-only audit where the shared maximum-TV cutoff
does not, without a higher false-positive rate. It does not change the original OLMo
checks or the prespecified Llama analysis, which stay as reported.

## Setting and models

- OLMo-3-7B, 4 agents on a cycle, cheap talk, last-round history, T = 20 rounds.
- Cached tables: `results/extra/audit/olmo-3-7b`, tag `cheap_talk_d2_enumerated`
  (the tables of extra arm 9a; kernel fingerprint as in its analysis).
- **Null H0:** the exact actions-only chain built from the derived actions-only table.
- **Alternative H1:** the exact signal-aware chain built from the signal-aware table. It is
  a model-based alternative, so every sensitivity estimate is conditional on these kernels.
- **Batch:** R = 50 whole 20-round trajectories, drawn independently from one chain, as in
  one live cohort.
- **Reference law** for every TV statistic: the exact binned law of X_t (five bins) under
  the actions-only chain, as in the original analysis.

## Statistics and decision rules (each at nominal 5%)

1. **S1, shared maximum TV.** M = max_t Delta_t, with Delta_t the binned TV between the
   batch and the reference law in round t.
   - S1a: reject if M > 0.19064822871734904, the existing simultaneous cutoff of the
     original analysis, applied unchanged.
   - S1b: reject if M > c_M, where c_M is the smallest calibration value v with
     (share of calibration set A batches with M > v) <= 0.05.
2. **S2, per-round tail, calibrated jointly across rounds.** For round t and observed
   Delta_t = x, the tail probability is p_t(x) = (1 + #{set A batches with Delta_t >= x})
   / (1 + N_A). The statistic is m = min_t p_t(Delta_t). Reject if m <= c_m, where c_m is
   the largest value v with (share of calibration set B batches with m <= v) <= 0.05.
3. **S3, fixed persistence event.** K = number of the 50 trajectories in which every agent
   bids high in every round 16-20. Under H0, K ~ Binomial(50, pi0), with pi0 the exact
   actions-only P(F = 1). Reject if K >= k*, the smallest integer with
   P_H0(K >= k*) <= 0.05.

## Simulation sizes and seeds

- Calibration set A: 20,000 null batches (per-round tail tables for S2; c_M for S1b).
- Calibration set B: 20,000 null batches (c_m for S2).
- Evaluation: 20,000 null batches and 20,000 alternative batches, independent of A and B.
- Seeds: SeedSequence(20261108) spawned into four independent streams (A, B, null
  evaluation, alternative evaluation), one per set, in that order.

## What is reported

- Cutoffs c_M, c_m and k*, and each test's exact or calibrated nominal level.
- For each test: the achieved false-positive rate on the null evaluation batches and the
  rejection rate on the alternative batches, each with a Clopper-Pearson 95% interval.
- One test is called more sensitive than another only if their alternative rejection-rate
  intervals do not overlap and both achieved null rates have upper interval limits at or
  below 0.06.
- As illustrations only, not as tests of the method: M, m, K and each decision for the
  two live cohorts already analysed (original arm 9a runs and the fresh unmodified E2 runs),
  using the same cutoffs.

## Scope

Model-based: the null and alternative are the cached audit chains, not the live process.
The live cohorts were seen before this plan was written, and the plan was motivated by the
original miss. Results speak to the sensitivity of the three checks under these kernels,
not to other settings.
