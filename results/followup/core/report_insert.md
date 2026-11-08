# Report insert (methods and results prose for the authors to adapt; about 850 words)

All numbers are traceable through `claims.csv`. Labels to keep: "pre-specified primary"
for the endpoint contrast; "exploratory" for everything in the second and third paragraphs
of Results.

## Methods (additions)

**Temporal scoring.** Besides the pre-specified endpoint score (the CRPS of y, the share of
high bids in rounds 16–20), we score each forecast round by round. For round t, a
predictor's forecast of the high count nX_t is the empirical distribution over its simulated
runs. The CRPS of each observed run uses the same normalisation as the endpoint score. We
average the CRPS over rounds 2–7 ("early"), 8–13 and 14–20 within each run, then over runs.
All intervals resample whole swarm runs. We also report the teacher-forced signed one-step
error p_H − κ(H|s), averaged over the swarm's logged decisions. Here p_H is the model's logged
probability and s the agent's coarse state. This measures bias on the swarm's own
histories, not a group forecast.

**Uncertainty layers.** The published interval for the paired contrast D is conditional on
the fitted audit tables and on one forecast simulation of 20,000 runs. We separate three
further sources:

1. **Simulation:** ten independent simulation batches of 20,000 runs each. A pre-set rule
   increases the simulation size if the batch-to-batch SD exceeds 10% of the interval's
   half-width.
2. **Audit estimation:** 200 resamples of the fitting histories within each state, keeping
   each six-outcome probability vector intact. Derived actions-only tables are recomputed
   from each resampled signal-aware table.
3. **A nested interval:** each table resample is paired with one whole-run resample. It
   excludes kernel drift (measured at 0) and the choice of prompt, model and payoffs.

**Clock-indexed audit (exploratory).** The original audit pools synthetic histories of
1–19 past rounds into one table per state. At round t, however, a full-history agent has
exactly t − 1 past rounds. The clock-indexed audit fits one table per length block (1–6,
7–12 and 13–19 past rounds) from the same stored fitting histories. At round t ≥ 2 it uses the
block that contains t − 1, giving a time-inhomogeneous composed chain. It uses no swarm data,
so it remains an autonomous audit forecast. Every state-block cell had at least 9 fitting
histories, so the pre-set fallback to the pooled entry was never used. We compare the
clock-indexed and pooled audits on the same runs with paired CRPS differences. The primary
follow-up metric is early-round CRPS for the signal-aware audit.

## Results

**The primary contrast is null.** On 50 live runs of Llama 3.1 8B (cheap talk, a cycle of
10 agents, full history), the pre-specified contrast D = CRPS(actions-only) −
CRPS(signal-aware) is −0.000086 (95% interval −0.000221 to +0.000035).

Simulation noise is not negligible at this scale. The SD of D across simulation batches
(7.8 × 10⁻⁵) is as large as D itself. At the pre-set precision of 740,000 simulated runs, D is
−0.000085 (−0.000225 to +0.000054). The nested runs × tables × simulation interval is
−0.00047 to +0.000086. Code written separately from the pipeline reproduces D within
simulation error. Conditioning the audit on messages therefore neither helps nor hurts the
endpoint forecast detectably, and its sign is not established.

**The endpoint null hides a transient failure (exploratory).** Both audits forecast that
about 38% of agents bid high from round 2 onward. The swarm averages 32% over rounds 2–7 and
rises to about 36% later (Figure 1b). Accordingly, the teacher-forced signed error is −0.058
(−0.062 to −0.054) in rounds 2–7 and about −0.02 afterwards. The audits over-predict high
bidding early, and the error does not cancel round by round.

In early-round CRPS the actions-only audit forecasts better than the signal-aware audit:
D_early = −0.00074 (−0.00103 to −0.00043). The nested interval is −0.00135 to −0.00022, and
none of the 200 table resamples reversed the sign. The roundwise binned total-variation
distance also exceeds its simultaneous null (0.26 and 0.29, against 95% thresholds of 0.24).

In the extra arm's OLMo-3-7B runs every run ends all-low, so the endpoint cannot
discriminate. The swarm still converges much faster than forecast: by round 5 only 20% of
runs lack consensus, against 71% forecast.

**History length explains most of the early bias; content explains the rest (exploratory).**
Matching the audit's history length to the round cuts the early teacher-forced bias by about
two-thirds: from −0.058 to −0.020 for Llama, and from −0.047 to −0.016 for OLMo. Early-round
CRPS improves by +0.0060 (+0.0033 to +0.0086) for Llama and by +0.041 (+0.029 to +0.051) for
OLMo (Figure 3).

The repair is not uniformly better. For Llama it is worse in rounds 8–13 and 14–20, and at
the endpoint (−0.0036; −0.0070 to −0.0002). Its late teacher-forced bias grows to −0.039,
against −0.020 for the pooled audit. The likely reason is a shift in history content.
Before the last round, the swarm's histories contain about 35% high bids, whereas the
synthetic histories contain 50% by construction. Long synthetic histories therefore raise
the model's propensity to bid high more than real histories do. The pooled audit's
acceptable late-round fit is consistent with these two errors offsetting each other; this is
our interpretation, not a causal test.

On the counts channel, the pooled audit is unbiased (+0.002) and the clock-indexed audit
changes no score detectably.

**Interpretation.** An isolated audit can match a swarm's endpoint while misforecasting its
transient. Endpoint scores alone are therefore not evidence of adequacy. Comparing
history-length-matched and pooled audits is a cheap diagnostic: it separates the
information an audit loses through history length from the information it loses through
history content. Repairing the content shift needs synthetic histories whose composition
tracks the forecast, which we leave for future work. These results come from one toy
procurement game, one prompt pair and one topology. They expose a failure mode of audits;
they do not detect collusion.
