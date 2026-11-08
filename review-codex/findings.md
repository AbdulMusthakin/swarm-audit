# Independent audit — initial checkpoint

Review branch: `codex/independent-audit-20261108`; base `982822f554e547cc5c4468b70d5aaaae88356814`.
No remote writes, paid compute, credentials, messages, original-data changes or merges.

## Verified: OLMo four-agent last-round cheap talk

Source: extra-arms `4a817051abffbdf208eb38a2b09160ab9d49aa0f`,
`results/extra/audit/olmo-3-7b/cheap_talk_d2_enumerated_{signal,actions}.json`
and 50 logs in `results/extra/runs/olmo_calibration_cheap_talk/cheap_talk_n4_cycle_last1/`.
A new implementation with reversed state/label ordering and no project imports agrees:
P(all high in each round 16–20) = 0.14239254032336024 (signal-aware),
0.0000763560316673361 (actions-only). Observed: 8/50.
Paired CRPS D = 0.0252262741; whole-run percentile interval [-0.003426035, 0.053878583].
Expected CRPS improvement under the signal-aware law is 0.02052404609.
This is an exact-model calculation conditional on the enumerated, stationary last-round
kernel and independent per-agent sampling; it is not a high-precision live effect estimate.
Full machine-readable laws and integrity checks: `independent/olmo_exact.json`.

## Material interpretation to preserve

Synthetic averaging over messages creates a different dynamical process. Own-message
retention alone recovers P(F=1)=0.13733945; neighbour retention alone gives 0.00006946.
Replacing both observations by `none` gives 0.14125134, close to the full 0.14239254.
Thus `none` is not equivalent to marginalising over messages. These are cached-kernel
counterfactuals, not fresh live experiments or evidence of intentional coordination.

## Priority 1 checkpoint: Llama and calibration independently reproduced

From main `982822f554e547cc5c4468b70d5aaaae88356814`, all saved Llama n=10 predictor
CDFs and all per-run paired scores reproduce exactly (maximum error 0). Rebuilt graph
records, states, action and signal draws, and deterministic seed streams have 0 errors
across 10,000 decisions. Original endpoint D=-0.0000864399, archived conditional interval
[-0.0002210329, +0.0000353221]. It remains a null endpoint result.

**Material: simulation variability was not negligible for this contrast.** Twenty
independent 20,000-run batches give endpoint D SD=0.0000539702, versus whole-run
SE=0.0000675245 for the original conditional comparison. Pooling 400,000 runs per
predictor gives D=-0.0000912341 and conditional interval [-0.000233911,+0.0000479695];
its remaining MC SE is approximately 0.0000120681. Do not attach scientific meaning to
the tiny sign or treat the original interval as including simulation uncertainty.

Independent 200 paired table resamples (intact six-vectors, actions rederived, START
fixed) give endpoint SD including MC=0.0000790166 and table-only SD approximately
0.0000577134. A nested run/table/MC sensitivity interval is [-0.000397476,+0.000088647].
This is exploratory sensitivity with 200 replicates, not a high-accuracy total interval.

The temporal discrepancy also reproduces: maximum binned TV=0.26145 (actions),
0.28855 (signals), both round 2. Independent maximum-over-20-round null thresholds
are about 0.23640/0.23395. These checks reject those fixed-chain timing predictions,
not the endpoint comparison and not a general statement about swarm safety.

Original G5 failure p=0.00759962, confirmation p=0.74361282 and pooled p=0.08399580
reproduce exactly under the archived seed and 20,000 null simulations. The original
failure and authorised exception remain part of the result.

Machine outputs: `independent/llama_primary.json`, `independent/secondary_checks.json`.

## Pending

Theorem/support proof; final severity table and claim table; verified related-work
comparison; final one-time branch refresh. Newer uncommitted jobs were not inspected.
