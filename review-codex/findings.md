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

## Pending

Llama primary and temporal replication; Monte Carlo/table uncertainty; theorem and
support review; gate chronology; claim table; related work; final branch refresh.
