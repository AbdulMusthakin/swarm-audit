# Comparison after independent results were committed

The independent Part 1 and Part 2 reports were committed locally as `1a7fb83dfebc7b991d9eed4faf6260dd97a6594b` before opening any pipeline code, gate output, or analysis output. The later GitHub upload may give that snapshot a different commit ID; its file content is unchanged. `check.py` remains independent. `compare_build.py` is explicitly a subsequent comparison and imports the build's code.

## Calibration verdict: AGREE

- Gate gap difference: 7.77e-16.
- Exact mean difference: 0.
- Independent p = 0.007499625019; published gate p = 0.007599620019; absolute difference 0.000099995000, within Monte Carlo error.
- Independent null 99th percentile = 0.199442801802; gate = 0.197138423561. These are separate finite Monte Carlo samples of a discrete statistic; the p-value determines the stated gate rule.
- The full signal-aware and actions-only PMFs also match the build's recomputed exact laws to 0.

The gate **fails** in both implementations. Log/table, state wiring, and draw checks show no discrepancy. This confirms the calculation; it does not demonstrate that chance is the cause. A correct null can produce rare failures, and this one remains a failed prespecified test.

## Arm 4b CRPS verdict: AGREE with recomputed build functions

There is no published arm 4b CRPS analysis output in the checked source snapshot; its calibration analysis file contains G5 tests only. After committing our results, we recomputed both exact laws using `swarm_audit.prediction.exact_law`, then used the build's CRPS and bootstrap functions with the same bootstrap seed. Both per-run score arrays, the paired mean and the interval agree to 1e-12. This comparison is a recomputation, not a claim of agreement with a nonexistent published output.

Signal-aware mean: 0.058135348831; actions-only mean: 0.058152831916; paired difference: 0.000017483085; 95% interval: [-0.000081964704, 0.000115815161]. The interval spans zero, so these 50 calibration runs do not show a clear CRPS benefit from retaining messages.

## Core comparison: pending data

No `results/runs/core_cheap_talk/cheap_talk_n10_cycle_*/run_*.jsonl.gz` files or fitted core table pair were found in any fetched branch (main, core, build-handoff, hand-off-review, extra-arms, independent-check). No core score is reported.

The independent simulator supports the ten-agent cycle. Its n=10 path was checked against the build's sampler using the available calibration tables: {'signal': {'cdf_gap': 0.009799999999999975, 'combined_dkw99': 0.024477468306808164}, 'actions': {'cdf_gap': 0.00799999999999998, 'combined_dkw99': 0.024477468306808164}}. These checks are not core experiment results.

When the fitted tables and logs arrive, verify their names and run:

```sh
OPENBLAS_NUM_THREADS=1 python independent/check.py \
  --core results/runs/core_cheap_talk/cheap_talk_n10_cycle_FULL_DIRECTORY \
  --signal results/audit/llama-3.1-8b/cheap_talk_d2_signal.json \
  --actions VERIFIED_DERIVED_ACTIONS_TABLE_PATH
```

Each predictor uses 20,000 simulated runs of its last-round chain and denominator 50. Commit `part2_core.md` and `part2_core.json` before comparing with the build's numbers (target agreement about 0.001). No future monitoring or external team message has been configured.
