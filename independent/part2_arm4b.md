# Independent arm 4b CRPS comparison — before pipeline comparison

Uses the fixed signal-aware exact CDF from Part 1 and an independently enumerated 16-profile actions-only chain, with the supplied `cheap_talk_d2_enumerated_actions.json` table. Both carry the count in rounds 16–20. Scores sum over k = 0,…,20 and divide by 20.

- Signal-aware mean CRPS: 0.058135348831.
- Actions-only mean CRPS: 0.058152831916.
- Mean paired difference (actions-only minus signal-aware): 0.000017483085.
- 95% percentile interval: [-0.000081964704, 0.000115815161].
- 2,000 bootstrap resamples of the 50 whole runs; seed 2026110703; predictors fixed; numpy linear quantiles.
- Unit checks: every point-mass perfect forecast scores zero; for probabilities (.25, .50, .25) at (0, 1, 2) and observation 1, CRPS = (.25² + (−.25)² + 0²)/2 = .0625.

Positive differences favour signals. This report will be committed before opening the build's analysis outputs. Machine-readable CDFs and per-run scores are in `part2_arm4b.json`.
