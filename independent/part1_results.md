# Independent calibration check — recorded before pipeline inspection

Only the supplied hand-off, the two enumerated audit tables, and the 50 logged runs were read. No pipeline code, gate output, or prior analysis was opened. The source data are from core commit f2449ea69fffe8943d6a34618db13a0e81724033; this branch also retains the existing hand-off commit 183adb0. Input hashes are in `input_sha256.json`.

Reproduce: `OPENBLAS_NUM_THREADS=1 python independent/check.py`. Requires Python and numpy. The exact computation enumerates all 1,296 profiles, constructs the product transition matrix, and propagates profile/count mass. Simulation uses a separate hierarchical bid/message sampler.

## Log checks

- Decisions: 4000; complete, unique 20 × 4 grids per run.
- Maximum probability discrepancies: {'p_H': 1.1102230246251565e-16, 'sig_after_H': 2.220446049250313e-16, 'sig_after_L': 2.220446049250313e-16}; overall 2.220446049250313e-16.
- Wiring/draw mismatches: {'state': 0, 'action': 0, 'signal': 0} (including START checks).
- Run y values, in filename order: [8, 3, 8, 6, 7, 6, 2, 9, 4, 6, 7, 6, 6, 7, 7, 6, 5, 9, 2, 8, 4, 6, 9, 4, 6, 3, 2, 6, 7, 4, 1, 8, 6, 3, 7, 6, 4, 6, 3, 6, 7, 7, 5, 9, 6, 8, 7, 5, 6, 8].
- Observed mean: 5.820000000000.

## Exact law and per-run test

- Exact mean: 5.471501977436; mass: 0.9999999999999986.
- Maximum transition row-sum error: 6.66e-16. Count-law mean agrees with the sum of round 16–20 expectations to 1e-11.
- Observed CDF gap: 0.202861576439.
- Null draws: 20,000 samples of 50 y values; seed 2026110702 (numpy default_rng/PCG64).
- Null 99th percentile: 0.199442801802 (numpy linear quantile).
- Exceedances, using 1e-12 tolerance: 149; corrected p = 0.007499625019; gate **FAIL** at p > .01.
- 200,000 forward simulations, seed 2026110701: maximum CDF error 0.001612801802, below 99% DKW bound 0.003639477080.

| y | Observed frequency | Exact PMF | Exact CDF |
| --- | --- | --- | --- |
| 0 | 0 | 0.002977816748 | 0.002977816748 |
| 1 | 1 | 0.017783420111 | 0.020761236859 |
| 2 | 3 | 0.053514975209 | 0.074276212068 |
| 3 | 4 | 0.106944554314 | 0.181220766382 |
| 4 | 5 | 0.158222035420 | 0.339442801802 |
| 5 | 3 | 0.183418774637 | 0.522861576439 |
| 6 | 15 | 0.172302243023 | 0.695163819462 |
| 7 | 9 | 0.133955201209 | 0.829119020671 |
| 8 | 6 | 0.087334807003 | 0.916453827674 |
| 9 | 4 | 0.048118234114 | 0.964572061788 |
| 10 | 0 | 0.022479594440 | 0.987051656228 |
| 11 | 0 | 0.008901723021 | 0.995953379249 |
| 12 | 0 | 0.002976377889 | 0.998929757138 |
| 13 | 0 | 0.000833977040 | 0.999763734178 |
| 14 | 0 | 0.000193533601 | 0.999957267779 |
| 15 | 0 | 0.000036562538 | 0.999993830316 |
| 16 | 0 | 0.000005485248 | 0.999999315564 |
| 17 | 0 | 0.000000629565 | 0.999999945128 |
| 18 | 0 | 0.000000052034 | 0.999999997163 |
| 19 | 0 | 0.000000002766 | 0.999999999929 |
| 20 | 0 | 0.000000000071 | 1.000000000000 |

## Per-round diagnostic

All entries count agents per round (out of four); observed columns average the 50 runs. These are descriptive diagnostics, without post-hoc significance claims.

| Round | H observed | H exact | high observed | high exact | low observed | low exact | none observed | none exact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.340000000 | 2.248706004 | 0.260000000 | 0.365256032 | 0.340000000 | 0.229879763 | 3.400000000 | 3.404864206 |
| 2 | 1.020000000 | 1.198468411 | 1.420000000 | 1.266556454 | 0.980000000 | 1.115480106 | 1.600000000 | 1.617963440 |
| 3 | 1.020000000 | 1.081216282 | 1.420000000 | 1.248695797 | 1.500000000 | 1.357683413 | 1.080000000 | 1.393620790 |
| 4 | 0.940000000 | 1.088546330 | 1.200000000 | 1.234035098 | 1.320000000 | 1.402075433 | 1.480000000 | 1.363889468 |
| 5 | 1.260000000 | 1.093061774 | 1.060000000 | 1.230480766 | 1.520000000 | 1.409837162 | 1.420000000 | 1.359682072 |
| 6 | 1.240000000 | 1.094098504 | 1.180000000 | 1.229843756 | 1.640000000 | 1.411201903 | 1.180000000 | 1.358954341 |
| 7 | 0.980000000 | 1.094274337 | 1.380000000 | 1.229750407 | 1.120000000 | 1.411442426 | 1.500000000 | 1.358807167 |
| 8 | 1.060000000 | 1.094298033 | 1.340000000 | 1.229739362 | 1.240000000 | 1.411484648 | 1.420000000 | 1.358775989 |
| 9 | 1.180000000 | 1.094300374 | 0.900000000 | 1.229738474 | 1.320000000 | 1.411492000 | 1.780000000 | 1.358769526 |
| 10 | 1.180000000 | 1.094300450 | 1.160000000 | 1.229738485 | 1.380000000 | 1.411493268 | 1.460000000 | 1.358768246 |
| 11 | 1.080000000 | 1.094300412 | 0.940000000 | 1.229738510 | 1.520000000 | 1.411493485 | 1.540000000 | 1.358768005 |
| 12 | 0.980000000 | 1.094300399 | 1.320000000 | 1.229738517 | 1.420000000 | 1.411493522 | 1.260000000 | 1.358767962 |
| 13 | 1.000000000 | 1.094300396 | 1.160000000 | 1.229738518 | 1.420000000 | 1.411493528 | 1.420000000 | 1.358767954 |
| 14 | 1.200000000 | 1.094300396 | 1.300000000 | 1.229738518 | 1.320000000 | 1.411493529 | 1.380000000 | 1.358767953 |
| 15 | 0.940000000 | 1.094300396 | 1.340000000 | 1.229738518 | 1.420000000 | 1.411493529 | 1.240000000 | 1.358767953 |
| 16 | 1.160000000 | 1.094300395 | 1.300000000 | 1.229738518 | 1.220000000 | 1.411493529 | 1.480000000 | 1.358767953 |
| 17 | 1.240000000 | 1.094300395 | 1.280000000 | 1.229738518 | 1.180000000 | 1.411493529 | 1.540000000 | 1.358767953 |
| 18 | 1.120000000 | 1.094300395 | 1.160000000 | 1.229738518 | 1.360000000 | 1.411493529 | 1.480000000 | 1.358767953 |
| 19 | 0.940000000 | 1.094300395 | 1.400000000 | 1.229738518 | 1.380000000 | 1.411493529 | 1.220000000 | 1.358767953 |
| 20 | 1.360000000 | 1.094300395 | 1.060000000 | 1.229738518 | 1.400000000 | 1.411493529 | 1.540000000 | 1.358767953 |

## Preliminary verdict against the supplied hand-off

The independently computed mean and gap agree with the hand-off's rounded 5.47 and 0.203. The p-value agrees with its 0.0076 within the requested Monte Carlo tolerance. Agreement verifies the failed calibration gate; it does not establish that chance is the cause of the failure. The gate file will be compared only after this report is committed.
