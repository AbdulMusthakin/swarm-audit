# Compact design and results table (main-text candidate)

All rows: T = 20 rounds, 50 live runs per condition, payoffs (12, 10, 4, 0), cycle C_n (degree
2), full history, audits fitted only on isolated synthetic histories (p_test: lengths 1–19,
50 fitting histories per state). Every number is in `claims.csv` with its source.

**Notation.**

- **D:** CRPS(actions-only) − CRPS(signal-aware); negative favours actions-only.
- **Clock gain:** CRPS(pooled) − CRPS(clock-indexed); positive favours the clock audit.
- **Early bias:** the teacher-forced mean of p_H − κ(H|s) over rounds 2–7.
- **Interval:** 95%, whole-run bootstrap, conditional on the tables (simulation at the
  stated size).
- **"Nested":** also resamples the tables and the simulation.

| | Llama 3.1 8B, cheap talk, n = 10 (primary) | Llama 3.1 8B, counts, n = 10 | OLMo-3-7B, cheap talk, n = 10 (extra-arm runs) |
|---|---|---|---|
| Status | endpoint D: **pre-specified primary**; rest exploratory | secondary / exploratory | exploratory (core reanalysis of imported runs) |
| Endpoint D | **−0.000086 [−0.000221, +0.000035]**; at 740k sims −0.000085 [−0.000225, +0.000054]; nested [−0.00047, +0.00009] | n/a (one audit) | −0.00007 (degenerate: every run all-low) |
| Early-round D (rounds 2–7) | −0.00074 [−0.00103, −0.00043]; nested [−0.00135, −0.00022] | n/a | −0.0074 [−0.0086, −0.0060] |
| Swarm vs forecast X_t, rounds 2–7 | 0.32 vs 0.38 | 0.58 vs 0.58 | 0.064 at round 4; forecasts 0.24 (actions-only), 0.26 (signal-aware) |
| Early bias, pooled audit | −0.058 [−0.062, −0.054] | +0.002 [−0.003, +0.006] | −0.047 [−0.050, −0.045] |
| Early bias, clock-indexed audit | −0.020 [−0.023, −0.016] | +0.008 [+0.003, +0.012] | −0.016 [−0.019, −0.014] |
| Clock gain, rounds 2–7 | **+0.0060 [+0.0033, +0.0086]** | −0.0001 [−0.0003, +0.0002] | +0.041 [+0.029, +0.051] |
| Clock gain, endpoint | −0.0036 [−0.0070, −0.0002] | +0.0008 [−0.0002, +0.0018] | +0.000005 (degenerate) |
| Swarm share of high bids before the last round (p_test: 0.5) | 0.35 | 0.59 | 0.13 at round 12 |

**Replication, Llama cheap talk n = 20 (secondary, fresh runs).**

| quantity | value |
|---|---|
| Endpoint D (2M sims, conditional on the tables) | −0.00031 [−0.00045, −0.00017] |
| Endpoint D, nested | [−0.00081, +0.00008]: sign not established |
| Early D | −0.00097 [−0.00119, −0.00075] |
| Early bias, pooled → clock | −0.060 → −0.021 |
| Clock gain, rounds 2–7 | +0.0089 [+0.0066, +0.0111] |
| Clock gain, endpoint | −0.0083 [−0.0109, −0.0058] |
