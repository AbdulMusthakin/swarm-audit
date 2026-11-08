# Llama 3.1 70B: the core matrix, for interest only

**Not part of the report or the submission.** Abdul asked, on 8 November 2026, for runs with a
70B model in the Modal workspace used for the core (`aliyusufbelal`), "for interest". Nothing in
`report/`, `setup-paper/` or `results/` uses these results, and no claim of the submission rests on
them. The main README does not link here.

## What ran

`configs/model-70b.yaml` is `configs/core.yaml` with only the model, the serving hardware, the
output folder and the arm names changed. The prompts, payoffs, T, p_test, audit sizes, seeds,
analysis settings and settled decisions are the core's.

- **Model:** `meta-llama/Llama-3.1-70B-Instruct` at revision `1605565b` (HF main, 8 Nov 2026). It
  is the same family and tokenizer as the core's 8B, so the labels are the same single tokens with
  the same ids. `setup_workspace.py` checked this on the final prompts (`serving/workspace_setup.json`).
  The HF token in the workspace has no access to Llama 3.3 70B.
- **Serving:** one B200 (192 GB), bf16, no tensor parallelism, vLLM 0.32.0, batch invariance on.
  It ran as its own Modal app `swarm-audit-serving-70b` with its own weights volume
  `swarm-audit-weights-70b`, so the core's app and volume were never touched.
- **Arms:** the same arm ids as the core (4, 41, 5, 6, 7, 71). Each 70B run therefore uses the same
  random draws as the 8B run with the same index.
  - n = 4 calibration (last round only), both channels.
  - The cycle with full history at n = 10 and n = 20, both channels.
  - K_10 (d = 9, actions-only tables), both channels.
  - 50 runs per condition.
- **Gates:** G3 was re-measured on the B200. G5 ran on the n = 4 arms. The arm names (`m70_*`) skip
  the core-only checks for the G4 plan and the G5 launch block; G5 was run and passes anyway.
- **Cost:** $34.43 for 5.5 B200-hours at the $6.25/h list price, against Abdul's $40 cap. The
  figures are wall-clock estimates in `spend.json`; the Modal dashboard has the billed amount.

Reproduce in two steps:

```
SWARM_AUDIT_HF_SECRET=huggingface python scripts/setup_workspace.py --config model-70b.yaml
bash interest/model-70b/run_all.sh
bash interest/model-70b/run_stage2.sh
```

`run_all.sh` covers G3, the calibration arms, the d = 2 tables, n = 10 and the analysis.
`run_stage2.sh` covers n = 20 and K_10. `compare.py` writes `comparison.md`, the 8B-vs-70B table
for every condition, and `comparison.json`.

## Results

**Gates**

| | 8B (core, L4) | 70B (B200) |
|---|---|---|
| G3 serving floor (batch invariant) | 0 | 0 |
| G5 counts | pass | pass |
| G5 cheap talk | **fail** on the per-run test | pass, per decision and per run |
| drift, every condition | 0 | 0 |

**The kernel is very different.** The 70B opens with H about 4–9% of the time, against 56–67%
for the 8B. On counts it follows its neighbours much more sharply: in the enumerated table P(H)
is 0.89 after both neighbours bid H and 0.03 after both bid L. From that low start the swarm climbs
to a 23–43% share of high bids in rounds 16–20. No run of either model reaches consensus, except
one run in 50 for the 70B on cheap talk at n = 10, on both the cycle and K_10 (all low).

**The primary contrast flips.** D is CRPS(actions-only) minus CRPS(signal-aware) on y, the cheap
talk cycle with full history; positive favours the signal-aware audit.

| | 8B | 70B |
|---|---|---|
| n = 10 (the primary contrast) | −0.00009 [−0.00022, +0.00004] | **+0.0061 [+0.0045, +0.0077]** |
| n = 20 | −0.00004 [−0.00011, +0.00003] | **+0.0063 [+0.0044, +0.0079]** |

For the 8B, conditioning the audit on messages adds nothing. For the 70B it helps, and the result
replicates at n = 20. The 70B's messages carry information about its next bid that its action
history does not.

**But the 70B cheap-talk audits are biased.** On the cycle, both tables under-predict high bids
late in the game: the forecast mean minus the swarm is −0.05 to −0.08. The order-0 baseline (every
agent independent at the audit's average rate) scores better than either table:

| CRPS, cheap talk, cycle | order-0 | actions-only | signal-aware |
|---|---|---|---|
| 8B n = 10 | 0.041 | 0.040 | 0.040 |
| 70B n = 10 | **0.049** | 0.069 | 0.063 |
| 8B n = 20 | 0.033 | 0.030 | 0.030 |
| 70B n = 20 | **0.037** | 0.056 | 0.050 |

The signal-aware audit beats the actions-only one, but neither is a good forecast for the 70B on
this channel. The 70B's defects ε_swarm, the gap between its swarm behaviour and the audit table,
are about twice the 8B's on the actions-only tables (0.11–0.13 against about 0.06).

**Counts on the cycle is fine for both.** CRPS at n = 10: 0.043 for the 8B, 0.046 for the 70B.
At n = 20: 0.032 and 0.035. No forecast bias in either.

**K_10, the second topology.** On counts the 70B audit fails clearly. The d = 9 actions-only table
over-predicts high bids by about 9 points (F_error +0.093 [+0.056, +0.132]), and its CRPS is 0.10
against 0.04 for the 8B. On cheap talk the two models are similar (actions-only CRPS 0.057 against
0.060), and the 70B's bias is smaller than the 8B's.

**In one line.** A larger model makes the messages informative and the n = 4 calibration exact. Yet
the one-agent audit tables forecast the 70B's swarms worse than the 8B's on three of six conditions
(cheap talk on the cycle at both sizes, counts on K_10). Calibrating at n = 4 did not guarantee a
good forecast at n = 10.

## Caveats

- Exploratory. Nothing was fixed in advance for this model, and nothing here was tuned.
- A single model at a single revision, on a single serving setup. The 70B and the 8B also differ in
  GPU (B200 against L4); G3 gives an exact floor on both, so this should not affect the kernels.
- The intervals are the core's: 95% percentile intervals over 2,000 resamples of whole runs,
  conditional on the fitted tables.

## Clean-up

The weights volume `swarm-audit-weights-70b` (141 GB) is still in the `aliyusufbelal` workspace.
Delete it once it is no longer needed:

```
modal volume delete swarm-audit-weights-70b
```
