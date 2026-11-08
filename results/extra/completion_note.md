# Extra arms: completion note (8 Nov 2026)

This note follows `02_Extra_Arms_Coding_Handoff.md`. Branch: `extra-arms`. Abdul merges it.
Every output is under `results/extra/`. Nothing was pushed to the core branch, and none of
its tables or runs were replaced.

## Account checks

- **Modal workspace:** `sensomichele` (the teammate's). It is checked before every GPU
  session.
  - `abdulmusthakin` and `aliyusufbelal` are in `forbidden_workspaces`, and no session ran
    there.
- **X1, set-up: PASS.**
  - The HF secret first held the token under the wrong key, and Llama's download gave a 401.
  - After the teammate renamed the key to `HF_TOKEN`, the Llama and OLMo weights were cached
    in this workspace.
  - Every label is a single token with the configured id on the final prompts: Llama at
    d = 2 and d = 4, and OLMo.
  - Records: `serving/workspace_setup.json`, `serving/label_check_d4.json`,
    `serving/olmo-3-7b/workspace_setup.json`.
- **X2, repeatability: PASS for both models.** Batch invariance was on. The max difference
  was 0 alone, in a shuffled batch and warm, so the floor is 0.
  - Records: `serving/g3_batchinv1.json`, `serving/olmo-3-7b/g3_batchinv1.json`.
- **X3, cross-account drift panel:** all 136 panel prompts were byte-identical, with max
  difference 0 against the core's start, end and mixed panels for arms 4 and 4b. The
  serving settings and serving-code hash were also the same.
  - Record: `serving/x3_cross_account.json`.
- **Drift during runs:** in every GPU session, start against end gave max difference 0.

## Core fixes incorporated (merged, not reimplemented)

| commit | content |
|---|---|
| `61de1cb` | spend-ledger lock |
| `a83410a` | C1: fresh drift and repeatability measurements; serving identity in the cache key |
| `470c271` | C2: immutable provenance, checked on resume and load |
| `9edb338` | G6 pilot: payoffs frozen at (12, 10, 4, 0) |
| `04cfc68` | C3 and C5: G5 blocks launch; discrete Monte Carlo p-value |
| `846ce77` … `f2449ea` | C4 and C6: matched Δ_t null; labelled conditional intervals; G5 investigation |

- **Later core commits** (up to `d3f4ccd`) add results only. They change no shared code,
  configs or tests, so there was nothing further to merge.
- **Tests:** 104 passed, 1 skipped.

## Frozen configuration

- **Common to both arms:**
  - payoffs (12, 10, 4, 0), checked against the core pilot's `frozen_payoffs` before each
    run;
  - T = 20; the core's prompts and p_test;
  - vLLM 0.32 on an L4, batch invariance on, numerical floor 0;
  - base seed 20261107.
- **Arm 8:** `configs/extra-scale.yaml`. It differs from `core.yaml` only in the arm, the
  results directory, the $25 cap and the forbidden workspaces, and a test checks this.
- **Arm 9:** `configs/extra-olmo.yaml`. It differs from the arm 8 config only in the model
  (OLMo-3-7B-Instruct, revision `6e5971d9…`) and the arms, and a test checks this.

## Arm 8: complete

- **Design:** n = 100, one random 4-regular graph (seed 20261108), counts only, full
  history, **50 runs**. The graph has diameter 6 and 200 edges.
- **Results:**
  - The swarm settles near a high-bid fraction of 0.55 and never reaches consensus.
  - Actions-only audit CRPS 0.0142 [0.0113, 0.0173]; order-0 0.0170 [0.0140, 0.0204];
    paired difference +0.0028 [0.0016, 0.0039].
  - The actions-only audit overstates late high bidding by about 0.01.
  - Max Δ_t is 0.166, under the simultaneous null of 0.180.
  - ε(p_test) is 0.061 and ε(p_swarm) 0.058.
  - The bound is vacuous from round 2.
- **Details:** `arm8_summary.md`.

## Arm 9: trigger not met; run as EXPLORATORY by Abdul's decision; complete

- **Trigger:** d = 4 length-1 responsiveness was 0.0553, against a 0.05 rule fixed in
  `a2657c5`. So the trigger was **not met**.
- **Decision:** Abdul decided at 20:15 on 7 Nov to run it as exploratory, after seeing
  Llama's d = 2 value (0.038) and the arm 4b CRPS null.
- **9a:** G5 PASS. The lock-in stop did not apply (42 of 50 runs all-low, 8 all-high).
  Paired D +0.025 [−0.003, +0.054].
- **9b:** 50 runs, and every run locked into all-low.
  - Both audits forecast the end state (CRPS ≤ 0.0001).
  - Both are about two rounds too slow on the way there (max Δ_t 0.47–0.48 against a null
    of about 0.21).
  - Paired D is −0.0001 with a degenerate interval.
- **Details:** `arm9_summary.md`.

## Paths

`results/extra/`:
- **Summaries:** `README.md`, `arm8_summary.md`, `arm9_summary.md`, `arm8_numbers.json`.
- **Decision records:** `arm8_benchmark.json`, `arm9_trigger.json`, `arm9a_lock_in.json`,
  `arm9b_projection.json`.
- **Graph:** `graph_rr4_n100.json`.
- **Serving checks:** `serving/`, `serving/olmo-3-7b/`.
- **Audit tables:** `audit/llama-3.1-8b/`, `audit/olmo-3-7b/`.
- **Runs:** `runs/` (logs, provenance, drift, manifests).
- **Gate records:** `gates/olmo-3-7b/`.
- **Analyses and figures:** `analysis/`, `figures/`.
- **Spend ledger:** `spend.json`.

## Actual spend

**$5.16 of the $25 cap.** These are wall-clock estimates at the L4 list price, with each
session's 10-minute idle counted.

| item | spend |
|---|---|
| arm 8, including X1–X3 and the d = 4 table | $2.57 |
| arm 9, including OLMo X2 and both tables | $2.59 |

Check the Modal dashboard for the billed amount.

## Unresolved limitations

- **Arm 8:**
  - One graph realization, so the variation across graphs was not measured.
  - Counts only, so it says nothing about signal effects at n = 100.
  - Not a size-only contrast with the core's cycles.
- **Arm 9:**
  - It is exploratory and was chosen after seeing results, so it is not confirmatory and
    not a replication.
  - 9b's outcome is constant, so its intervals are degenerate and it cannot speak to the
    incremental value of signals.
  - One sampled table draw.
- **All intervals** are conditional on the fitted audit tables.
- **Cross-account kernel identity** was checked on the 136-prompt panel only.
- **The deployed app** `swarm-audit-serving` is still deployed in `sensomichele`. Its
  containers scale to zero after 10 minutes idle, so it costs nothing at rest. Stop it with
  `modal app stop swarm-audit-serving` if it is no longer wanted.
