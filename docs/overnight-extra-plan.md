# Extra-arms overnight follow-up: plan (frozen before computing)

**Status: exploratory.** These analyses were chosen after seeing the arm 9a result. The
specification is frozen here, and in `results/extra/followup/plan.json`, before any new
comparison is computed. That improves credibility, but it does not make them preregistered
hypotheses. Every specified condition will be reported, including results that contradict
the suggested mechanism.

- **Branch:** `extra-arms`.
- **Outputs:** `results/extra/followup/`.
- **Scripts:** `scripts/extra_followup_*.py`.
- **New config:** `configs/extra-followup-olmo.yaml`.
- **Workspace:** `sensomichele` only.
- **Spend:** the $25 extra-arms cap covers everything. $5.16 was spent before this
  follow-up, and the follow-up's GPU ceiling is $8.

## Starting check (E0)

These were recomputed from the committed files and agree with the hand-off to 8 decimals:

| quantity | value |
|---|---|
| exact signal-aware P(F = 1) | 0.14239254 |
| exact actions-only P(F = 1) | 0.000076356 |
| live P(F = 1) | 8/50 |
| paired D | 0.0252263 [−0.0034260, 0.0538786] |
| expected CRPS advantage under the signal-aware law, Σ(Q − G)²/20 | 0.02052405 |
| K(H \| H-2, both neighbours "high", own "high") | 0.99956944 |
| K(H \| H-2, both neighbours "high", own "low") | 0.70578504 |
| K(H \| H-2, both neighbours "high", own "none") | 0.99919591 |
| actions-only κ(H \| H-2) | 0.90499137 |

## Evaluation of the hand-off, and deviations

I follow the hand-off, with these points settled in advance.

1. **The E2 "uniform replacement" sensitivity is the same chain as the E1 averaged audits.**
   - Suppose each hidden message is replaced independently by a uniform signal at each
     observation. Then the agent's next-step law is the kernel averaged over that
     replacement, which is exactly the E1 average with p_test's weights. Concretely:
     - own message randomised gives the E1 "neighbours only" variant;
     - neighbour messages randomised give "own only";
     - both randomised give "neither".
   - So the sensitivity is computed through the intervention code, checked against E1
     within 1e-12, and reported as that identity, not as extra evidence.
   - Read the other way, the identity gives E1 a process interpretation: an audit that
     averages over a message is the process in which that message is uninformative noise.
   - The deterministic `none` replacement is the distinct E2 experiment.
2. **The new last-1 live cells add no kernel information.** These are the E3 n = 10 last-1
   cell and the E2 live cells. Every prompt they can produce is one of the 109 enumerated
   last-1 prompts, and the serving floor is 0. So their laws are already fixed by the
   cached table.
   - Their live runs check the serving path and the intervention code, and give a new
     sample of runs.
   - They are labelled that way and never presented as new kernel evidence.
3. **Reusing the 9b table for the n = 4 full-history cell.** The prompt depends only on
   degree, history and payoffs: `Query` carries no n, agent id or condition. So the 9b
   sampled d = 2 full-history tables apply at n = 4.
   - analyse.py's kernel-fingerprint check must also pass.
   - A finite-state table is still only an approximation to a full-history swarm, and it
     is reported as one.
4. **Output layout.**
   - The new config keeps `results_dir: results/extra`, so the follow-up shares the single
     spend ledger with the earlier extra arms, and the OLMo serving and G5 records with
     arm 9.
   - New runs go to `results/extra/followup/runs/` (`--runs-root`).
   - New analyses and figures go to `results/extra/followup/`.
5. **Live E2 needs an additive hook in shared code.** It is an optional `observe` function
   in `swarm.simulate` that maps the actual record to the record the agent sees.
   - The default is none, which leaves behaviour unchanged.
   - Tests cover the default path and the routing.
   - It runs only after E3, if the projection fits.
6. **The primary event is F = 1**, that is y = 20 at n = 4: every agent bids high in every
   round 16–20. It is never mixed with "ever reached three-round consensus".

## Tasks, in priority order

Details and every numeric setting are in `plan.json`.

- **E0 (mandatory).** The starting check above, plus this frozen plan.
- **E1 (mandatory).** Four audit information variants, built exactly on the same
  1,296-profile space at n = 4:
  - **Variants:** `both`, `own_only`, `neighbours_only` and `neither`. The weights are
    p_test's, with equal weights as one sensitivity.
  - **Laws:** exact P(F = 1), P(F = 0), E[F], the law of y and E[X_t].
  - **Scores:** expected excess CRPS against `both`, and live scoring on the 50 arm 9a
    runs with paired whole-run intervals.
  - **Mechanism diagnostics:** occupancy and own-message composition of the all-H set,
    entry, one-step exit (an oracle, occupancy-weighted diagnostic) and finite-horizon
    survival.
  - **Table grid:** the full K(H | s, o, c) grid.
  - **Llama control:** the same four variants on Llama's table.
  - **Recovery tolerance:** |ΔP(F = 1)| ≤ 0.01 and expected excess CRPS ≤ 0.001.
- **E2, exact part (mandatory).**
  - **Processes:** four deterministic observation interventions (unmodified, own `none`,
    neighbours (0, 0, 2), both), exact.
  - **Primary outcome:** P(F = 1), with the factorial interaction.
  - **Also:** the full own-message grid, and the uniform identity above.
- **E3 (first GPU priority).**
  - **New cells:** OLMo n = 10 last-1 (arm_id 92) and n = 4 full-history (arm_id 93), 50
    runs each.
  - **Before spending:** the CPU forecasts. That is 100,000 simulated runs for n = 10
    last-1, and the exact law from the 9b table for n = 4 full.
  - **Benchmark:** 5 runs, then the projection with a ×1.25 margin.
  - **Intervals:** Clopper-Pearson for P(F = 1), Newcombe for its differences, and the
    whole-run bootstrap for mean F.
- **E2 live (if capacity remains).** Four cells (arm_ids 94–97), 50 runs each, with a
  familywise α of 0.01 (Bonferroni).
- **E4 and packaging (mandatory).**
  - **Write-up:** `mechanism_summary.md`, `claims.csv`, a Figure 2 candidate (PDF and
    PNG) and the appendix figures.
  - **Reproducibility:** an independent calculation of the key numbers, reproduction
    commands and the test output.
  - **Notes:** `completion_note.md`, plus a list of any unrun tasks.

## Safeguards

- **No change to the experiment:** no change to payoffs, temperature, tokens or prompts,
  and no extending runs on significance.
- **No new scope tonight:** no new model, free-text arm or graph sweep.
- **Audit weights:** they are never refitted on swarm histories. Occupancy weighting
  appears only as a labelled oracle diagnostic.
- **Like for like:** binary-action and six-outcome TVs are never compared. One-step
  comparisons use action marginals.
- **Exact results** carry a numerical tolerance, not a sampling interval. **Live
  results** carry whole-run or binomial intervals.
- **Wording:** no "intention", "commitment" or "collusion". The measured concept is
  dependence on own-message history.
