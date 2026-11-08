# Overnight follow-up: completion note (8 Nov 2026, about 01:40 UK; review items added about 06:15)

- **Hand-off:** `02_Extra_Arms_Overnight_Handoff_2026-11-08.md`, evaluated in
  `docs/overnight-extra-plan.md`.
- **Branch:** `extra-arms`. Abdul merges it; nothing was pushed to the core branch.
- **Outputs:** everything new is in `results/extra/followup/`. No original output, prompt,
  frozen config, `setup-paper/` file or report file was edited.

## Status: every specified task finished

| task | result | files |
|---|---|---|
| E0 starting check, frozen plan | reproduces the hand-off to 8 decimals | `plan.json`, `docs/overnight-extra-plan.md` (`d9ec5fc`, before any comparison) |
| E1 four audit variants (exact), equal-weight sensitivity, live scoring, diagnostics, Llama control | own message only recovers the full forecast; neighbours only does not; Llama null | `audit_ablations.json` |
| E2 four deterministic interventions (exact), interaction, own-message grid, uniform sensitivity | "none" keeps P(F = 1) at about 0.141 in all four; interaction 0.0016 | `observation_interventions.json` |
| E2 live validation, 4 × 50 runs | all pass: per decision (difference 0), per run (p 0.23–1.0), routing | `observation_live.json`, `runs/followup_obs_*` |
| E3 n = 10 last-1, n = 4 full history, 50 runs each, CPU forecasts first | size effect, no detectable memory effect | `size_memory.csv`/`.json`, `e3_forecast.json`, `e3_benchmark.json` |
| Independent calculation, CRPS identity | PASS (1e-10) | `independent_check.json`, `scripts/extra_followup_check.py` |
| Figure 2 candidate and appendix figures | PDF and PNG at print size | `figures/` |
| Summary and claims | 39 claims, every one with a source key | `mechanism_summary.md`, `claims.csv` |
| Reproduction and tests | 119 passed, 1 skipped | `REPRODUCE.md` |

`unrun_tasks.md` lists what was out of scope. Nothing specified was left unrun.

## Recommended claim for the report authors

> In an exploratory OLMo-3-7B condition (n = 4 cycle, last-round cheap talk), the
> signal-aware audit composes to a 0.142 probability that every agent bids high
> throughout rounds 16–20 (live: 8/50). The derived actions-only audit gives 0.00008.
> Exact information ablations on the same cached kernel locate the loss.
> - Retaining only each agent's own last message recovers the outcome (0.137);
>   retaining only neighbours' messages does not (0.00007).
> - The cause is synthetic averaging. The audit weights the own message "low" at one
>   third. The full process almost never occupies that state inside all-high (0.01%), and
>   there an agent stays high with probability only 0.68–0.73, against ≥ 0.9987 otherwise.
> - Agents do not need their messages to sustain high bidding. Showing them "no message"
>   instead leaves the outcome nearly unchanged (0.141).
>
> The high outcome arises almost entirely from an all-high first round followed by
> persistence. In live runs it disappears at n = 10 under both memory settings
> (0/50 each), consistent with 0.6225ⁿ. The nominal size contrast excludes 0 under
> both memories. With a multiplicity correction, it excludes 0 only under last-round
> memory. No memory effect is detectable at n = 4.

**Required qualifiers:**
- exploratory, chosen after earlier results;
- one model, game, prompt, topology and horizon;
- exact numbers are conditional on the cached kernel;
- "none" is meaningful content;
- no intention, commitment or collusion is claimed;
- the Llama control shows no such effect;
- the original Llama primary null remains the primary result.

## Spend

- **GPU sessions:** six, all in workspace `sensomichele`, with
  `forbidden_workspaces: [abdulmusthakin, aliyusufbelal]` checked before each.
- **Follow-up total:** $1.43 at the L4 list price, including six 10-minute idle upper
  bounds ($0.80), against the $8 follow-up ceiling.
  - E3 n = 10 last-1: $0.11, in two sessions (5-run benchmark, then 45 runs).
  - E3 n = 4 full history: $0.26.
  - E2 live: 4 × $0.03.
- **Extra-arms total:** **$6.59 of the $25 cap**, from `results/extra/spend.json`.
- **Projection:** the benchmark projected $1.55 with the ×1.25 margin (`e3_benchmark.json`).
- **Billing:** check the Modal dashboard for the billed amount.

## Checks on every live session

- **Drift:** start against end, and against the mixed batch, was 0 at floor 0, so the
  action was "ok".
- **Validity:** no p_valid flag.
- **Kernel identity:** the kernel fingerprints match the reused tables.
  - n = 10 last-1: all 10,000 logged probabilities equal the enumerated table.
  - E2 live: all 16,000 logged probabilities equal the transformed-state entries.

## Deviations, corrections and failures

1. **E2 uniform sensitivity.** It is the same law as the E1 averaged audits. It was
   reported as that identity and checked to 1e-16, as stated in the plan before
   computing.
2. **Shared ledger.** The config keeps `results_dir: results/extra`, so the ledger, the
   OLMo serving checks and the G5 records are shared. Runs went to
   `results/extra/followup/runs/` via `--runs-root`.
3. **Shared-code changes, all additive and tested:**
   - `swarm.simulate` and `run_condition` take an optional `observe` hook. The default
     None is unchanged, and a test checks it.
   - `run_swarm.py` passes an arm's `observe` to that hook.
   - New module `swarm_audit/followup.py`.
4. **Independent review (Codex, PR #9 note): two items acted on in `11571f0`.**
   - **M2.** `analysis.is_primary_contrast` now requires the core's arm
     (`core_cheap_talk`, arm_id 6). Arm 9b's stored `is_primary_contrast: true` was wrong;
     the original JSON is kept as provenance, and
     `results/extra/analysis/classification.json` gives the corrected labels. **For the
     core agent:** this changes `swarm_audit/analysis.py`. Arm 6's flag is unchanged
     (tested).
   - **m1.** The survival diagnostic now uses first entry, as frozen. Its effect is at
     most 2e-6, and the all-entry version is kept as a labelled secondary key.
   - Codex's M4 concerns the core's wording, not this branch.
7. **Independent review, 05:30 continuation (PR #9 note): two items acted on after the
   06:00 check-in.**
   - **m4, resume guard.** The observation intervention was not bound in run provenance,
     so a cell could resume under a different intervention. `swarm.bind_observation` now
     writes `observation.json` before a new cell's first run and rejects any later
     session with a different (or missing) intervention. Older cells are checked
     against the interventions their manifest sessions recorded. Runs without an
     observer are unchanged (tested).
     - Codex's probe is now rejected with a ProvenanceError.
     - The four completed E2 cells each ran one intervention with 0 routing
       mismatches. They are recorded in `observation_binding_legacy.json`; their
       provenance is not rewritten, and no rerun is needed.
   - **Multiplicity.** The frozen plan's nominal E3 intervals stand as the specified
     analysis. A post hoc Bonferroni sensitivity was added
     (`size_memory.json` `multiplicity_sensitivity`; it matches the review). Under it,
     the full-history size contrast includes 0: [−0.23, +0.01] over the two size
     contrasts. The wording in `mechanism_summary.md` and `claims.csv` is narrowed
     accordingly.
5. **Figure 2 panel A, first draft.** Its labels overlapped; it was redrawn and checked
   visually.
6. **No GPU failure, no stop rule triggered, no extension on significance.**

## Source commits

- **Inputs:** `818a9ef` (start; main merged).
- **Plan:** `d9ec5fc`.
- **Forecasts:** `a4fbb63`.
- **E1/E2 exact:** `4a81705`.
- **Hook and checks:** `649ab5b`.
- **Runs:** `335c438` (n = 10 last-1), `526e4ac` (n = 4 full), `f8aa27b` (E2 live).
- **Review fixes:** `11571f0`.
- **E3 analysis:** `5554d8b`.
- This note and the final summary follow.

## Left as is

- **Deployed app.** `swarm-audit-serving` is still deployed in `sensomichele`. It scales to
  zero after 10 minutes idle, so it costs nothing at rest. Stop it with:

  ```bash
  modal app stop swarm-audit-serving
  ```
- **Open PRs.** [PR #9](https://github.com/AbdulMusthakin/swarm-audit/pull/9) (extra-arms
  → main) is open. Its earlier conflict with main after PR #10 was fixed in `818a9ef`.
