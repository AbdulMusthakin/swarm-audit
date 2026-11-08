# Reproducing the follow-up evidence (results/followup/core/)

Everything here is CPU-only. It reads the logged runs and the committed audit tables; nothing
is re-queried from a model. Plan and rules: `docs/overnight-core-plan.md`.

## Environment

- Python 3.13.18; numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2 (as installed in the core
  session's container on 8 Nov 2026). The serving stack (vLLM 0.32.0 on Modal, L4) is needed
  only for new runs, not for anything below.
- Run from the repository root on branch `claude/funny-turing-qv2y66`. The commits that
  produced each file are in `claims.csv` (`source_commit`).

## Inputs (committed)

| input | path |
|---|---|
| Llama cheap talk n = 10 runs (primary) | `results/runs/core_cheap_talk/cheap_talk_n10_cycle_full/` (50 runs, provenance.json) |
| Llama counts n = 10 / n = 20 runs | `results/runs/core_counts/counts_n{10,20}_cycle_full/` |
| Llama audit tables | `results/audit/llama-3.1-8b/{cheap_talk_d2,counts_d2}_*.json` (fitting histories `fit`, lengths `fit_L`) |
| OLMo runs and tables (extra arm 9b, read-only) | `results/extra/runs/olmo_core_cheap_talk/cheap_talk_n10_cycle_full/`, `results/extra/audit/olmo-3-7b/` |
| Original analyses (for reproduction checks) | `results/analysis/...`, `results/extra/analysis/...` |

## Commands

Run each one in its own step. Expected wall-clock times on 4 CPU cores are in brackets.

C1, temporal scores and uncertainty (primary, [3 min]):

```bash
python scripts/followup_core_temporal.py --runs results/runs/core_cheap_talk/cheap_talk_n10_cycle_full --tables results/audit/llama-3.1-8b --tag cheap_talk_d2 --status primary
```

C1, secondary conditions ([1 min] each; OLMo [3 min]):

```bash
python scripts/followup_core_temporal.py --runs results/runs/core_counts/counts_n10_cycle_full --tables results/audit/llama-3.1-8b --tag counts_d2
```

```bash
python scripts/followup_core_temporal.py --runs results/runs/core_counts/counts_n20_cycle_full --tables results/audit/llama-3.1-8b --tag counts_d2
```

```bash
python scripts/followup_core_temporal.py --config extra-olmo.yaml --runs results/extra/runs/olmo_core_cheap_talk/cheap_talk_n10_cycle_full --tables results/extra/audit/olmo-3-7b --tag cheap_talk_d2 --published results/extra/analysis/olmo_core_cheap_talk/cheap_talk_n10_cycle_full.json
```

C2, clock-indexed audit ([6 min] each for cheap talk, [1 min] for counts):

```bash
python scripts/followup_core_clock.py --runs results/runs/core_cheap_talk/cheap_talk_n10_cycle_full --tables results/audit/llama-3.1-8b --tag cheap_talk_d2 --status primary
```

```bash
python scripts/followup_core_clock.py --runs results/runs/core_counts/counts_n10_cycle_full --tables results/audit/llama-3.1-8b --tag counts_d2
```

```bash
python scripts/followup_core_clock.py --config extra-olmo.yaml --runs results/extra/runs/olmo_core_cheap_talk/cheap_talk_n10_cycle_full --tables results/extra/audit/olmo-3-7b --tag cheap_talk_d2
```

Mechanism diagnostic (teacher-forced errors, history content; [10 s] each):

```bash
python scripts/followup_core_mechanism.py --runs results/runs/core_cheap_talk/cheap_talk_n10_cycle_full --tables results/audit/llama-3.1-8b --tag cheap_talk_d2
```

```bash
python scripts/followup_core_mechanism.py --runs results/runs/core_counts/counts_n10_cycle_full --tables results/audit/llama-3.1-8b --tag counts_d2
```

```bash
python scripts/followup_core_mechanism.py --config extra-olmo.yaml --runs results/extra/runs/olmo_core_cheap_talk/cheap_talk_n10_cycle_full --tables results/extra/audit/olmo-3-7b --tag cheap_talk_d2
```

Independent re-derivation (imports nothing from `swarm_audit`; [10 s]):

```bash
python -I scripts/followup_core_check.py
```

Figures, then the claim map and tables:

```bash
python scripts/followup_core_figures.py
```

```bash
python scripts/followup_core_collect.py
```

Tests for the new code:

```bash
python -m pytest -q tests/test_followup.py
```

## Seeds

All follow-up randomness uses `numpy.random.SeedSequence([20261108, tag, ...])`
(`swarm_audit/followup.py: seeded`):

- 1, 2, 3, 5, 6: bootstraps;
- 101 + batch: Monte Carlo batches;
- 104: the precision run;
- 202 + j: table resample j;
- 301 + batch: clock-audit batches;
- 302: clock-audit bootstraps;
- 303 + j: clock-audit table resamples;
- 304 + batch: round-2 sensitivity;
- 401: mechanism bootstraps;
- 909: independent check.

The reproduction of the original predictors uses the original analysis seed (7, predictors
8). Re-running gives identical files except for the `time`, `seconds` and `git_commit` fields.

## Missing inputs and limits

- No full text of the related-work papers was read (arxiv.org is blocked from this session);
  see `related_work.md`.
- Cheap talk n = 20 and arm 7 (K_10) were still running when this package was first built;
  their analyses are added under `results/analysis/` when they finish, and are not part of
  the follow-up claims.
- The clock-indexed audit was chosen after the original mismatch was seen and is scored on
  the same runs. No new GPU histories were collected for it (decision and reason in
  `summary.md`).
