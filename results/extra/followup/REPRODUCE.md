# Reproducing the overnight follow-up

Run everything from the repository root, on branch `extra-arms`.

## CPU only

These steps need no GPU and no model queries. Each one reads committed tables and logs.

E3 forecasts, made before any spend (writes `e3_forecast.json`):

```bash
python scripts/extra_followup_forecast.py
```

E1 audit variants and E2 exact interventions (writes `audit_ablations.json` and
`observation_interventions.json`):

```bash
python scripts/extra_followup_exact.py
```

Independent recalculation of the E1/E2 endpoints and of the CRPS identity (writes
`independent_check.json`):

```bash
python -I scripts/extra_followup_check.py
```

E3 two-by-two (writes `size_memory.csv`, `size_memory.json` and `analysis/`):

```bash
python scripts/extra_followup_size_memory.py
```

E2 live validation, once its runs exist (writes `observation_live.json`):

```bash
python scripts/extra_followup_live.py
```

Fresh four-agent cohort: the original arm 9a roundwise check (exact round laws of the cached
audit chains, original simultaneous cutoffs) reapplied to the 50 unmodified E2 runs; it also
recomputes the original cohort and stops unless that reproduces the committed analysis
(writes `fresh_cohort_roundwise.json`, about 5 s):

```bash
python scripts/report_fresh_cohort.py
```

Diagnostic sensitivity within the four-agent setting, pre-registered in
`sensitivity/plan.md` (writes `sensitivity/results.json` and `sensitivity/summary.md`,
about 20 s):

```bash
python scripts/report_sensitivity.py
```

Figure 2 candidate and the appendix figures (writes `figures/`):

```bash
python scripts/extra_followup_figures.py
```

Claims table (writes `claims.csv`):

```bash
python scripts/extra_followup_claims.py
```

Tests, including the follow-up module and the observation hook:

```bash
python -m pytest -q
```

## GPU runs

These steps already ran in the Modal workspace `sensomichele`. Don't rerun them: resume
skips logged runs, and new runs would add spend.

Before any GPU run, set these two environment variables. Both are non-secret names:

```bash
export SWARM_AUDIT_HF_SECRET=huggingface SWARM_AUDIT_BATCH_INVARIANT=1
```

E3, n = 10 last-1 (arm_id 92). The first command is the 5-run benchmark; the second runs
the rest:

```bash
python scripts/run_swarm.py --config extra-followup-olmo.yaml --arm followup_olmo_n10_last1 --runs-root results/extra/followup/runs --max-runs 5
```

```bash
python scripts/run_swarm.py --config extra-followup-olmo.yaml --arm followup_olmo_n10_last1 --runs-root results/extra/followup/runs
```

E3, n = 4 full history (arm_id 93):

```bash
python scripts/run_swarm.py --config extra-followup-olmo.yaml --arm followup_olmo_n4_full --runs-root results/extra/followup/runs
```

E2 live cells (arm_ids 94–97). Run the command once per arm, substituting `<arm>` with
each of `followup_obs_unmodified`, `followup_obs_own_null`,
`followup_obs_neighbour_null` and `followup_obs_both_null`:

```bash
python scripts/run_swarm.py --config extra-followup-olmo.yaml --arm <arm> --runs-root results/extra/followup/runs
```
