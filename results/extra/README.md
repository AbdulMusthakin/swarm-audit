# Extra arms (docs/handoff-extra-arms.md)

Separate arms on a second Modal account. Nothing here feeds the core's primary contrast.

- **Modal workspace:** `sensomichele` (the second teammate's). The configs forbid
  `abdulmusthakin` and `aliyusufbelal`.
- **Branch:** `extra-arms`, with the core branch (`claude/funny-turing-qv2y66`) merged in.
- **Configs:** `configs/extra-scale.yaml` (arm 8). `configs/extra-olmo.yaml` (arm 9) only if
  its trigger is met.
- **Spend ledger:** `results/extra/spend.json`, cap $25. Spent so far: $0 on GPU. The X1
  attempt and its diagnostic used only CPU containers, a few minutes each.

## Status

| step | status | file |
|---|---|---|
| X1, workspace set-up | **PASS** (7 Nov). The secret first stored the token under key `huggingface` (Llama download: 401); after the teammate renamed it to `HF_TOKEN`, the token (HF account `micsenso`, read) has Llama access. Weights cached in this workspace's `swarm-audit-weights` volume (16.1 GB, 56 s, CPU). Every label one token with the configured id: actions 473/445, signals 1579/3428/7000, on 148 action and 250 signal positions (core d = 2 prompts), and on 66 counts prompts at d = 4 (arm 8's degree, longest 782 tokens) | `serving/workspace_setup.json`, `serving/label_check_d4.json` |
| X2, serving repeatability | **PASS** (7 Nov, 19:11 UTC). Batch invariance on: 48 prompts alone, in a shuffled batch of 448 others, and warm; max difference **0** in P(H) and in every signal probability (floor 0, as in the core's G3). 4.07 prompts/s. Fresh queries (no replay) | `serving/g3_batchinv1.json` |
| X3, cross-account drift panel | **identical**: all 136 panel prompts byte-identical, max difference **0** in P(H) and every signal probability, against the core's start, end and mixed panels of arms 4 and 4b (6 comparisons); same serving settings and serving-code hash. Informational | `serving/x3_cross_account.json` |
| Arm 8, 100 agents, random 4-regular | graph saved (simple, connected, 200 edges, measured diameter 6; pair distances 1–6: 200/579/1412/1980/758/21; 1 triangle). Core pilot froze the payoffs unchanged, (12, 10, 4, 0) = this config's (checked by `build.check_core_pilot_payoffs`). Audit table (d = 4, 1,502 histories) running | `results/extra/graph_rr4_n100.json`, `runs/`, `audit/` |
| Arm 9, OLMo-3-7B (conditional second-model extension) | **Trigger not met; arm 9 not run.** Llama's signal responsiveness on the final cheap-talk prompt, on the reading fixed in advance (d = 4 screening grid, length 1), is 0.0553, not below 0.05 (core G4). Reported, not deciding: d = 2 length 1 is 0.038, d = 4 length 5 is 0.022 | `arm9_trigger.json` |

## Code added (additive; core behaviour unchanged)

- `topology.random_regular(n, d, seed)`, `topology.diameter`, `make("random_regular", n, d=, seed=)`.
- `Condition.graph_kwargs` (empty for core arms, so core names and logs are unchanged);
  `swarm.graph_of(data)`.
- `results_dir` config key (default `results/`): `build.results_root`, `spend_file`,
  `deploy_file`, `table_dir`. Scripts take `--config` and write there.
- `run_swarm.py`: `check_core_pilot: true` checks the core pilot's frozen payoffs.
- `check_serving.py --panel`: also saves the drift panel's outputs (X3).
- `analyse.py --collect` reads and writes under the config's results tree;
  `benchmark.py` (the core's G4) refuses an extra config.
- Tests: `tests/test_extra.py`.

## Core commits merged

- `61de1cb`: spend-ledger lock, G4 info() fix, core cap $40.
- `a83410a`: review C1, fresh repeatability/drift measurements, serving identity in the cache key.
- `470c271`: review C2, immutable provenance for runs and tables, checked on resume and load.
- `9edb338`: G6 pilot, payoffs frozen at (12, 10, 4, 0).
- `04cfc68`: review C3 and C5, G5 blocks the core launch; discrete Monte Carlo p-value.
- `main` up to `1676dcd` (teammates' hosted scoping evidence), so PR #7 has no conflicts.
- Still to merge when they land: the matched TV null reference and labels for
  conditional uncertainty (review C6 or later).

## Arm 9 trigger: the reading, fixed before any result (7 Nov, about 19:00 UTC)

`responsiveness.measure` reports two grids. The trigger uses **`d4.signal_responsiveness_length1_mean`**,
the screening grid: the 0.05 bar and Llama's screening value 0.051 were defined on it, and
the hand-off says "on the same grid". Arm 9 runs only if that number is **strictly below
0.05**. The d = 2 and length-5 values are reported but do not decide. Source: the core's
`results/serving/signal_responsiveness.json` if it exists when arm 8 finishes; otherwise
measured on this account with the same function, and recorded as such. No responsiveness
result for the final prompt existed on either branch when this was written.
