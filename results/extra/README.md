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
| X2, serving repeatability | not run | `results/extra/serving/` |
| X3, cross-account drift panel | not run | `results/extra/serving/drift_panel.json` |
| Arm 8, 100 agents, random 4-regular | graph saved (simple, connected, 200 edges, measured diameter 6; pair distances 1–6: 200/579/1412/1980/758/21; 1 triangle). Not run; also needs the core pilot's frozen payoffs (G6 has not run on the core branch) | `results/extra/graph_rr4_n100.json`, `runs/`, `audit/` |
| Arm 9, OLMo-3-7B | trigger not checked yet | – |

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

- Up to `61de1cb` (spend-ledger lock, G4 info() fix, core cap $40). The review's correctness
  repairs (fresh repeatability bypassing replay, cache identity, config fingerprints,
  final payoff freeze, G5 discrete-threshold fix, matched TV null) were not on the core
  branch at this merge; they will be merged, not re-implemented, when they land.
