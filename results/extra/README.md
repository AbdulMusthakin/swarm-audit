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
| X1, workspace set-up | **blocked**: the `huggingface` secret in `sensomichele` has no `HF_TOKEN` key (diagnostic run, 7 Nov: no HF variable reaches the container; Llama download gets 401 "gated repo") | – |
| X2, serving repeatability | not run | `results/extra/serving/` |
| X3, cross-account drift panel | not run | `results/extra/serving/drift_panel.json` |
| Arm 8, 100 agents, random 4-regular | code ready, not run; also needs the core pilot's frozen payoffs (G6 has not run on the core branch) | `results/extra/graph_rr4_n100.json`, `runs/`, `audit/` |
| Arm 9, OLMo-3-7B | trigger not checked yet | – |

## Code added (additive; core behaviour unchanged)

- `topology.random_regular(n, d, seed)`, `topology.diameter`, `make("random_regular", n, d=, seed=)`.
- `Condition.graph_kwargs` (empty for core arms, so core names and logs are unchanged);
  `swarm.graph_of(data)`.
- `results_dir` config key (default `results/`): `build.results_root`, `spend_file`,
  `deploy_file`, `table_dir`. Scripts take `--config` and write there.
- `run_swarm.py`: `check_core_pilot: true` checks the core pilot's frozen payoffs.
- `check_serving.py --panel`: also saves the drift panel's outputs (X3).
- Tests: `tests/test_extra.py`.
