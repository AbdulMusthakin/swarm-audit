# Can you audit a swarm by auditing its agents?

Apart Research Swarm Alignment Hackathon, 6–8 November 2026 · Track 2

## Team and roles

## Repository layout

- `swarm_audit/`: the package.
  - `game.py`, `ladder.py`, `tokens.py`: payoffs, prompt rendering, chat templating and labels.
  - `topology.py`: the cycle C_n and the complete graph K_n.
  - `policies.py`: the kernel backends behind one interface: scripted policies (gate G2) and
    the LLM (`Server.label_logprobs` on Modal).
  - `swarm.py`: lockstep swarm runs, per-decision logs, resume.
  - `states.py`, `audit.py`: coarse states, p_test, audit tables κ and κ′.
  - `prediction.py`, `baselines.py`: composed chains, exact laws at n = 4, baselines.
  - `stats.py`, `analysis.py`, `calibration.py`, `plots.py`: statistics, scores, gates, figures.
  - `serving.py`: the Modal vLLM server. `screening.py`: model screening.
  - `build.py`: configs, backends and the input gate for GPU scripts.
- `configs/`: `calibration.yaml`, `core.yaml`, `secondary.yaml` (the run manifest),
  `harness.yaml` (G2), `pilot.yaml` (G6), `screening.yaml`.
- `scripts/`: `check_harness.py` (G2), `check_serving.py` (G3), `benchmark.py` (G4),
  `run_audit.py`, `run_swarm.py`, `analyse.py` (analysis and G5), `run_pilot.py` (G6),
  and the screening scripts.
- `results/`: everything kept (see `results/README.md`). `data/` and `logs/` are git-ignored.
- `docs/`: the hand-offs. `setup-paper/`: the set-up paper.

## Setup

    pip install -e ".[dev]" "modal[api-proxy-support]"
    python -m pytest -q

GPU work needs an authenticated Modal client and, for the gated Llama model, the Modal
secret holding the Hugging Face token (`SWARM_AUDIT_HF_SECRET=huggingface`). Every GPU
script refuses to start until Abdul's inputs are recorded under `inputs:` in the configs.

## Reproducing the runs

In the order of `docs/handoff-build.md`. GPU runs use a teammate's Modal workspace,
never Abdul's: sign in with `python scripts/modal_login.py`, create the secret
"huggingface" there, then set it up once.

    python scripts/check_harness.py                                   # G2, CPU
    SWARM_AUDIT_HF_SECRET=huggingface python scripts/setup_workspace.py   # weights, labels
    SWARM_AUDIT_HF_SECRET=huggingface SWARM_AUDIT_BATCH_INVARIANT=1 python scripts/deploy_server.py
                                                  # warm GPU shared by the steps below

GPU steps run with `SWARM_AUDIT_HF_SECRET=huggingface SWARM_AUDIT_BATCH_INVARIANT=1`
(batch invariance was chosen in G3). Stop the deployed server when done:
`modal app stop swarm-audit-serving`.
    python scripts/check_serving.py --config core.yaml                # G3
    python scripts/benchmark.py --config core.yaml                    # G4
    python scripts/run_audit.py --config calibration.yaml --channel counts --enumerated
    python scripts/run_audit.py --config calibration.yaml --channel cheap_talk --enumerated
    python scripts/run_swarm.py --config calibration.yaml --arm calibration_counts
    python scripts/run_swarm.py --config calibration.yaml --arm calibration_cheap_talk
    python scripts/analyse.py --gate-g5 --floor <G3 floor> --runs ... --tables ... --tag ...   # G5
    python scripts/run_pilot.py                                       # G6
    python scripts/run_audit.py --config core.yaml --channel counts
    python scripts/run_audit.py --config core.yaml --channel cheap_talk
    python scripts/run_swarm.py --config core.yaml --arm core_counts
    python scripts/run_swarm.py --config core.yaml --arm core_cheap_talk
    python scripts/analyse.py --runs results/runs/<arm>/<condition> --tables results/audit/<model> --tag <channel>_d2
    python scripts/analyse.py --collect

## Report

The final report is `report/main.tex` (built PDF: `report/main.pdf`; build with
`cd report && latexmk -pdf main.tex`). The report and its scripts are at commit
[`0ee4fdb`](https://github.com/AbdulMusthakin/swarm-audit/tree/0ee4fdbeabb2bfef6696d0a84ef3152408517158); its evidence is pinned at commit
[`b524342`](https://github.com/AbdulMusthakin/swarm-audit/tree/b52434290039a1bc775137271a38ee35c06d432c).
Everything it reports can be recomputed on CPU from committed tables and logs, with no
model queries:

- Llama follow-up: [`results/followup/core/reproduce.md`](results/followup/core/reproduce.md).
- OLMo mechanism, interventions and memory controls:
  [`results/extra/followup/REPRODUCE.md`](results/extra/followup/REPRODUCE.md).
- Fresh four-agent OLMo cohort (the original roundwise check reapplied to the 50 fresh
  runs): `python scripts/report_fresh_cohort.py`.
- Diagnostic sensitivity within the four-agent setting (Appendix F; pre-registered in
  [`results/extra/followup/sensitivity/plan.md`](results/extra/followup/sensitivity/plan.md)):
  `python scripts/report_sensitivity.py`.
- Report figures: `python scripts/report_figures.py` (after the fresh-cohort line above).

## Results

See `results/README.md` and `results/validation.md`.

The [Dobrushin contraction comparison](results/followup/dobrushin/README.md) gives
current-round TV and mean-error bounds alongside Theorem 1, with figures and
CPU-only reproduction scripts.

## Pre-existing code

Only scaffold and stubs predate the sprint. The initial commit of 3 November 2026
([`1e52f0b`](https://github.com/AbdulMusthakin/swarm-audit/commit/1e52f0b)) holds the folder
layout, placeholder configuration files and package metadata, docstring-only module and
script stubs, and a set-up paper skeleton. No working code predates Friday 6 November.

The implementation (game, prompts, serving, audit tables, live runs, analysis, figures and
report) was written during the sprint, 6–8 November 2026. The build (from 7 November) reuses
only this repository's own sprint code (the screening of 6–7 November).

Earlier scoping work informed the design but is not reused. Before the build, the
teammates' scoping study (`abdul_openrouter_study.py`, `abdul_adaption_analysis.py`, outside
this repository) ran a message-free calibration through a hosted API and found that the same
prompt received different probabilities in isolation and during live games. That finding
shaped the design (models served by us, no hosted APIs); none of its code is in this
repository.
