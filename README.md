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

## Results

See `results/README.md` and `results/validation.md`.

## Pre-existing code

The repository scaffold (folder layout and empty files) was created on 3 November 2026, before the sprint. Any code written before Friday 6 November is listed here.

The build (from 7 November 2026) reuses only this repository's own sprint code (the
screening of 6–7 November). The teammates' scoping study (`abdul_openrouter_study.py`,
`abdul_adaption_analysis.py`) was not in the repository when the build was written, so
none of its code is reused; its findings shaped the design (no hosted APIs).
