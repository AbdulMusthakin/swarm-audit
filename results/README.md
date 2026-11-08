# Results

Everything kept from the build lives here (`data/` and `logs/` are git-ignored, and a
cloud session's disk is temporary). Every number comes from logged data.

| path | what it is | written by |
|---|---|---|
| `validation.md` | The gates G1–G6: results, the numerical floor (from G3), drift. | `check_harness.py`, by hand for the rest |
| `harness/summary.json` | G2: per scripted-policy condition, the expected result, the observed values, the exact-law and G5-path checks, and the full analysis. | `scripts/check_harness.py` |
| `harness/expected_vs_observed.md` | G2: the expected-against-observed table (also in `validation.md`). | `scripts/check_harness.py` |
| `harness/audit/*.json` | G2: the scripted policies' audit tables (fitted on p_test). | `scripts/check_harness.py` |
| `figures/harness/*.png` | G2: Figure 1 for every scripted-policy condition. | `scripts/check_harness.py` |
| `audit/README.md` | p_test, the table sizes and the table construction. | by hand |
| `audit/<model>/*.json` | LLM audit tables: κ, n, SE, ε(p_test) and every per-history probability. | `scripts/run_audit.py` |
| `runs/<arm>/<condition>/run_<r>.jsonl.gz` | One line per decision (record, state, prompt sha256 and tokens, p_H, log-probabilities, p_valid, draws, action; on cheap talk both signal distributions, draw and signal). | `scripts/run_swarm.py` |
| `runs/<arm>/manifest.json` | Config, git commit, model, backend and engine settings, seeds, timings, p_valid flags. | `scripts/run_swarm.py` |
| `runs/<arm>/<condition>/drift.json` | The drift panel at the start and end of the condition and mixed into a swarm batch. | `scripts/run_swarm.py` |
| `serving/workspace_setup.json` | The teammate's workspace, weights download and the label check on the final prompts. | `scripts/setup_workspace.py` |
| `serving/signal_responsiveness.json` | Llama's signal responsiveness on the final cheap-talk prompt (d = 4 and 2). | `scripts/benchmark.py` |
| `serving/g3_batchinv{0,1}.json` | G3: repeatability maxima and throughput. | `scripts/check_serving.py` |
| `serving/g4_benchmark.json` | G4: throughput, tokens at round 20, cost per run, extrapolation to the manifest. | `scripts/benchmark.py` |
| `spend.json` | GPU spend recorded by the build scripts (wall-clock at list price). | GPU scripts |
| `pilot/` | G6 pilot runs and summary (never analysed). | `scripts/run_pilot.py` |
| `analysis/<arm>/<condition>.json` | The section 7 analysis of one condition (G5 results in `*_G5.json`). | `scripts/analyse.py` |
| `figures/fig1_<arm>_<condition>.png`, `figures/fig2.png` | Figures 1 and 2. | `scripts/analyse.py` |
| `tables/scores.md`, `tables/defects.md` | CRPS, mean errors, Δ against the floor; defects and bounds. | `scripts/analyse.py --collect` |
| `screening/` | Model screening (6–7 Nov): see `screening/summary.md`. | screening scripts |

| `gates/` | G5 gate records (`G5_<channel>.json`, the exception, the confirmation and `history.jsonl`). | `scripts/analyse.py --gate-g5`, `scripts/confirm_g5.py` |
| `followup/core/` | Exploratory follow-up of 8 Nov (temporal scores, uncertainty, clock-indexed audit, claims map, figures, report insert): start at `followup/core/summary.md`. | `scripts/followup_core_*.py` |
| `extra/` | The extra arms (8 and 9), owned by the extra-arms session. | `extra-arms` branch |

## Status

The status as of 8 Nov 2026 is in the completion note at the end of this file. The
manifests and gate records are authoritative.
