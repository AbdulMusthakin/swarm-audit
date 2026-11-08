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

## Completion note: core session, 8 Nov 2026 (about 03:45 UTC)

**All scheduled core work is finished.** Every arm in `results/serving/g4_plan.json` ran,
including arm 7 (second topology), which was planned only if nothing had been cut. The
deployed server `swarm-audit-serving` in workspace `aliyusufbelal` was stopped at 03:40 UTC.

### Gates

| gate | result | record |
|---|---|---|
| G1 unit tests | PASS (now 118 passed, 1 skipped) | `pytest` |
| G2 harness | PASS (rerun after review C4/C5) | `validation.md`, `harness/` |
| G3 serving | PASS: batch invariance on, numerical floor 0 | `serving/g3_*.json` |
| G4 throughput | PASS: everything fits (8.5 GPU-hours, $6.82 projected) | `serving/g4_*.json` (`64bafb8`) |
| G5 counts | PASS | `gates/G5_counts.json` |
| G5 cheap talk | **FAIL** on the per-run test (p = 0.0076; per decision exact) | `gates/G5_cheap_talk.json` (`5fd9505`) |
| | investigated: sampling, wiring and end to end all pass | `gates/G5_cheap_talk_investigation.json` (`f2449ea`) |
| | Abdul's exception: accepted as chance | `f93170a` |
| | confirmation on fresh seeds: new 50 p = 0.744, pooled 100 p = 0.084 → **PASS** | `gates/G5_cheap_talk_confirmation.json` (`8807b9b`) |
| G6 pilot | 0 of 5 runs met the rule, so no change; payoffs frozen at (12, 10, 4, 0) | `pilot/summary.json` (`9edb338`) |
| drift | 0 in every condition (floor 0); the mixed-batch panel for cheap talk n = 10 was lost (disclosed) | `runs/*/*/drift.json`, `validation.md` |

### Runs (Llama 3.1 8B, every condition 50 runs, T = 20)

All runs share kernel fingerprint
`30c45b03786f3adb6be559aa1f78bcc1cf8071cc277ea9b772fd7a91e8c4e4cc` (prompts, model,
labels, payoffs, serving). Run fingerprints are in each condition's `provenance.json`.

| arm | condition | runs | status |
|---|---|---|---|
| 4 (calibration) | counts n = 4 cycle, last-1 | 50 | G5 |
| 4b (calibration) | cheap talk n = 4 cycle, last-1 | 50 + 50 confirmation | G5 + confirmation |
| 5 | counts n = 10 / n = 20 cycle, full history | 50 / 50 | secondary |
| 6 | cheap talk n = 10 cycle, full history | 50 | **primary contrast** |
| 6 | cheap talk n = 20 cycle, full history | 50 (two sessions) | secondary |
| 7 | counts / cheap talk n = 10 complete K_10 (d = 9, actions-only tables) | 50 / 50 | secondary |

**Exclusions:** none. No run was dropped. No probability failed the p_valid check.

### Results

- **Primary contrast** (pre-specified; arm 6, n = 10):
  - D = CRPS(actions-only) − CRPS(signal-aware) = **−0.000086 [−0.000221, +0.000035]**
    (95%, whole runs, conditional on the tables and one 20,000-run simulation).
  - The signal-aware audit does not forecast the endpoint detectably better.
  - The follow-up shows simulation noise is material. At 740,000 simulated runs,
    D = −0.000085 [−0.000225, +0.000054]; the nested interval is [−0.00047, +0.00009]. The
    null stands.
- **n = 20 (secondary):**
  - The published D is −0.00004 [−0.00011, +0.00003].
  - At 2,000,000 simulated runs, conditional on the tables: −0.00031 [−0.00045, −0.00017].
  - With table uncertainty the sign is not established (nested [−0.00081, +0.00008]).
- **Arm 7, K_10 (secondary).** Actions-only audit against order-0:
  - Cheap talk: CRPS 0.0596 against 0.0764; max Δ_t 0.228 against a null of 0.239.
  - Counts: CRPS 0.0405 against 0.0399; max Δ_t 0.216 against a null of 0.237.
  - On the complete graph the local audit helps on cheap talk but not on counts.
- **All conditions:** `tables/scores.md`, `tables/defects.md`, `figures/fig2.png`. Figure
  2's middle-panel labels overlap (cosmetic); the tables carry the same numbers.
- **Exploratory follow-up** (`followup/core/summary.md`):
  - The endpoint null hides an early-round over-prediction of high bids.
  - Matching the audit's history length to the round removes about two-thirds of that bias
    (Llama and OLMo). Later rounds get worse, which is consistent with a history-content
    shift (cause not established).
  - These results replicate at n = 20.

### Spend

$8.74 recorded in `spend.json` against the $40 cap. This is wall-clock time at the list
price, including queueing and the 10-minute idle window, so it overestimates. The Modal
dashboard of `aliyusufbelal` is authoritative. The follow-up used no GPU.

### Independent checks

- **Codex review** (`codex/independent-audit-20261108`): it reproduced the core headline
  numbers and found no critical error. Its material points on the core (MC error, the
  primary flag, causal wording) are handled; see `followup/core/codex_review_response.md`.
- **Independent-check branch:** it reproduced G5 and the arm 4b CRPS earlier (PR 8).

### Remaining work (for the authors)

- **The report itself.** The insert is `followup/core/report_insert.md`, and the figures are
  `followup/core/figures/` plus the extra arm's Figure 2. Use the official template, and
  include limitations and a dual-use appendix.
- **The theorem proof and setup-paper wording** (Codex review M3 and m3).
- **Related work.** arxiv.org was unreachable from this session. The Codex review's reading
  of Flint et al. shows high overlap, so make no "first" claims.
- **A content-matched audit** is the obvious next test of the follow-up's mechanism. It was
  not run.
