# Validation

The gates of `docs/handoff-build.md` section 4, in order. Every number comes from a
logged run; probabilities are always the logged ones, never re-queried.

<!-- G1 start -->
## G1, unit tests: PASS

`python -m pytest -q`: 44 passed, 1 skipped (7 Nov 2026). The skipped test loads the real
Qwen3 tokenizer from Hugging Face, which the cloud container used for this build cannot
reach; it passed during screening. The section 8 tests are in `tests/test_build.py`.
<!-- G1 end -->

<!-- G2 start -->
## G2, harness check (CPU): not run yet
<!-- G2 end -->

## G3, serving repeatability: not run

Needs the main model confirmed and a Modal account (hand-off section 1).
Script: `scripts/check_serving.py`. The numerical floor will be recorded here.

## G4, throughput benchmark: not run

Script: `scripts/benchmark.py` (writes `results/serving/g4_benchmark.json`).

## G5, calibration (arms 4 and 4b): not run

Script: `scripts/analyse.py --gate-g5` (code path exercised on scripted policies in G2).

## Drift panel: not run

Recorded per condition in `results/runs/<arm>/<condition>/drift.json` by `scripts/run_swarm.py`.

## G6, pilot: waiting for Abdul's rule (or "none")
