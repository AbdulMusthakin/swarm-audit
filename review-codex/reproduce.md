# Reproduction, source snapshots and test results

All audit computations are offline CPU work. No live model query, paid compute,
Modal operation or original-data mutation is required. The reviewer used an isolated
branch based on main `982822f554e547cc5c4468b70d5aaaae88356814`.

## Source snapshots

| Use | Branch | Commit |
|---|---|---|
| Original primary and theorem | main | 982822f554e547cc5c4468b70d5aaaae88356814 |
| Initial core follow-up code | claude/funny-turing-qv2y66 | f020fe2a0c93c42177a48ccdf9141ead9c185d1a |
| Initial OLMo results and E1/E2 | extra-arms | 4a817051abffbdf208eb38a2b09160ab9d49aa0f |
| Final one-time core refresh | claude/funny-turing-qv2y66 | 3912f32915bea8f3850c30b779e3be50522f5108 |
| Final one-time extra refresh | extra-arms | 76a647ec5f62b361f4048dbca05d30956e8e42de |

`initial_refs.txt` and `final_refs.txt` preserve other branch heads. Inventory and
provenance JSONs contain exact paths, SHA256 hashes, run ranges and kernel identities.
Initial source snapshots were reviewed first. The remote branches were fetched
once near the end; new results are clearly separated in `new_results_review.json`.
No uncommitted job output was read and no jobs were polled.

The initial snapshot has complete Llama n10 both channels and counts n20, original
and confirmation n4 calibration, OLMo n4 and n10, and n100 extra counts. Core n20
cheap talk has 25 committed runs at the initial core snapshot; no completed analysis
was claimed. The initial E1/E2 are exact cached-kernel counterfactuals, not live
experiments. The final extra snapshot adds 50 live n10 last1 OLMo runs; n4 full and
E2 live outputs are absent there. The final core snapshot adds C1/C2 evidence packages.

## Restore and calculate

The review branch is published, after the user's subsequent explicit instruction
to make the commits visible to the repository. No merge or change to another
agent's branch was performed.

```sh
git clone https://github.com/AbdulMusthakin/swarm-audit.git swarm-audit
cd swarm-audit
git switch codex/independent-audit-20261108
git worktree add --detach ../extra-snapshot 4a817051abffbdf208eb38a2b09160ab9d49aa0f
git worktree add --detach ../core-snapshot f020fe2a0c93c42177a48ccdf9141ead9c185d1a
git worktree add --detach ../core-final-snapshot 3912f32915bea8f3850c30b779e3be50522f5108
git worktree add --detach ../extra-final-snapshot 76a647ec5f62b361f4048dbca05d30956e8e42de
python -m venv --system-site-packages ../review-venv
../review-venv/bin/python -m pip install numpy scipy
OPENBLAS_NUM_THREADS=1 ../review-venv/bin/python review-codex/independent/exact_review.py --source ../extra-snapshot --out review-codex/independent/olmo_exact.json --followups
OPENBLAS_NUM_THREADS=1 ../review-venv/bin/python review-codex/independent/primary_review.py --source . --out review-codex/independent/llama_primary.json --table-resamples 200 --mc-batches 20
OPENBLAS_NUM_THREADS=1 ../review-venv/bin/python review-codex/independent/secondary_checks.py --main . --extra ../extra-snapshot --out review-codex/independent/secondary_checks.json
../review-venv/bin/python review-codex/independent/provenance_checks.py --main . --extra ../extra-snapshot --out review-codex/independent/provenance_checks.json
OPENBLAS_NUM_THREADS=1 ../review-venv/bin/python review-codex/independent/extra_review.py --main . --extra ../extra-snapshot --out review-codex/independent/extra_review.json
OPENBLAS_NUM_THREADS=1 ../review-venv/bin/python review-codex/independent/new_results_review.py --core ../core-final-snapshot --extra ../extra-final-snapshot --out review-codex/independent/new_results_review.json
../review-venv/bin/python review-codex/independent/inventory.py --root . --root ../core-snapshot --root ../extra-snapshot --out review-codex/independent/inventory.json
../review-venv/bin/python review-codex/independent/build_claims.py
```

The scripts import only standard Python, NumPy, SciPy and review-only modules;
they do not import project prediction, analysis or statistics functions. For
original results the review worktree's non-review files are unchanged from main's
fixed source snapshot. Inventory records its current review HEAD as well.

Used versions: Python 3.12.14; NumPy 2.3.5; SciPy 1.17.0. Optional repository unit
checks also used pandas 2.2.3, PyYAML 6.0.3, pytest 9.2.0 and Modal SDK 1.6.3.
The SDK was installed solely to satisfy offline import/unit-test dependencies;
no authentication or remote operations were performed.

## Independent checks

- Exact OLMo computation reverses source action/signal ordering, builds product
  transitions from explicit ring neighbours, and propagates the final-five count
  by a probability-generating polynomial. A separate killed-chain calculation
  verifies the all-high event. Row/law errors are below 3e-13.
- All original Llama primary predictor CDFs and paired per-run CRPS scores reproduce
  exactly (maximum error 0). New MC batches use independent documented seeds.
- Original, confirmation and pooled calibration p-values reproduce exactly with
  the original null seed and 20k draws. New bootstrap endpoints need not equal the
  archived 2k-draw endpoints, because their seeds and draw counts differ.
- Paired table bootstrap resamples intact six-vectors, fixes START and rederives
  actions-only forecasts, preserving dependence between fitted forecasts.
- Logged run records, states, draws and seed streams were rebuilt. Fingerprints,
  template hashes, observed enumerated prompt hashes and softmax reconstruction
  from saved logprobs pass. These are evidence-integrity checks, not a fresh
  verification of model weights or serving repeatability.
- Final-refresh C2 and E3 checks have their own source commits and outputs. See
  `new_results_review.json` and the final findings addendum for practical limits.

## Existing repository unit checks

After resolving missing local pytest/Modal import dependencies in a separate venv:

```sh
OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../review-venv/bin/python -m pytest -q tests/test_build.py tests/test_review_c1.py tests/test_review_c2.py tests/test_review_c3.py tests/test_review_c4.py tests/test_decisions.py tests/test_settled.py tests/test_extra.py
```

At main base: **72 passed, 8 skipped**, 43.61s. Skipped tokenizer-dependent tests
reflect missing optional tokenizer/transformer packages; no live inference test was
attempted. Existing tests emit empty-slice warnings for short synthetic test runs
and a pytest plugin deprecation; no failed assertion.

`tests/test_followup.py` at initial extra `4a81705`: **5 passed**.
`tests/test_followup.py` at initial core `f020fe2`: **11 passed**.
These were run in detached worktrees. No original tracked file changed. This is a
focused offline verification set, not a claim that the full optional/GPU suite ran.

## Checkpoints, authorisations and continuation

Local checkpoints: `7fc8a27` (first exact result), `a441a69` (Priority 1), `7ae30da`
(Priority 2), `03f344c` (findings/report), `eb2dc4c` (claim table/source refresh).
Subsequent commits add final-refresh checks and final packaging. User first
allowed coordination where problems require other agents' action, then explicitly
asked for repository visibility; the isolated review branch was pushed only after
that instruction. The original no-merge/no-agent-branch/no-paid-compute boundaries
remain in force.

Coordination note: https://github.com/AbdulMusthakin/swarm-audit/pull/9#issuecomment-6715458905
Review branch: https://github.com/AbdulMusthakin/swarm-audit/tree/codex/independent-audit-20261108/review-codex

A continuation was scheduled for 05:30 Europe/London on 8 Nov. Its
future run must verify file/environment access before relying on local state.
Shared local backup: `/workspace/shared/swarm-audit-review/review.bundle` and
checkpoint archives. The prescribed projectless output path under `/codex/browser/`
was unavailable in this cloud executor (read-only/permission-denied root), so these
are preserved in the workspace and the now-authorised repository branch instead.

## Continuation: 05:30 Europe/London, 8 Nov

One new research-ref fetch selected core `f8ad9bbb88ee538d1c9563c51e0a1d97e4401711`
and extra `9aa31705b9ae5ee45f27a87feeb1a2e2415f318f`. Ref records are in
`independent/continuation_20261108/refs.txt`. No later research-branch polling is used.

```sh
git worktree add --detach ../core-continuation-snapshot f8ad9bbb88ee538d1c9563c51e0a1d97e4401711
git worktree add --detach ../extra-continuation-snapshot 9aa31705b9ae5ee45f27a87feeb1a2e2415f318f
OPENBLAS_NUM_THREADS=1 python review-codex/independent/continuation_olmo.py --source ../extra-continuation-snapshot --out review-codex/independent/continuation_20261108/olmo.json
```

This uses the prior independently calculated exact laws after checking that the
current enumerated kernel has the same byte hash. New intervention live logs and
the full-memory n4 live cell are checked independently. Existing E3 outcomes are
re-read to check the completed table, without rerunning their prior integrity audits.

Additional continuation commands:

```sh
OPENBLAS_NUM_THREADS=1 python review-codex/independent/continuation_llama.py --source ../core-continuation-snapshot --out review-codex/independent/continuation_20261108/llama.json
OPENBLAS_NUM_THREADS=1 python review-codex/independent/continuation_provenance_probe.py --source ../extra-continuation-snapshot --out review-codex/independent/continuation_20261108/provenance_guard.json
```

`continuation_llama.py` reuses only review-owned independent code, generalises the
score support to n20 and independently implements the complete-graph chain. It
reproduces archived CDFs before using fresh seeds for the precision sensitivity.
The n20 table-bootstrap arrays are source evidence, not an independently rerun
sensitivity in this continuation.

The provenance probe is deliberately different: it imports the actual project
runner under test and uses its scripted CPU backend in a temporary directory.
It never invokes the LLM backend or a remote service. It demonstrates the missing
observer binding and leaves all original run evidence unchanged.

Continuation regression checks at core `f8ad9bb`:

```sh
(cd ../core-continuation-snapshot && OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ../review-venv/bin/python -m pytest -q tests/test_followup.py tests/test_followup_core.py)
```

**22 passed**, 2.28s, one existing pytest-plugin deprecation warning. This includes
primary classification, first-entry and observation-hook tests. A preliminary
focused selection passed3 tests; the complete two-file set is the recorded result.
The new independent log audits cover60k decisions across the newly checked cells;
whole-run uncertainty remains computed per condition, not by treating decisions
as independent samples. All original tracked files and detached snapshots remained
unchanged. New machine outputs and claim-table validation passed.

Known limits of this continuation: no fresh server/weight check; no independent
rerun of the n20 table-bootstrap arrays; no exhaustive novelty search or final
paper rewrite. The original primary comparison and calibration chronology were
not recomputed unnecessarily. PR9 responses were checked once; the later update
posts the new minor safeguard finding and the completed checks.
