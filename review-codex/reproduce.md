# Reproduction — checkpoint 1

Python 3 with NumPy (already available). Offline CPU calculations only.

```sh
GIT_TERMINAL_PROMPT=0 git -c credential.helper= clone https://github.com/AbdulMusthakin/swarm-audit.git swarm-audit
cd swarm-audit
git switch -c codex/independent-audit-20261108 982822f554e547cc5c4468b70d5aaaae88356814
git worktree add --detach ../extra-snapshot 4a817051abffbdf208eb38a2b09160ab9d49aa0f
OPENBLAS_NUM_THREADS=1 python review-codex/independent/exact_review.py --source ../extra-snapshot --out review-codex/independent/olmo_exact.json --followups
```

The review files are supplied separately (bundle/commit); apply them before the final command.
Source refs before work: `independent/initial_refs.txt`. Core snapshot:
`f020fe2a0c93c42177a48ccdf9141ead9c185d1a`. These are fixed snapshots, not moving branches.

The independent script uses ring neighbours directly, reverses source action and signal
label order, builds product transitions, and propagates the generating polynomial of
the final-five-round high-bid count. A separate killed-chain calculation checks the
all-high event. All transition normalization and outcome-law checks passed at 3e-13.
It verifies every OLMo calibration log record, state, sampled draw, joint probability
and within-state prompt hash. Additional priorities are pending; see findings.md.

## Priority 1 additional commands

```sh
OPENBLAS_NUM_THREADS=1 python review-codex/independent/primary_review.py --source . --out review-codex/independent/llama_primary.json --table-resamples 200 --mc-batches 20
OPENBLAS_NUM_THREADS=1 python review-codex/independent/secondary_checks.py --main . --extra ../extra-snapshot --out review-codex/independent/secondary_checks.json
```

Both scripts are independent of project modules. `secondary_checks.py` imports only the
new review's exact calculation and SciPy for binomial intervals. All archived original
Llama predictor CDFs and paired per-run scores matched exactly; gate MC p-values
matched exactly. New intervals use independently documented RNG seeds and need not
have identical bootstrap endpoints to a 2,000-draw archived interval.
