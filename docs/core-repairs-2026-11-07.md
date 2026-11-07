# Core repairs after the independent review (7 Nov 2026)

For the extra-arms session (`docs/handoff-extra-arms.md`, branch `extra-arms`): merge
these from `claude/funny-turing-qv2y66` (`git merge origin/claude/funny-turing-qv2y66`)
rather than implementing your own versions. The core session owns these shared pieces.

## Commits

| item | commit | what changed |
|---|---|---|
| G6 pilot | `9edb338` | Payoffs frozen at (uLH, uHH, uLL, uHL) = (12, 10, 4, 0): 0 of 5 runs reached all-low consensus by round 5. No config change. |
| C1 | `a83410a` | Fresh measurements for repeatability, drift and timing; serving identity in the call-log key. |
| C2 | `470c271` | Provenance fingerprints for runs and tables, checked on resume and load. |
| C3, C5 | `04cfc68` | G5 gate records; core launch blocked without passing records; discrete Monte Carlo p-value. |
| C4, C6 | `846ce77` | Matched null for Δ_t; labelled analysis outputs; G2 rerun. |

## Schema and interface changes you will meet

- **`Backend.query(queries, fresh=False)`**: every backend takes `fresh`. Any server call
  wrapper must accept `fresh` and return `"replayed"` per result
  (`build.modal_backend` shows how). `KernelOut.replayed` is logged per decision as
  `"replayed"` in the run files.
- **`serving.call_logged(..., fresh=False, return_status=False)`**: `fresh=True` never
  replays. Log entries carry `"fresh_measurement"` and the `"serving"` identity.
- **`build.serving_identity(cfg)`**: hf_id, revision, method, dtype, engine kwargs, vLLM
  version, batch invariance, GPU, and `serving_code_sha256` (a hash of
  `swarm_audit/{serving,tokens,ladder,game,__init__}.py`).
- **`build.FreshBackend`, `build.panel_outputs`, `build.PanelMixer`**: drift panels and
  mixed batches are always fresh and raise if anything was replayed.
- **`swarm_audit/provenance.py`**: `kernel_identity(backend)`, `fingerprint(obj)`,
  `bind_run_provenance(...)`, `table_provenance(...)`, `check_tables_match_runs(...)`.
  - Each condition folder gets `provenance.json` **before** any run is computed:
    `fingerprint`, `kernel_fingerprint`, `payload` (kernel, design, sampling protocol),
    `created`, `git_commit`. It is never rewritten. Resuming with a different
    configuration raises `ProvenanceError`; runs without `provenance.json` cannot be
    resumed.
  - `manifest.json` entries are now `{"provenance", "sessions": [...], "summary"}`.
  - `AuditTable.meta["provenance"]` holds `kernel`, `kernel_fingerprint`, `design`,
    `fingerprint`. `scripts/analyse.py` refuses tables whose kernel differs from the runs'.
  - `RunData.meta` has `fingerprint`, `kernel_fingerprint` and `design`.
- **Gate records** (`calibration.gate_record`, `write_gate`, `require_g5`):
  `results/gates/G5_<channel>.json` plus `results/gates/history.jsonl`. `GATES_DIR` is a
  module variable and every function takes a `folder`, so the extra arms can keep their
  own under `results/extra/gates/`. `analyse.py --gate-g5` exits 1 on a failed required
  check.
- **`calibration.per_run`**: returns `p_value`, `alpha` (0.01) and `rule`; passes if
  p > alpha. `threshold_q99` is still reported.
- **`stats.null_reference(X_pool, law_ref, R, ...)`** replaces `stats.noise_floor` for
  Δ_t; `ModelRuns.null_pool` is a separate simulation pool. Analysis outputs carry
  `notes`, `is_primary_contrast` and a `measure` label on each defect.
- **`build.use_deployed(cfg)`**, `SWARM_AUDIT_DEPLOYED=0/1`, `scripts/deploy_server.py`,
  `build.check_deployed` (reads `build.DEPLOY_FILE`): if your configs copy
  `serving.deployed: true` from `core.yaml`, you need your own deployment record. Add a
  config key for the record path (default `results/serving/deployed.json`) rather than
  reusing the core's, which names workspace `aliyusufbelal`.
- **Spend ledger**: `build.record_spend` takes a file lock. `build.SPEND_FILE` is still
  a module constant; per your hand-off, add a config key defaulting to it.

## Signal responsiveness (core G4, for your arm 9 trigger)

Measured on the final cheap-talk prompt (`results/serving/signal_responsiveness.json`):
0.0553 at d = 4 (the screening grid, length 1; screening's 0.051 was on the old wording)
and 0.038 at d = 2 (the cycle's degree).

**Abdul's decisions (7 Nov 2026, 20:15), for the extra-arms session:**

1. **The trigger uses the d = 4 screening grid**, as committed in `a2657c5` before any
   result. 0.0553 > 0.05, so **the arm 9 trigger is NOT met. Record that** in
   `results/extra/README.md`.
2. **Separately, after arm 8 finishes, run arm 9 (9a, then 9b subject to the lock-in stop)
   as an EXPLORATORY arm**, decided after seeing Llama's d = 2 responsiveness (0.038) and
   the arm 4b CRPS null. **Label it exploratory** in `results/extra/arm9_summary.md` and in
   `results/extra/README.md`. Same design, cost cap and stop rules as
   `docs/handoff-extra-arms.md`.
