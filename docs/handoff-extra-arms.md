# Hand-off: extra arms on the second Modal account

For a second Claude Code session, working in this repository alongside the core build. Read this file in full before starting.

**When:** Saturday 7 November 2026, evening, during the Apart Research Swarm Alignment Hackathon. These arms were fixed at 18:00 on 7 November, before any core GPU data existed, so they cannot have been chosen for their results.

**Owner:** Abdul Musthakin. He owns the mathematics, the prompts and the design. The code may be entirely AI-generated. Do not edit anything under `setup-paper/` or `report/`.

**Relation to the core build.** Another Claude Code session runs `docs/handoff-build.md` (the core arms 1–7) on the branch `claude/funny-turing-qv2y66`, in the Modal workspace `aliyusufbelal`. This session runs **separate** arms on a **different** Modal account. Nothing here feeds the core's primary contrast, and nothing here may touch the core's runs, tables, spend ledger or workspace.

## 0. Read first

1. `docs/handoff-build.md` in full, including "Notes for a cloud session": network hosts, Modal sign-in with `scripts/modal_login.py`, the `huggingface` secret and `SWARM_AUDIT_HF_SECRET`.
2. `results/validation.md`: G1–G3 and the decisions settled in advance. The same rules apply here unless this file says otherwise.
3. `configs/core.yaml`: the frozen inputs (model, revision, labels, payoffs, T, p_test, audit sizes, seeds, serving, decisions).

## 1. Branch, paths and accounts

- **Branch:** work on `extra-arms`, which starts from the core branch and holds this file. First merge in the latest core commits (`git merge origin/claude/funny-turing-qv2y66`), and again whenever you need newer core code (`git merge origin/claude/funny-turing-qv2y66`); never push to the core branch. Abdul merges `extra-arms` at the end.
- **Shared code:** changes to `swarm_audit/` or `scripts/` must be additive, and must leave the core's behaviour unchanged. Where a script hard-codes a results path (for example `build.SPEND_FILE`, `table_dir`, `results/serving/`), add a config key or option whose default is the current path. Add a test for anything new.
- **Outputs:** everything under `results/extra/`, including its own spend ledger, `results/extra/spend.json`. Configs: `configs/extra-scale.yaml` and `configs/extra-olmo.yaml`, copied from `core.yaml` with only the fields below changed.
- **Modal workspace:** the second teammate's. In both configs set
  `forbidden_workspaces: [abdulmusthakin, aliyusufbelal]`, so these arms can never spend Abdul's account or the core's credits. Record the workspace name in `results/extra/README.md`.
- **Cost cap:** $25 for everything in this file, in the extra ledger. Use the G4 margin rule (projected cost × 1.25 must fit what is left). Ask before exceeding it.

## 2. Gates for this account

Run these in order. **If a gate passes by its stated rule, continue without waiting for Abdul.** Stop and report only on the stop conditions in §6.

- **X1, workspace set-up.** The teammate signs in through `scripts/modal_login.py` (they open the printed link in a private window, signed in to their own Modal account). Their workspace needs a Modal secret named `huggingface` holding `HF_TOKEN`, a Hugging Face token with access to `meta-llama/Llama-3.1-8B-Instruct`. Then run `scripts/setup_workspace.py` with the extra config: weights cached in this workspace's volume, and every label a single token with the configured id on the frozen prompts.
- **X2, serving repeatability on this account.** `check_serving.py` with batch invariance **on** (the core's G3 setting), on about 50 prompts inside a shuffled batch of about 500. The maximum difference must be 0. If it is not, apply the core's G3 rule (floor up to 0.07 continues; above that, stop). Write to `results/extra/serving/`.
- **X3, cross-account check.** Query the core's drift panel (`build.drift_panel(cfg)`) here, and compare it with the core's logged panel outputs once they exist on the core branch. Report the maximum difference. This tells Abdul whether the two accounts serve an identical kernel. It is informational, not a stop.

## 3. Arm 8: 100 agents on a random 4-regular graph (always run)

**Why:** it gives a result at swarm scale. A bigger ring adds little, because by round 20 an agent on a ring is influenced only by agents within 19 steps of it. On a random 4-regular graph of 100 agents the diameter is about 5 or 6, so influence spans the whole group within 20 rounds, while every agent keeps the same number of neighbours.

**Design (fixed):**

| Setting | Value |
|---|---|
| Model | Llama 3.1 8B Instruct, same revision, labels and frozen counts prompt as the core |
| Channel | Counts only, full history |
| Graph | One random 4-regular graph on 100 agents: simple and connected. Seed `20261108`, recorded; the same instance for every run and for the predictors. Save the edge list and diameter to `results/extra/graph_rr4_n100.json` |
| T | 20 |
| Runs | 50, `arm_id: 8` (distinct from every core arm id, so random streams never coincide) |
| Payoffs, temperature, p_test, sampling | As in `core.yaml`, unchanged. If the core's pilot applied the fallback (uLH = 11), use the core's frozen payoffs |

**Code:** add `topology.random_regular(n, d, seed)` and a `make("random_regular", n, d=..., seed=...)` path, with tests: every degree is 4, no self-loops or repeated edges, connected, same seed gives the same graph. Everything else (states, audit, prediction, analysis) already works from `g.regular_degree`.

**Audit:** the actions-only counts table at d = 4: states (own last bid, number of the 4 neighbours bidding high last round), 2 × 5 = 10, plus START. 100 histories per state to fit, 50 to score, as in the core. Run with `run_audit.py --channel counts --d 4`, writing to `results/extra/audit/`.

**Order:**
1. Fit and score the table.
2. **Benchmark:** run 5 of the 50 runs in lockstep, then project the cost and time of the remaining 45 from them. If the projection fits the cap and finishes by 10:00 on Sunday 8 November, continue. If more than one container helps and the code supports splitting a lockstep batch across containers, use up to 4; otherwise one.
3. The remaining 45 runs. `run_swarm.py` resumes from logged runs, so a dropped connection loses nothing.

**Analysis:** run the core analysis code on this condition. Report:
- the order-0, actions-only (k = 1) and fitted-from-swarm predictors: CRPS of the late high-bid count y (rounds 16–20, out of 500 decisions) and the mean error in F, with whole-run bootstrap intervals;
- Δ_t in the five bins against its noise floor;
- ε(p_test) against ε(p_swarm), the bound with its first saturation round, and the defect shift by round block;
- edge disagreement, consensus, and the shared-neighbour check by graph distance (1, 2, 3 or more).

Write `results/extra/arm8_summary.md`, with one figure of X_t (swarm against each predictor) and one of the defects. **Label it as its own arm:** its degree is 4, not 2, so it is not a size contrast with the core rings.

## 4. Arm 9: OLMo-3-7B on the signal comparison (conditional)

**Trigger, fixed now:** run this arm only if Llama's signal responsiveness on the final cheap-talk prompt, measured in the core's G4 (`results/serving/signal_responsiveness.json` on the core branch), is **below the 0.05 screening bar**. If that file doesn't exist yet when arm 8 finishes, measure it yourself on this account with `swarm_audit/responsiveness.py` on the same grid, and record that you did.

**Why:** Llama was only just over the bar in screening (0.051), and the reworded prompt may lower it. If Llama barely reacts to messages, the core's signal comparison will be a null by construction. OLMo-3-7B was the only screened model with a strong signal response (0.11). Its known risk is lock-in: in screening, an OLMo agent that bid low in cheap talk almost never bid high again (P(H) about 0.00001).

**Design (fixed):**
- **Model:** `allenai/Olmo-3-7B-Instruct`, revision `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` (from `configs/screening.yaml`). Not gated. Label ids from `results/screening/tokenizers.json`. The same frozen prompt templates as Llama; only the labels' token ids differ. Repeat X1 and X2 for this model first.
- **Arm 9a (`arm_id: 9`):** the cheap-talk calibration, exactly as the core's arm 4b: cycle, n = 4, last round only, 50 runs, the enumerated signal-aware table (108 states plus START) and the derived actions-only table. G5's checks apply as in the core.
- **Lock-in stop, fixed now:** if at least 45 of the 50 runs of 9a have every agent bidding low in all of rounds 16–20, stop after 9a and report. There would be nothing left to predict.
- **Arm 9b (`arm_id: 91`):** cheap talk, cycle, n = 10, full history, 50 runs. The signal-aware audit at d = 2 (50 to fit and 25 to score per state, or the core's reduced 25/15 if the cost projection needs it), with the actions-only table derived from it.
- **Analysis:** the same paired CRPS difference as the core's primary contrast (actions-only minus signal-aware), with whole-run bootstrap intervals, on 9a and 9b, plus each predictor's absolute CRPS. Write `results/extra/arm9_summary.md`. **Label it as a replication on a second model,** not part of the core's primary result.

If the trigger is not met, write one line saying so in `results/extra/README.md` and don't run arm 9.

## 5. Persistence

- The logs follow the core's format (`docs/handoff-build.md` §9), under `results/extra/runs/<arm>/<condition>/`, with a `manifest.json` per arm: the workspace, the git commit, the model and revision, the vLLM version, the batch-invariance flag, the seeds, the graph seed and timings.
- **Commit and push to `extra-arms` after each gate and each arm.** A cloud session's disk is temporary.
- Keep `results/extra/README.md` current: what ran, where each file is, and the spend so far.

## 6. Limits and stop conditions

- **No tuning towards a result.** Don't change payoffs, temperature, prompts, the graph seed or run counts. Any change is a new condition: log it and tell Abdul.
- **No invented numbers.** Every figure comes from logged data.
- **No hosted APIs.** Modal vLLM only.
- **Stop and ask Abdul if:**
  - X1 or X2 fails, or G5 fails on 9a
  - the cost would pass the $25 cap
  - the active Modal workspace is a forbidden one
  - drift exceeds the floor by more than 0.035
  - something here contradicts `docs/handoff-build.md` or the paper's definitions

## 7. Done when

- [ ] X1–X3 reported.
- [ ] Arm 8: table fitted and scored, 50 runs logged, analysis and `arm8_summary.md` pushed.
- [ ] Arm 9: trigger checked and recorded; if met, 9a run (and 9b unless lock-in stopped it), analysed and pushed.
- [ ] `results/extra/README.md` lists every output and the total spend.
