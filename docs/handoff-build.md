# Hand-off: build and run the experiment

For Claude Code, working in this repository. Read this file in full before starting.

**When:** from Saturday 7 November 2026, during the Apart Research Swarm Alignment Hackathon. The Warwick team chose this project on 7 November. Code written during the sprint needs no declaration; anything reused from before Friday 6 November does (see the README).

**Owner:** Abdul Musthakin. He owns the mathematics, the prompts, the design decisions and all paper text. The code may be entirely AI-generated. Do not edit anything under `setup-paper/` or `report/`.

**Branch:** start from this branch (`build-handoff`), which contains the screening code. Work on your own branch and push as you go.

## 0. Read first

1. `docs/handoff-model-screening.md`: what already exists (`game.py`, `ladder.py`, `tokens.py`, `serving.py`, the prompt drafts, the tokenizer screen).
2. `results/screening/summary.md` and `results/screening/runs/*/summary.md`: the screening results so far.
3. `setup-paper/sections/2-setup.tex` and `4-experiment.tex`: definitions and notation (`K_i`, `kappa_i`, `p_test`, `q`, `pi_k`, `rho_i`). Use the same names in code where natural.
4. **The teammates' scoping study, if it is in the repo** (`scripts/abdul_openrouter_study.py`, `scripts/abdul_adaption_analysis.py` and their outputs). It ran the counts calibration on a hosted API (OpenRouter, Gemma 4 26B A4B, n = 4, 8 rounds), and the frozen audit missed. The cause was the serving, not the agents: for the most common history, the same prompt got P(H) = 12.5% in isolation and 18.7% during live games. Reuse their state enumeration, exact simulator and calibration test where they fit, and note the reuse in the README. Don't reuse their hosted-API backends.

**This file supersedes the paper on the run matrix, the calibration arm, the replay step, the sampling protocol and the choice of graphs.** These changes follow an external review of the design on 7 November. Where the paper and this file disagree, follow this file. If something here contradicts the paper's *definitions*, stop and ask Abdul.

## 1. Inputs from Abdul

Build everything that runs on CPU without waiting for these. Stop and ask before any GPU run that needs one of them.

1. **The main model.** It comes from the screening. Until then, develop against the scripted policies (§6) and, for prompt plumbing, the CPU stand-in used in screening.
2. **Approved prompt templates** (`swarm_audit/prompts/counts.txt`, `cheap_talk.txt`). Open questions are listed at the top of each file. Two matter most for this build:
   - Does "I'll bid high" refer to this round's bid (sent with it) or next round's? The current draft leaves this open.
   - Do agents see their own past messages? The current draft shows them. This decides whether the signal-aware state has 36 or 108 values (§5), and how many prompts arm 4b enumerates.
3. **The pilot rule.** Either "no adjustment", or one rule for adjusting the payoffs, fixed before the pilot (§4, gate G6).
4. **Whose Modal account** to use, as in the screening hand-off. Never use Abdul's own account without asking.

## 2. Design

- **Game:** as in the screening hand-off, section 1. Payoffs `(uLH, uHH, uLL, uHL) = (12, 10, 4, 0)` unless the pilot rule changes them. The payoff shown to an agent is always computed from its action and its neighbours' actions with `game.u`.
- **Graph:** the undirected cycle C_n, read as directed both ways. Every agent has in-degree 2, so size changes without degree changing. The optional second topology is the complete graph K_10.
- **Channels** (the ladder steps): `counts`, and `cheap_talk` with three signals (high, low, none).
- **Memory:** core agents see their full history. Calibration agents see only the last round, via `ladder.render(..., last_k=1)`.
- **Horizon:** T = 20, never disclosed. Round 1 is the prompt alone.
- **Agents act independently given the past.** Each run has one random stream, `numpy.random.Generator(PCG64(SeedSequence([base_seed, arm_id, condition_id, run_index])))`. Agents draw from it in a fixed order. vLLM's own sampling is never used, and no seed is shared across agents.

### Run manifest (the only one)

| Order | Arm | What runs | n | Graph | Runs | Notes |
|---|---|---|---|---|---|---|
| 1 | Harness check | Scripted policies (§6) | 4, 10 | Cycle | Many, CPU only | Before any LLM call |
| 2 | Serving check | Repeatability of probabilities (§4, G3) | — | — | About 200 prompts | Gate |
| 3 | Audit tables | Actions-only and signal-aware tables (§5) | — | Degree 2 | See §5 | One table serves n = 10 and 20 |
| 4 | Calibration | Counts, last round only | 4 | Cycle | 50 | Exact check (G5) |
| 4b | Calibration | Cheap talk, last round only | 4 | Cycle | 50 | Agents forget nothing, so the signal-aware audit should be exact and the actions-only one need not be: the cleanest test of dropped signals (G5) |
| 5 | Core | Counts | 10, 20 | Cycle | 50 each | Launch once G1–G6 pass |
| 6 | Core | Cheap talk | 10, 20 | Cycle | 50 each | The key arm |
| 7 | Second topology | Both channels, actions-only audit | 10 | Complete | 50 each | Only if Abdul approves after the G4 benchmark |

**Cut order if time runs short:** arm 7, then n = 20 in both channels. Never cut arms 1–4b.

**Dropped:** the replay step, chain and star graphs, n = 40 and 100, the k = 2 calibration arm, free text, the second model family and the 70B model.

**No hosted APIs.** Every scored run, audit and calibration uses the Modal vLLM server. The teammates' pilot showed that a hosted endpoint's probabilities can shift between isolated and live calls on identical prompts. Leave their config files in place, with a comment at the top saying they are superseded by this hand-off.

## 3. One decision, exactly

For agent i in round t of a run:

1. Render its prompt from its history h_i^t (its records for rounds 1 to t−1) with `ladder.render`. Then chat-template and tokenize with `tokens.chat_prefix` and `tokens.encode`, as in screening.
2. Get the action probabilities from `Server.label_logprobs(prompts, action_ids)` (method `logprob_token_ids`). Batch **every agent of every concurrent run** into one call per round: runs are independent, so run all R runs of a condition in lockstep.
3. Sample the action in Python: `a = "H" if u < p_H else "L"`, with `u` from the run's stream. Log `p_H`, both raw log-probabilities, `p_valid`, `u` and `a`.
4. **Cheap talk only:** query the signal distribution after **both** action labels (`Rendered.signal_prefix(label)` for H and for L), so the full six-way output kernel P(a) · P(m | a) is logged. Sample the signal after the sampled action, with the same stream. Log both signal distributions, their `p_valid`, the draw and the signal. Prefix caching makes the extra calls cheap.
5. Once every agent has acted, build each agent's record for round t from its in-neighbours' outputs (action counts and, on cheap talk, signal counts). Rounds are synchronous: nothing from round t is visible to anyone in round t.

**The logged `p_H` is the kernel K_i(H | h).** Every later step uses the logged numbers, never a re-query. If `p_valid` falls below 0.9 anywhere, flag it in the run summary; don't drop the decision.

## 4. Serving, and the gates

### Changes to `swarm_audit/serving.py`

- Add a config flag that sets `VLLM_BATCH_INVARIANT=1` in the image environment. In vLLM 0.32 this is a beta feature, supported on GPUs of compute capability 8.0 or higher; the L4 is 8.9. Off by default until G3 decides.
- Allow `max_containers` above 1 when the arm is split across containers, up to the account's GPU concurrency limit.
- Keep `max_model_len` at 4096 unless a round-20 cheap-talk prompt at n = 20 exceeds it. Measure this in G4.

### Gates, in order

Report to Abdul after each gate. If a gate fails, stop and report; don't work around it silently.

- **G1, unit tests.** `pytest` passes, including the new tests in §8.
- **G2, harness check (CPU).** Run the full pipeline (swarm, audit, prediction, analysis) on the scripted policies of §6, through the same code path the LLM uses. Each policy must give its expected result.
- **G3, serving repeatability (GPU, main model).** Take about 200 prompts: audit prompts and swarm-like prompts from rounds 1–20, both channels. Query each (a) alone, (b) inside a batch of about 1,000 other prompts in shuffled order, and (c) after warming the prefix cache. Report the maximum difference in `p_H`, and in each signal probability, between any two of these calls.
  - **If it is 0,** keep the default settings.
  - **If not,** repeat with batch invariance on. Report both maxima and the throughput of each. Abdul decides which to use, and the chosen setting's maximum difference is recorded as the numerical floor in `results/validation.md`.
  - The earlier vLLM-against-transformers check stays informational: every gap seen in screening was the logit difference landing one bf16 step (0.125) apart.
- **G4, throughput benchmark.** One full core run each of counts at n = 10 and cheap talk at n = 20. Report decisions per second, prompt tokens at round 20, wall-clock time and dollars per run. Then extrapolate to the whole manifest and report before launching anything larger.
- **G5, calibration (arms 4 and 4b).** Cycle, n = 4, last round only. The prompt is then a function of the state, so query each distinct prompt once to get the table: 6 states plus round 1 on counts; 36 or 108 signal-aware states plus round 1 on cheap talk. Two checks per arm:
  - **Per decision:** every logged probability in the 50 runs (`p_H`, and on cheap talk both signal distributions) must equal the table entry for its state, up to the G3 floor. Prompts with the same state must also be byte-identical.
  - **Per run** (the teammates' test): let y be the number of high bids in rounds 16–20, out of 20 decisions. Compute the largest gap between the empirical distribution function of y over the 50 runs and the exact one from the chain (§7). Compare it with the 99th percentile of the same statistic over 20,000 simulated sets of 50 runs from that chain. The gate passes if the gap is below that threshold.

  On arm 4b the gate uses the **signal-aware** chain. The actions-only chain's result there is a finding, not a gate.

  A miss means investigating, not automatically fixing: differing prompt bytes, a leaked round number, drift (below), the wiring, then sampling variation. Report what you find.
- **Drift panel, in every arm from G5 on.** Fix a panel of about 50 prompts: every calibration state, plus a spread of core audit prompts from both channels. Query the panel at the start and end of each arm, and mix it once into a full-size swarm batch. Any change beyond the G3 floor goes in `results/validation.md` and is reported to Abdul at once.
- **G6, pilot.** If Abdul gives a payoff rule, apply it once, using 5 counts runs at n = 10, then freeze the settings for **both** channels. Temperature stays at 1. Save the pilot logs to `results/pilot/` and report them. Pilot runs never enter the analysis. With no rule, skip this gate.

Launch the counts core (arm 5) as soon as G1–G6 pass, then the cheap-talk core (arm 6). Don't wait for Sunday.

## 5. Audit tables (`swarm_audit/audit.py`)

All tables are fitted on synthetic histories from one agent alone. Swarm data never fits them, except the fitted-from-swarm baseline (§7).

**States on the cycle (degree 2):**
- **Actions-only:** s = (own last action, neighbours bidding high last round), giving 2 × 3 = 6 states, plus START (round 1).
- **Signal-aware:** s′ = s plus the neighbours' signal counts last round (6 possible counts over 2 neighbours), giving 36 states. If agents see their own past messages, add the agent's own last signal: 108 states. Plus START.
- **K_10 (arm 7 only):** actions-only, 2 × 10 = 20 states, plus START.

**The synthetic distribution p_test.** This is part of the design, so put its parameters in the config and describe them in `results/audit/README.md`.
- The number of past rounds L is uniform on {1, …, 19}. START is the single empty-history prompt.
- The last round is set to the target state. In each earlier round, the agent's own action is uniform on {H, L}, and each neighbour's action is independently uniform.
- On cheap talk, every message not fixed by the target state is uniform over the three signals, the agent's own included. That holds even for the last round when the state is actions-only, so the actions-only table averages over them.
- Flag this choice to Abdul in your first report; he may change it.

**Sizes.** Two disjoint sets per state, from different seeds:
- **Actions-only, counts:** 100 histories to fit κ and 50 to score ε(p_test).
- **Signal-aware:** 50 to fit and 25 to score.

At 108 states with three calls each, the signal-aware table is about 24,000 short calls, so report its cost in G4 before running it.

**Fitting.** κ(H | s) is the mean of the logged `p_H` over the fitting histories in state s. For the signal-aware table, κ′(a, m | s′) is the mean of P(a | h) · P(m | h, a). Save per state: κ, the number of histories, its standard error, and every per-history probability.

**Actions-only on cheap talk: derived, not separately queried.** Build it by marginalising the signal-aware table over the signal part of the state:

    κ(H | s) = Σ_{s′ → s} w(s′ | s) κ′(H | s′)

Here w(s′ | s) is the probability that p_test gives the signal part of s′, given s. So both predictors come from the same isolated prompts and are paired. This saves the separate actions-only cheap-talk audit. Score the derived table's ε(p_test) on the signal-aware scoring set, weighted the same way.
- **Primary weights:** p_test's own, each message uniform, so the neighbour signal counts follow a multinomial.
- **Sensitivity check** (the teammates' choice): equal weight on each observable signal configuration. Report both; Abdul picks if they disagree.
- **Arm 4b:** the same construction from the enumerated last-round table.

**Order-0 baseline.** κ₀(H) is the mean of `p_H` over the whole fitting set of that channel, weighted as p_test generates the states.

## 6. Scripted policies (`swarm_audit/policies.py`, new)

Make the backend an interface: given a batch of (step, history, d), return P(H) and, on cheap talk, P(m | a) for both actions. The LLM backend renders and queries. The scripted backend computes these directly. The swarm, audit, prediction and analysis code must not know which backend it is using.

| Policy | Rule | Expected result |
|---|---|---|
| Coin | P(H) = 0.5; uniform signals | ε = 0; the gap Δ stays inside the noise floor |
| Copy | P(H) = 0.8 if both neighbours bid high last round, 0.5 if one did, 0.2 if none did; 0.5 in round 1 | Order-1 in actions, so actions-only ε = 0 and Δ inside the noise floor, for full-history agents too (Proposition 2) |
| Two-round memory | Like Copy, but averaged over the last two rounds | Actions-only ε > 0, and the bound is above the measured Δ |
| Hidden signal | On cheap talk: P(H) = 0.8 if any neighbour said "high" last round, else 0.2; agents say "high" with probability 0.7 after bidding H, 0.1 after L | Actions-only ε > 0; signal-aware ε = 0, and its chain's gap is inside the noise floor |

Run each policy at n = 4 and 10 on the cycle with 200 runs. At n = 4, also compare with the exact law (16 states, or 1,296 with signals). Report a small table of expected against observed values.

## 7. Prediction and analysis

### Predictors (`swarm_audit/prediction.py`, `baselines.py`)

Each predictor is simulated for 20,000 runs on CPU, using the same graph, T and START handling as the swarm.
- **Order-0:** every agent bids H with probability κ₀(H), independently.
- **Actions-only, k = 1:** the chain on the last action profile, using κ(H | s).
- **Signal-aware, k = 1 (cheap talk only):** the chain on the last action-and-signal profile, using κ′(a, m | s′). This is an autonomous prediction. Separately, report the *one-step* comparison: on the logged swarm histories, the mean |p_H − κ′(H | s′)| against the mean |p_H − κ(H | s)|. Label it as one-step (teacher-forced), not as a group prediction.
- **Fitted from the swarm:** κ_swarm(s) is the mean of the **logged** `p_H` over agent-rounds in state s in runs 1–25 of the same condition. Score it on runs 26–50. Also score every other predictor on runs 26–50 in a separate table, and note that this baseline uses swarm data.
- **Exact law at n = 4:** propagate the chain exactly, with START at round 1, and give the law of X_t for each t and of y. The actions-only chain has 16 states; the signal-aware chain on arm 4b has 6⁴ = 1,296 (each agent's last bid and message).

### Statistics (`swarm_audit/stats.py`)

Per run:
- **X_t**, the fraction bidding high in round t.
- **y**, the number of high bids in rounds 16–20 (the primary score uses its distribution), and **F** = y / 5n, the mean of X_16 to X_20.
- **Edge disagreement**, the fraction of edges whose two ends bid differently, per round and averaged over rounds.
- **Consensus:** the first t ≤ T − 2 with X_t = X_{t+1} = X_{t+2} ∈ {0, 1}, recorded with its side (high or low). Runs without one are censored at T. Report the event probability by side and a survival curve. Never average a time that can be infinite.

### Estimands

- **Score:** the continuous ranked probability score (CRPS) of each predictor for each swarm run r. Let y_r be the number of high bids in rounds 16–20, out of 5n decisions, and let G be the predictor's distribution function of y (from 20,000 chain runs, or exact at n = 4). Then

      CRPS_r = (1 / 5n) Σ_{k=0}^{5n} (G(k) − 1[y_r ≤ k])²

  Lower is better. It scores the whole predicted distribution, not only its mean. A signal that two agents share changes the spread of y, not its mean, so a mean-based score would miss it.
- **Primary contrast:** on the cheap-talk swarm at n = 10, the paired difference D_r = CRPS_r(actions-only) − CRPS_r(signal-aware), averaged over runs. Give a 95% interval from resampling **whole swarm runs** (2,000 resamples). Positive values favour keeping the signals. Report each predictor's absolute CRPS alongside. The model side is treated as fixed; say that intervals are conditional on the fitted tables.
- **Key secondary:** the same paired difference on arm 4b (n = 4, last round only), where no history is forgotten.
- **Secondary:** E_swarm F − E_model F for each predictor, with the same bootstrap, where F is the mean of X_16 to X_20.
- **Secondary:** counts against cheap talk; n = 10 against 20; edge disagreement; per-round curves of X_t.
- **Binned gap Δ_t:** the total-variation distance between the swarm and model laws of X_t, in five fixed bins, [0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8) and [0.8, 1]. Show it against a noise floor: the 95th percentile of the same distance between two independent samples of R runs from the model, over 1,000 draws.
- **Defects:**
  - ε(p_test): per state, on the scoring set.
  - ε(p_swarm): per agent and round, from logged probabilities, |p_H − κ(H | s)|, averaged over runs. On cheap talk, also the six-way output distance for the signal-aware table.
- **Bound:** B_t, the sum of ε(p_swarm) over rounds up to t and over agents. Report it uncapped and capped at 1, with its first saturation round and a run-level bootstrap interval. Call it an estimate of the bound, not a certificate.
- **Defect shift:**

      ε(p_swarm) − ε(p_test) = Σ_s [w_P(s) − w_Q(s)] e_Q(s) + Σ_s w_P(s) [e_P(s) − e_Q(s)]

  Here w_P and w_Q are the state frequencies under the swarm and under p_test, and e_P and e_Q are the mean defects within a state. The first term is changed state frequencies; the second is changed histories within a state. Compute it pooled, and for round blocks 2–7, 8–13 and 14–20.
- **Shared-neighbour check:** agents i − 1 and i + 1 share neighbour i, so they see the same messages from it. For each round, compare how often the two bid alike in the swarm with each predictor's rate. Do the same for adjacent agents and for agents 3 or more apart as references. Use whole-run bootstrap intervals; on the 4-cycle, agents 2 apart share both neighbours. A swarm rate above the actions-only rate is the signature of a common signal the audit dropped.

### Figures and tables

- **Figure 1:** X_t for the swarm against each predictor, with run-level bands. Put the bound in a separate panel, because its scale hides the measured gap.
- **Figure 2:** the paired CRPS differences (core n = 10 and 20, arm 4b) with intervals; ε(p_test) against ε(p_swarm) by channel and size; and actions-only against signal-aware defects on the same histories.
- **Validation table:** G2's expected and observed results, G3's floor, G5's checks, the drift panel, the number of runs, and whether probabilities were logged or re-queried (always logged).

## 8. Code layout and tests

**Modules:**
- `swarm_audit/topology.py`: `cycle(n)`, `complete(n)`, in-neighbours, edges.
- `swarm_audit/swarm.py`: lockstep runs, logging, resuming from logged runs.
- `swarm_audit/audit.py`, `policies.py`, `prediction.py`, `baselines.py`, `stats.py`, `plots.py`.

**Scripts:**
- `scripts/run_swarm.py`, `run_audit.py`, `analyse.py`.
- New: `scripts/check_serving.py` for G3 and `scripts/benchmark.py` for G4.
- Mark `run_replay.py` and `coverage.py` as superseded.

**Configs:** `calibration.yaml`, `core.yaml` and `secondary.yaml`. Each records the model and revision, labels, payoffs, seeds, run counts and audit sizes.

**New tests:**
- Cycle and complete-graph degrees and neighbours.
- A round's records built from outputs on a 4-cycle, checked against hand-computed counts.
- Payoffs derived from counts.
- RNG: the same seed reproduces a run, and different agents get different draws.
- `q`, `pi_k` and both state functions.
- Calibration prompts: two histories with the same last round render byte-identical prompts, with no round number.
- The synthetic generator: the last round equals the target state, and lengths stay in range.
- The exact n = 4 law sums to 1 and matches a long simulation of the Copy policy; the 1,296-state law matches a simulation of the Hidden-signal policy.
- CRPS on a hand-computed example, and CRPS = 0 when the prediction puts all its mass on the observed y.
- The marginalisation weights w(s′ | s) sum to 1 for every s, under both weightings.

## 9. Outputs and persistence

`data/` is ignored by git, and a cloud session's disk is temporary, so write everything to keep under `results/` and **commit and push after each arm**.

- `results/runs/<arm>/<condition>/run_<r>.jsonl.gz`, one line per decision, with these fields:
  - arm, channel, n, graph, run, round and agent
  - the agent's record for the round and its coarse state
  - the prompt's sha256 and token count
  - `p_H`, both log-probabilities, `p_valid`, the draw and the action
  - on cheap talk, both signal distributions with their `p_valid`, the draw and the signal
- `results/runs/<arm>/manifest.json`: config, git commit, model and revision, vLLM version, engine settings, the batch-invariance flag, seeds and timings.
- `results/audit/`: the tables and per-history probabilities. `results/validation.md`: the G2–G5 results. `results/figures/`, `results/tables/`, and a `results/README.md` saying what each file is.

## 10. Limits and stop conditions

- **Cost cap:** $20 of Modal GPU time for everything in this hand-off. Report the spend after G4. Stop and ask before exceeding the cap.
- **No tuning towards a result.** Don't change payoffs, temperature or prompts to produce a particular gap. Any change after the pilot is a new condition: log it and tell Abdul.
- **Prompts:** fix template bugs only, and flag each one. Wording changes are Abdul's.
- **No invented numbers.** Every figure comes from logged data.
- **Stop and ask Abdul if:**
  - a gate fails
  - an input in §1 is missing when a GPU run needs it
  - the cost would pass the cap
  - this file leaves a design question open, or contradicts the paper's definitions

## 11. Done when

- [ ] G1–G5 pass, and G6 is done or skipped, each reported to Abdul.
- [ ] Audit tables fitted and scored; `results/audit/README.md` describes p_test.
- [ ] Arms 4, 4b, 5 and 6 run and committed; arm 7 only if approved.
- [ ] Analysis run: paired CRPS (primary and arm 4b), mean errors, Δ_t with noise floor, defects, bound, defect shift, shared-neighbour check and baselines.
- [ ] Figures 1–2 and the validation table written to `results/`.
- [ ] `results/README.md` lists every output.
