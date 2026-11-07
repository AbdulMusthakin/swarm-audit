# Hand-off: independent check of the calibration gate and the CRPS comparison

For a teammate working alone, with or without their own AI assistant. No Modal credits or GPU are needed: everything runs on a laptop from files already in the repository.

**Why this matters.** The cheap-talk calibration (arm 4b) failed its per-run check, and the build session attributes the failure to chance. Before the team relies on that, someone who did not write the pipeline should recompute the result from the definitions alone. The same independent code then checks the paper's main comparison when the core runs arrive.

## 0. Rules for independence

- **Write your own code** in Python with numpy (scipy is optional). Put it in a new folder, `independent/`, on your own branch, `independent-check`.
- **Don't open** anything under `swarm_audit/` or `scripts/`, or the files `results/gates/*`, `results/validation.md` and `scripts/diagnose_g5.py`, until you have written your numbers down (step 1.6). The definitions you need are all in this file.
- **Use only:** the logged runs, the audit tables, and this file.

Check out the core branch to get the data: `git fetch && git checkout origin/claude/funny-turing-qv2y66`, then create your branch from it.

## 1. The data

### The 50 logged runs of arm 4b

`results/runs/calibration_cheap_talk/cheap_talk_n4_cycle_last1/run_000.jsonl.gz` to `run_049.jsonl.gz`: gzipped JSON lines, one line per decision. Each run has 4 agents and 20 rounds, so 80 lines. Fields you need:

| Field | Meaning |
|---|---|
| `run`, `round` (1–20), `agent` (0–3) | Which decision |
| `action` | `"H"` or `"L"`, the bid actually drawn |
| `signal` | `"high"`, `"low"` or `"none"`, the message actually drawn |
| `state_sig` | The agent's signal-aware state when deciding (format below) |
| `p_H`, `sig_after_H`, `sig_after_L` | The model's probabilities at that decision: P(H), and P(high, low, none) after each bid |
| `u_action`, `u_signal` | The uniform draws used: H if `u_action < p_H`; the signal by inverse CDF in the order high, low, none |

### The graph

A cycle of 4 agents: agent i's neighbours are i − 1 and i + 1 (mod 4). Each agent sees **only the last round** (that is what makes this a calibration arm).

### States

`state_sig` is `"START"` in round 1. Otherwise it is `"<a>-<k>-<h>-<l>-<z>-<m>"`, where:
- `a`: the agent's own bid last round (H or L)
- `k`: how many of its 2 neighbours bid H last round
- `h`, `l`, `z`: how many of its 2 neighbours said high, low and none last round (h + l + z = 2)
- `m`: the agent's own message last round

There are 2 × 3 × 6 × 3 = 108 states, plus START.

### The signal-aware table

`results/audit/llama-3.1-8b/cheap_talk_d2_enumerated_signal.json`. Under `entries`, each state key maps to an object whose `kappa` is a 2 × 3 array: `kappa[a][m]` = P(bid a, then message m | state), with rows in the order H, L and columns in the order high, low, none. Each `kappa` sums to 1. That joint distribution is the whole kernel for the signal-aware chain.

### The actions-only table (for part 2)

`results/audit/llama-3.1-8b/cheap_talk_d2_enumerated_actions.json`. Under `entries`, the keys are `START`, `H-0`, `H-1`, `H-2`, `L-0`, `L-1` and `L-2` (own last bid, then the number of neighbours bidding H). Each `kappa` is a single number, P(H | state). This chain ignores messages.

## Part 1: recompute the calibration result

### 1.1 Check the logs against the table, per decision

For every one of the 4,000 decisions, compare the logged probabilities with the table entry for its `state_sig`:
- `p_H` must equal `kappa[0][0] + kappa[0][1] + kappa[0][2]`.
- `sig_after_H` must equal `kappa[0] / p_H`, and `sig_after_L` must equal `kappa[1] / (1 − p_H)`.

Report the largest absolute difference. It should be 0 up to floating-point rounding, about 1e-12.

### 1.2 Check the wiring

Recompute each agent's `state_sig` for rounds 2–20 from the **logged actions and signals** of the previous round: its own bid and message, and its two neighbours' bids and messages. It must equal the logged `state_sig`. Report the number of mismatches. Also check that every action and signal follows from its logged draw (`u_action`, `u_signal`) by the rule in the table above.

### 1.3 The statistic y

For each run, y = the number of H bids in rounds 16–20 over all 4 agents, an integer from 0 to 20. Report the 50 values, their mean and their histogram.

### 1.4 The exact law of y under the signal-aware chain

The chain's state is the profile of all 4 agents' last (bid, message) pairs: 6⁴ = 1,296 profiles.
- **Round 1:** every agent draws (a, m) independently from `kappa` of START.
- **Rounds 2–20:** given the last profile, each agent computes its state from its own and its neighbours' last pairs, and draws its new (a, m) independently from that state's `kappa`. The new profile is the product of the four agents' draws.

To get the law of y, carry the count of H bids so far in rounds 16–20 alongside the profile: a distribution over (profile, count), with count from 0 to 20. Push it forward from round 1 to round 20, adding each round's H count from round 16 on. At the end, sum over profiles.

Checks: the law sums to 1. Its mean equals the sum over rounds 16–20 of the expected number of H bids. A plain simulation of 200,000 runs of the chain gives the same law within sampling error.

Report the exact mean of y, and the exact CDF F(k) for k = 0 to 20.

### 1.5 The per-run test

- **Gap:** let Fₒ be the empirical CDF of the 50 observed y values. The gap is the maximum over k = 0 to 20 of |Fₒ(k) − F(k)|.
- **Null distribution:** repeat 20,000 times: draw 50 independent values of y from the exact law, form their empirical CDF, and compute the same gap against F. (You can sample y directly from its exact law; you don't need to simulate the chain.)
- **p-value:** p = (1 + number of null gaps ≥ the observed gap) / (1 + 20,000). Use a tolerance of 1e-12 in the comparison, because gaps take discrete values.
- **Gate rule:** pass if p > 0.01.

Report the gap, the 99th percentile of the null gaps, and p. Use a fixed random seed and report it.

### 1.6 Write your numbers down, then compare

Commit your results to `independent/part1_results.md` **before** opening `results/gates/G5_cheap_talk.json`. Then compare. The build session reported a gap of 0.203, a p-value of 0.0076 and an exact mean of y of 5.47, against an observed mean of 5.82.

- **If you agree** (same gap; a p-value within Monte Carlo error, about ±0.002; same exact mean): say so.
- **If you disagree:** find the first step where you differ (1.1, 1.2, 1.4 or 1.5) and report it to Abdul at once. A disagreement in 1.4 would mean the build's exact law, and so the gate, is wrong.

### 1.7 One extra diagnostic

Compute the per-round mean number of H bids in the 50 runs against the exact chain's expectation, for rounds 1 to 20. Also do the same for the messages. A single anomalous round or message type would point at something the per-run test can't locate. Report a small table.

## Part 2: the CRPS comparison

### 2.1 The score

For a predictor with CDF G of y (y counts H bids in rounds 16–20, out of 5n decisions), the score of a swarm run with value yᵣ is

    CRPS_r = (1 / 5n) · Σ_{k=0}^{5n} (G(k) − 1[yᵣ ≤ k])²

Lower is better. The paired difference is Dᵣ = CRPS_r(actions-only) − CRPS_r(signal-aware); positive values favour keeping the signals. Report the mean of Dᵣ over runs, with a 95% percentile interval from 2,000 bootstrap resamples of whole runs (resample run indices with replacement and recompute the mean). Treat the predictors' G as fixed.

### 2.2 Do it now on arm 4b (n = 4, so 5n = 20)

- **Signal-aware G:** the exact law from 1.4.
- **Actions-only G:** the exact law of the 16-profile chain (each agent's last bid). Use the actions-only table. Round 1: P(H) from START. Later rounds: P(H | own last bid, number of neighbours bidding H). Carry the count as in 1.4.

Report the mean of D with its interval, and each predictor's mean CRPS. This is the paper's key secondary result. Write it to `independent/part2_arm4b.md`, then compare with the build's analysis output once you have committed yours.

Unit checks: CRPS is 0 when G puts all its mass on yᵣ. Check one hand-computed example with three values.

### 2.3 Ready for the core runs (n = 10, cycle, full history)

When the core cheap-talk runs land (`results/runs/core_cheap_talk/cheap_talk_n10_cycle_*/`, same log format), run the same code with three changes:
- **Tables:** the fitted tables, `cheap_talk_d2_signal.json` (signal-aware) and the derived actions-only table, in `results/audit/llama-3.1-8b/`. Check the exact file names when they appear.
- **G by simulation:** at n = 10 there are 6¹⁰ (about 60 million) profiles, too many to enumerate, so get each predictor's G by simulating 20,000 runs of its chain on the 10-cycle. The chain uses only the last round, even though the swarm's agents saw their full history; that is the point of the comparison.
- **5n = 50.**

Report the mean of D with its interval, and each predictor's mean CRPS. Then compare with the build's numbers. Agreement within simulation error, about 0.001, is a pass.

## 3. What to send Abdul

- Tonight: Part 1's numbers and the agree/disagree verdict, plus 1.7's table and Part 2.2's result. Two or three lines in the team chat are enough, with a link to your branch.
- When the core lands: Part 2.3's number and the verdict.
- Push your branch, `independent-check`, so the code is in the repo. Don't merge it into the core branch.

**Time:** Part 1 is about 1–2 hours, and Part 2.2 another hour. Part 2.3 is about 30 minutes once the logs exist.
