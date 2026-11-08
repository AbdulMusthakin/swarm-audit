# Hand-off: final report pass to maximise the judging scores

For a Claude Code session revising `report/` on the branch `claude/gracious-bardeen-42sf3t`. Read this file in full before starting.

**When:** written Sunday 8 November 2026, 20:00 UK time, against commit `a35d7ab`.
**Deadline:** 23:59 AoE on 8 November, which is **11:59 UK time on Monday 9 November**. Resubmitting under the same title replaces the earlier submission, so a safe version should be submitted first and improved after.
**Owner:** Abdul Musthakin. He decides the wording; this file proposes it.

## 0. Why, and the targets

The judges score three dimensions from 1 to 5 (rubric in the sprint guidelines). An outside review of `a35d7ab` gives:

| Dimension | Now | Target | What moves it |
|---|---|---|---|
| Impact & Innovation | 3 | 4 | §2 (70B), §3 (compounding mechanism), §4 (why the averaged audit is not a strawman), §6 (related work) |
| Execution Quality | 4 | 5 | §2 (the prespecified contrast, applied unchanged to a model that reacts to messages, is clearly positive) and §7 (number hygiene) |
| Presentation & Clarity | 3–3.5 | 4 | §5 (abstract, contributions, settings table, empty half page) |

The revision at `a35d7ab` already fixed: the pipeline schematic, plain definitions of the roundwise check and the events, the "(optional)" headings, and the diagnostic-sensitivity comparison (Appendix F).

Work in the order of the sections below. §2–§5 are the score-moving changes; §6–§8 are smaller.

## 1. Guardrails

- **Labels.** Only the 8B Llama endpoint comparison was prespecified. Everything added here is exploratory and must say so. The 70B runs were decided on 8 November, after the 8B results were seen.
- **Every new number comes from a committed file or a committed script.** Do not paste the numbers in this file into the report without first reproducing them with the scripts in §2.4 and §3.2.
- **Do not copy prose from `interest/model-70b/README.md` or `comparison.md` about the direction of the audits' bias.** It is inverted (see §2.3). The numbers in those files are right; some sentences are not.
- **Limits:** main body at most 8 pages before references and appendices (it is about 5.5 now); abstract 150–250 words (it is 211 now, so any addition must be paid for by a cut).
- Rebuild with `cd report && latexmk -pdf main.tex`, then open the PDF and check every page you changed.

## 2. Bring in the 70B runs (Impact and Execution)

### 2.1 Merge

The runs are on `origin/claude/elegant-cori-g9tqdx` (Llama 3.1 70B Instruct, revision `1605565b`, one B200, vLLM 0.32.0, batch-invariant serving). Its merge base with this branch is `b524342`, and a trial merge into `a35d7ab` had **no conflicts**:

```
git merge origin/claude/elegant-cori-g9tqdx
```

Then update `interest/model-70b/README.md`: replace the opening "Not part of the report or the submission" paragraph with a sentence saying that §4.5 and Appendix B of the report use these results as an exploratory, post hoc rerun, and link it from the main README.

### 2.2 What the 70B shows

`configs/model-70b.yaml` is `configs/core.yaml` with only the model, hardware, output folder and arm names changed: same prompts, payoffs, T, test distribution, audit sizes, seeds and analysis settings. The runs reuse the core's arm ids, so run *i* uses the same random draws as the 8B's run *i*. Sources are `interest/model-70b/analysis/*/*.json` and `interest/model-70b/comparison.md`.

1. **The 70B reacts to messages.** On the cached last-round tables, using the same construction as the report's 3.8-point screen (two neighbours, one bid high, four neighbour-message variants, own message "none", spread of P(high) averaged over own last bid), the 70B's spread is **9.6 points** against **3.8** for the 8B. The same computation on the 8B table gives exactly the report's 3.8, which checks the method. See §2.4.
2. **The prespecified contrast, applied unchanged, is clearly positive.** On the cheap-talk ring with full history:
   - 10 agents: D = **+0.0061 [+0.0045, +0.0077]**
   - 20 agents: D = **+0.0063 [+0.0044, +0.0079]**
   Reading messages improves the audit once the model responds to them. (The analysis JSON marks these `is_primary_contrast: false` because `swarm_audit/analysis.is_primary_contrast` reserves that label for the core 8B arm; the analysis itself is the same.)
3. **But both audits forecast worse than a memoryless audit built from the same tests.** CRPS of F, 10 agents: order-zero 0.049, signal-aware 0.063, actions-only 0.069. Paired differences, whole-run bootstrap (§2.4):
   - signal-aware minus order-zero: +0.0136 [+0.0083, +0.0185] at 10 agents, +0.0132 [+0.0082, +0.0182] at 20
   - actions-only minus order-zero: +0.0197 [+0.0128, +0.0262] at 10 agents, +0.0195 [+0.0129, +0.0259] at 20
   For the 8B the three were within about 0.001 of each other (0.041, 0.040, 0.040; §4.1).
4. **The audits overpredict late high bidding.** In rounds 16–20, live games averaged 23% high bids (10 agents) and 25% (20 agents), against forecasts of about 30–31%. `F_error` (live minus forecast) is −0.078 (actions-only) and −0.068 (signal-aware) at 10 agents, and −0.063 and −0.053 at 20.
5. **The roundwise check flags both audits.**
   - 10 agents: largest gap at round 3, TV 0.400 (actions-only) and 0.382 (signal-aware), against cutoffs of 0.229 and 0.227.
   - 20 agents: largest gap at round 4, TV 0.504 and 0.483, against cutoffs of 0.206 and 0.204.
   The live share jumps to 34% in round 2 and falls to 15% in round 3; the audits plateau near 30% from round 3. This is the same kind of timing miss as the ten-agent OLMo setting, in a second model family.
6. **The four-agent last-round calibration passed** on both channels (cheap talk: per decision exact, per-run p = 0.024 > 0.01). In last-round settings the audit is exact by construction (Proposition 2). Passing calibration therefore validates the pipeline, not a full-history audit. This is a clean illustration of Proposition 2: exact when the audit remembers what the agents remember, wrong when agents remember more.
7. **The contraction route closes.** The 70B audit chain barely contracts (δ 0.90–1.00, against 0.10–0.87 for the 8B), and every full-history TV bound is vacuous from about round 4 (`interest/model-70b/README.md`, "Contraction bounds"). One sentence in Appendix A is enough.

### 2.3 Sign error to fix in the 70B files

`swarm_audit/stats.bootstrap_diff_of_means` returns E_swarm F − E_model F, so **negative `F_error` means the audit predicts too many high bids**. Fix these sentences:

- **`interest/model-70b/comparison.md` header**, generated by `interest/model-70b/compare.py` (lines 40–41): "F_error: mean forecast minus swarm (negative: the audit predicts too few high bids)" → "F_error: swarm minus mean forecast (negative: the audit predicts too many high bids)". Fix it in `compare.py`, then regenerate `comparison.md`.
- **`interest/model-70b/README.md`, cheap talk:** "both tables under-predict high bids late in the game: the forecast mean minus the swarm is −0.05 to −0.08" → "both tables over-predict high bids late in the game: the swarm minus the forecast mean is −0.05 to −0.08".
- **`interest/model-70b/README.md`, K_10 counts:** "over-predicts high bids by about 9 points (F_error +0.093 …)" → "under-predicts high bids by about 9 points". The live late share there is 43%, against a forecast near 34%.

### 2.4 Scripts for the new numbers

Commit one CPU-only script, e.g. `scripts/report_70b.py`, that writes `results/followup/report_70b.json` with these numbers. It needs three parts:

1. **Message responsiveness** for both models. Use the d = 2 enumerated signal tables (`results/audit/llama-3.1-8b/cheap_talk_d2_enumerated_signal.json` and `interest/model-70b/audit/llama-3.1-70b/cheap_talk_d2_enumerated_signal.json`). Keys are `"{own bid}-{neighbours high}-{n 'high'}-{n 'low'}-{n none}-{own message}"`, and P(high) is `sum(entry["kappa"][0])`. For own bid H and L, own message `none`, and neighbours-high = 1, take the spread max − min over the variants `2-0-0`, `0-2-0`, `0-0-2` and `1-1-0`. Average the two spreads. This gives 0.096 (70B) and 0.038 (8B).
2. **Paired CRPS differences** against order-zero. Take them from `scores.crps_per_run` in the two `m70_core_cheap_talk*` analysis JSONs, with 2,000 whole-run bootstrap resamples (the review used seed 20261108).
3. **Cross-check D** against `scores.paired_D` in the same files.

Add the claims to `results/followup/core/claims.csv` if that is still the claim map.

### 2.5 Where it goes in the report

**New subsection** after §4.4: "4.5 A larger model: messages matter, but the audit forecasts worse (exploratory)". Renumber "How robust are these results?" to 4.6. Draft wording to adapt:

> We reran the Llama settings with Llama 3.1 70B, changing only the model and serving hardware; prompts, seeds, test design and analysis were unchanged. The decision to run it was made after the 8B results, so this is exploratory. The 70B reacts to messages more than twice as strongly (9.6 against 3.8 points in the same screen), and the prespecified contrast, applied unchanged, favours the signal-aware audit: D = +0.0061 [+0.0045, +0.0077] with 10 agents and +0.0063 [+0.0044, +0.0079] with 20. Yet both audits forecast the endpoint worse than a memoryless audit built from the same tests (CRPS 0.063 and 0.069 against 0.049; paired difference for the signal-aware audit +0.0136 [+0.0083, +0.0185]). They overpredict late high bidding (about 30% against 23% live), and the roundwise check flags both, with the largest gap at round 3. The four-agent last-round calibration, where the audit is exact by construction, passed: a model whose messages carry information made the full-history audit worse, not better.

**A compact table** (8B and 70B side by side, cheap talk, ring, full history, 10 agents) with these rows:
- message responsiveness
- D
- CRPS for order-zero, actions-only and signal-aware
- live and forecast late high-bid share
- largest roundwise TV against its cutoff

The 8B values are already in Appendix B.

**Other edits:**
- **§4.1:** add one closing sentence pointing to §4.5.
- **Abstract:** one sentence, e.g. "In an exploratory 70B rerun, messages improved the audit, yet both audits forecast worse than a memoryless baseline." Pay for it by shortening the round-3 sentence.
- **Contributions:** add a bullet: the failure is not specific to small models.
- **§5 "What the results show":** add the 70B.
- **Limitations "Scope":** two small models, plus an exploratory 70B rerun of the Llama ring settings.
- **Conclusion:** replace "should be tested on larger models" with what the 70B showed, plus what remains untested (larger and frontier models, richer tasks).
- **Appendix B:** a 70B findings bullet with status "exploratory, post hoc; analysis unchanged".
- **Appendix C:** the 70B serving line (model, revision, B200, G3 floor 0, G5 pass on both channels, drift 0) and the settings.
- **Appendix D:** reproduction. The CPU-only re-analysis is the `step analysis` blocks of `interest/model-70b/run_all.sh` and `run_stage2.sh`, then `python interest/model-70b/compare.py` and the new script.
- **Appendix A:** one sentence on δ and vacuous bounds for the 70B.

## 3. Name the general mechanism behind the four-agent miss (Impact)

### 3.1 The mechanism

A small per-step error is nearly invisible in round-by-round shares, but it compounds multiplicatively over persistence. Persistent coordinated states are what the safety-relevant swarm failures look like, including the incident in the introduction.

The cached OLMo tables reproduce both exact event probabilities from two numbers:

- **The start.** Both tables give P(high) = 0.622 in round 1. So P(all four high in round 1) = 0.622⁴ = 0.150.
- **Actions-only retention.** In the all-high state an agent stays high with probability 0.905 (`results/extra/audit/olmo-3-7b/cheap_talk_d2_enumerated_actions.json`, entry `H-2`). That is 0.905⁴ = 0.671 per group-round. Over the 19 rounds after round 1 this gives 5.1 × 10⁻⁴, and 0.150 × 5.1 × 10⁻⁴ = **0.0076%**, the exact value.
- **Signal-aware retention.** The table keeps an all-high agent high with probability about 0.999 whenever its own last message is "high" or "none" (entries `H-2-*-*-*-high` and `H-2-*-*-*-none` in `cheap_talk_d2_enumerated_signal.json`). It almost never sends "low" after bidding high. This gives 0.150 × 0.9992⁷⁶ ≈ **14.1%**, against the exact 14.24%; the remainder is re-entry.
- **The lesson.** A gap of about 0.094 per agent-round becomes a factor of about 1,900 in the event probability over 76 agent-rounds. This also explains why the shared-cutoff roundwise check, which sees one round at a time, missed it.

### 3.2 What to do

- Compute these numbers in a small committed script (e.g. `scripts/report_persistence.py`), so they are reproducible.
- Add two or three sentences to §4.3 "Why the audit missed it".
- Add one clause to the abstract, e.g. "small per-step errors compounded over 76 agent-rounds".
- Tie it back to the introduction's incident in §5: dangerous swarm behaviour is coordinated and sustained, which is exactly where compounding bites.

## 4. Pre-empt the "strawman" objection (Impact)

A judge may ask why anyone would weight unseen message states one-third each. Add a short paragraph to §4.3 or §5 making three points:

- **Compression is forced at scale.** The signal-aware state has 108 situations with two neighbours, and it grows quickly with degree. The 10-agent complete-graph audits (d = 9) in this repository already use directly fitted actions-only tables.
- **Any compression needs weights.** The default weights are the test design's.
- **The fix is available from isolated tests.** Keeping the agent's own message, which isolated tests can supply, restores the forecast (13.73%). So the lesson is specific: when an audit summarises away part of the state, weight it by the agents' own on-path behaviour, not by the test design.

## 5. Presentation

- **Abstract** (keep 150–250 words): lead with the lesson; keep at most three or four numbers; add the 70B sentence and the compounding clause. Also add the positive diagnostic result from Appendix F: in the pre-registered simulation, a roundwise check calibrated round by round, or a direct persistence test, flagged the alternative in over 99.7% of batches, against 33% for the shared cutoff.
- **Contributions:** three bullets. The empirical findings (both OLMo misses; the 70B result), the mechanism (compounding over persistence), and the diagnostic (per-round calibration and a named event, validated by the pre-registered simulation).
- **Empty half page.** Remove the `\clearpage` before "4. Results" (`report/main.tex`, about line 250). Page 3 is half empty. Use the space for a **settings table**: model, agents, memory, channel, status (prespecified, secondary or exploratory), games. Include the 70B rows.
- **§4.6 (robustness):** shrink to two or three sentences and move the rest to Appendix B.
- **Figure 1:** leave as it is.

## 6. Related work (Impact)

Add one short paragraph on collusion between LLM agents, after verifying the details on the linked pages:

- Fish, Gonczarowski and Shorrer, *Algorithmic Collusion by Large Language Models*, arXiv:2404.00806 (https://arxiv.org/abs/2404.00806).
- Motwani et al., *Secret Collusion among AI Agents: Multi-Agent Deception via Steganography*, NeurIPS 2024, arXiv:2402.07510 (https://arxiv.org/abs/2402.07510).

Take author lists and venues from those pages, not from memory. Also check the sprint's Resources list and cite anything there that the report builds on. Made-up or wrong references can disqualify the submission.

## 7. Number hygiene (Execution)

- After all edits, re-run the separately written re-derivation script on the final numbers and fix any drift.
- Check every number in the abstract against its section, and every section number against Appendix B.
- After the final commit, re-pin Appendix D and the README to that commit (as `a35d7ab` did for `0ee4fdb`), and check that every link in the PDF opens.

## 8. For Abdul, not the session

- Confirm "Mercial" is the intended affiliation for all five authors.
- Submit a safe version before the deadline, then resubmit under the same title if later versions improve on it.
- Optional: a 3–5 minute video demo, which the guidelines allow.
