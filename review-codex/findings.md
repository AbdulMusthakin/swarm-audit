# Independent scientific and technical audit

Review branch: `codex/independent-audit-20261108`, based on main
`982822f554e547cc5c4468b70d5aaaae88356814`. Fixed additional snapshots:
core `f020fe2a0c93c42177a48ccdf9141ead9c185d1a`, extra-arms
`4a817051abffbdf208eb38a2b09160ab9d49aa0f`. See `independent/initial_refs.txt`,
`independent/inventory.json`, and the final refresh in `reproduce.md`.

**Assessment:** the independently checked headline calculations reproduce. No critical
numerical invalidation was found. The primary endpoint result remains null. The most
important corrections concern simulation uncertainty and experiment classification;
the strongest explanatory result concerns own-message retention and synthetic
averaging in one exploratory, bounded-memory OLMo condition.

All computations were local CPU calculations. No paid compute, model queries, agent
credentials, Modal workspaces, deployed servers, original evidence, or agent branches
were modified. After the user's subsequent authorisation to coordinate corrections,
a concrete note was posted on [PR 9](https://github.com/AbdulMusthakin/swarm-audit/pull/9#issuecomment-6715458905).
No code was pushed or merged. `coordination.md` preserves that note.

## Results independently established

| Quantity | Independent result | Interpretation |
|---|---|---|
| Llama original primary endpoint D | −0.0000864399; archived conditional 95% interval [−0.0002210329, +0.0000353221] | No demonstrated incremental endpoint value from signals |
| Llama pooled 400k simulation reference D | −0.0000912341; independent run-bootstrap interval [−0.000233911, +0.0000479695] | Same conclusion, conditional on fitted table/reference |
| Llama max binned per-round TV | 0.26145 actions, 0.28855 signals; both round 2 | Timing discrepancy despite small endpoint score difference |
| OLMo n=4 last-1 exact P(F=1) | 0.14239254032336024 signals; 0.0000763560316673361 actions | Different late all-high probabilities under the composed laws |
| OLMo n=4 last-1 observed F=1 | 8/50; exact binomial 95% interval [0.0717008, 0.2911263] | Finite live sample, despite exact cached-law calculation |
| OLMo observed paired CRPS D | +0.0252262741; run-bootstrap interval [−0.003426035, +0.053878583] | Positive estimate, interval includes zero |
| Expected CRPS advantage when Y follows the full OLMo law | 0.0205240460902184 | Proper-score calculation under a specified reference law |
| OLMo own-only / neighbours-only P(F=1) | 0.1373394474 / 0.0000694596 | Own-message retention nearly recovers this endpoint law |
| Both observations replaced with `none` | P(F=1)=0.1412513396 | Literal none and averaging messages are different interventions |
| OLMo n=10 full-history observations | y=0 in all 50 runs; all runs at X=0 by round 12 | End state predictable; timing missed by both surrogate chains |

F is the fraction of high bids in rounds 16–20; F=1 requires every agent high in
**each** of those rounds. It differs from instantaneous all-high occupancy and from
first occurrence of three consecutive all-high rounds. Full laws and source hashes
are in `independent/`; `claims.csv` distinguishes calculation from source extraction
and original primary, secondary and exploratory evidence.

## Material findings and minimal fixes

### M1. Monte Carlo error is material for the tiny primary contrast

**Evidence:** main's `swarm_audit/stats.py:bootstrap_diff_of_means` says 20k model runs
make simulation noise negligible. Conditioning on the simulated reference is a
valid conditional analysis, but does not establish negligibility relative to D.
Twenty independent 20k batches give endpoint-D SD **0.0000539702**, compared with
the original whole-run SE **0.0000675245**. Pooling 400k runs per predictor reduces
the estimated MC SE to **0.0000120681**. For one 20k batch, 1.96×SD is about 0.000106.

Run sampling, table estimation and simulation are distinct uncertainty sources.
Two hundred paired table resamples, preserving fitting-history six-vectors and
rederiving both forecasts together, give endpoint SD including MC **0.0000790166**,
approximate table-only SD **0.0000577134**, and exploratory nested run/table/MC
interval **[−0.000397476, +0.000088647]**. Variance subtraction is approximate; MC
variance can change with the table. Two hundred replicates give imprecise tail quantiles.

**Fix:** preserve the original primary result and conditional interval; add separately
labelled MC/table sensitivities and use a higher-precision reference before
interpreting the tiny sign. Replace the unconditional negligible-noise comment by
a requirement to assess reference variability relative to the contrast. The core
follow-up plan already proposes this remedy. The primary endpoint conclusion survives.

Reproduce: `primary_review.py`; output `llama_primary.json`, keys `original_paired`,
`monte_carlo`, `table_bootstrap`.

### M2. Exploratory OLMo output is mechanically marked as primary

**Evidence:** `swarm_audit/analysis.py:analyse_condition` tests channel, n, graph and
memory only for `is_primary_contrast`. At extra snapshot `4a81705`,
`results/extra/analysis/olmo_core_cheap_talk/cheap_talk_n10_cycle_full.json` has this
flag `true` and inherited primary notes. Its provenance identifies arm 91; the
original primary is Llama arm 6. The OLMo trigger was not met and the extension
was chosen after earlier results. `arm9_summary.md` correctly calls it exploratory.

**Fix:** store classification explicitly with the frozen model, design and run set,
and use it in combined tables. At minimum include the original arm identity in
the boolean. Preserve original outputs and correct classification in derived
outputs. Later n=10 full-memory follow-ups also need explicit exploratory labels.
Reproduce: `secondary_checks.py`, key `OLMo_n10_archived_is_primary_contrast`.
Reported to the team on PR 9.

### M3. The theoretical proof and scope must accompany the paper

**Evidence:** `setup-paper/sections/3-results.tex` states the TV bound and exact
transfer without an active proof. The theorem is valid under its finite domain,
product conditional sampling and compatible coarse wiring. An action-only hybrid
cannot omit messages and still invoke the original future kernels without defining
those missing signals. `independent/theory_review.md` supplies a lift using
K(message|action,history), arbitrary choices on zero-mass fibers, and chronological
true-prefix hybrids. It also gives counterexamples to dropping independence or
replacing full-history support with coarse-state coverage.

**Fix:** include that proof or an equivalent coupling proof; retain full-history
support; distinguish the population RHS from its logged estimate. Main's Llama
binary B20=11.9786 and joint B20=43.1398 already clip to 1. The analysis correctly
calls these estimates rather than certificates; preserve that wording.

If setup-paper text is reused, remove `5-hypotheses.tex`'s suggestion that a similar
empirical binned-TV growth ratio establishes tightness of the trajectory bound in n.
A coarsened marginal and an uncapped upper bound cannot establish that claim.
Qualify `2-setup.tex`'s closing statement about exactness only on counts as referring
to actions-only audits: last-1 enumerated signal-aware cheap talk can also be exact.

## Minor findings and fixes

### m1. A specified first-entry diagnostic includes re-entries

At extra `4a81705`, `scripts/extra_followup_exact.py:diagnostics` and its E2
`survival_own_entry` loop use unrestricted outside-all-H occupancy for entry flows.
The frozen plan specifies **first** entry. Independent correction gives:

- P(ever all-H by round 20)=0.15023057291869008;
- expected number of entries=0.15024415031308289;
- re-entry mass=0.00001357739439281;
- TV between normalised entry-profile mixtures=0.00004138525945618;
- largest survival correction=0.00000194175407187.

**Fix:** propagate unnormalised mass that has never hit all-H, record incoming flow,
and kill it before continuing. `secondary_checks.py:entry_distribution(first=True)`
is a concrete correction. Alternatively label the old result an all-entry mixture
and disclose the departure from the frozen plan. No endpoint or main mechanism
changes. Reported to the team on PR 9.

### m2. Status descriptions are stale

Main's `results/README.md` ends by saying no GPU results exist, despite saved results.
`results/validation.md` says G6 and drift have not run despite saved pilot and
per-condition records. **Fix:** update from manifests, cross-checking `inventory.json`.

### m3. Narrow payoff and motivation wording in reused setup text

`2-setup.tex` says no result depends on payoffs. The theorem does not require payoff
optimisation, but measured kernels and numerical results can depend on payoff text.
Use: the theorem does not assume agents optimise payoffs.
`1-question.tex` says the motivating incident was completely unprompted. METR
supports unsanctioned communication and attack participation within an assigned
benchmark context; use that narrower description. Direct verification of the cited
Greenblatt X quotation failed with 403; verify it or omit the exact quote.

## Important qualifications and checks that passed

**Chronology:** original Llama G5 failure p=0.00759962; subsequent fresh confirmation
p=0.74361282; pooled p=0.08399580. All reproduce exactly under the archived seed and
20k null simulations. The investigation found no probability, wiring or draw error;
this supports, but does not prove, the chance explanation. The authorised exception
is a separate record. Its commit and the confirmation precede core provenance.
Do not compress the sequence to G5 passed or call the pooled set an independent
replication. Approximate MC SEs of the p-values are 0.00061, 0.00309, 0.00196.
Uniformity tests alone do not prove independence.

**Units/multiplicity:** paired resampling of whole runs is appropriate. Agents and
rounds are dependent; thousands of decisions are not thousands of independent
endpoint samples. Separate simulation pools for the TV null are appropriate.
The maximum-over-rounds null handles the rounds within one predictor, not all
models, arms and later contrasts. E1/E2/E3 remain exploratory after prospectively
freezing their follow-up specifications.

**Endpoint/path:** Llama's peak TVs exceed archived simultaneous thresholds
0.23758/0.23536 and independent fresh thresholds about 0.23640/0.23395. The early
score comparison is exploratory; actions-only is slightly better (D about −0.000675
in the 400k reference), which is not a general model-superiority claim. OLMo's 50
constant late outcomes make run-bootstrap intervals collapse. A zero event count
is not zero probability: 0/50 gives a two-sided 95% binomial upper limit 0.07112.
Independent 100k surrogate simulations still predict round-3 X means 0.340/0.352
against observed 0.136. Late-outcome calibration does not imply path calibration.

**Mechanism:** E1/E2 laws reproduce within 6e-15. In full-model all-high occupancy
over rounds 2–20, own-message shares are high 0.438794, none 0.561105, low 0.00010146;
the synthetic average gives low weight 1/3. With own action H and two high-bidding
neighbours whose messages are both high, P(H) is 0.99956944/0.70578504/0.99919591
for own high/low/none. The action-only value is 0.90499137; its next-round all-high
survival is 0.67077637. This supports sensitivity to own-message state under these
weights. It does not establish necessary neighbour communication, intentions,
commitment, collusion, or a safety property. Literal none preserves a near-full
outcome law. Uniform-replacement identities assume independent observation noise
per agent and step; a shared replacement broadcast is a different process.

**Exactness:** independently reversed labels/order reproduce the outcome laws.
Joint vectors preserve action-message dependence: START's joint-versus-product
TV is 0.0386283. Replacing them with products of marginals changes the process.
Exact propagation is finite arithmetic on a recorded stationary last-1 kernel,
conditional on prompt/serving identity and independent sampling. It is not exact
unrestricted LLM generation. The experiment renormalises selected-label
probabilities over H/L and high/low/none; this review made no fresh model queries.

**Direct integrity checks:** all original Llama primary predictor CDFs and paired
per-run scores match exactly. Its 10k decisions have zero rebuilt-record, state,
action/signal draw or seed-stream errors. Calibration checks cover 12k decisions
(Llama original+confirmation and OLMo). Across four conditions totalling 32k
provenance-checked decisions, template hashes, run/table identities, raw-probability
reconstruction, seed replay and observed enumerated prompt hashes all pass.
OLMo calibration visits 55/109 states; Llama calibration visits 109/109.
Saved table rows normalise within 5e-16; logprob-to-probability reconstruction errors
are at most 3e-16. Serving/drift records were inspected, not independently rerun.

**Limits:** counts n=10/n=20 and n=100 are inventoried/read but not independently
rerun end to end. One d=4 random graph at n=100 cannot isolate size from topology
and degree. Initially partial n=20 cheap talk and absent follow-up live cells are
not complete results. Final-refresh results are separately documented. No
comprehensive novelty claim, independent live replication, or new safety validation
is asserted.
