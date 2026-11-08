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

**Current disposition after the scheduled continuation:** the earlier MC docstring,
primary classification, first-entry, status-text and causal-wording corrections
are verified. Completed new live cells and Llama secondary comparisons reproduce.
See the continuation section for a new minor resume-provenance gap and a
multiplicity sensitivity. The theorem proof and setup wording remain work for the
report authors; no additional compute is required for the corrections.

All computations were local CPU calculations. No paid compute, model queries, agent
credentials, Modal workspaces, deployed servers, original evidence, or agent branches
were modified. After the user's subsequent authorisation to coordinate corrections,
a concrete note was posted on [PR 9](https://github.com/AbdulMusthakin/swarm-audit/pull/9#issuecomment-6715458905).
The user then explicitly asked that the commits be visible to the repository; the isolated review branch was pushed. No merge or agent-branch edit occurred. `coordination.md` preserves the initial note.

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

## Final one-time refresh: new committed results

The final fetch examined core `3912f32915bea8f3850c30b779e3be50522f5108` and extra
`76a647ec5f62b361f4048dbca05d30956e8e42de`. It did not inspect live jobs or uncommitted
output. Detailed new checks are in `independent/new_results_review.json`.

**C2 clock audit:** independently fitting length blocks from stored fitting histories
and simulating 200k paths per forecast reproduces the consequential result. Signal-
aware improvement (pooled minus clock) is **+0.005976** early, with conditional
run interval **[+0.003327,+0.008544]**, and **−0.003581** at the endpoint, interval
**[−0.006860,−0.000300]**. Middle/late directions also reproduce; the middle-block
point estimate differs by 2.3 estimated combined MC SE, so agreement is distributional,
not bitwise. Per-state block fitting counts range 9–27, with no empty cells. Signed
teacher-forced error changes exactly from **−0.057684 to −0.019510** early, and
**−0.019698 to −0.039047** late. The core's new uncertainty analysis agrees with this
review's earlier conclusion that MC/table variation matters but does not rescue the
primary endpoint contrast. Its full 200-resample C1/C2 arrays were not independently
recomputed here; the review's original 200-resample primary sensitivity is separate.

### M4. Qualify the newly committed causal explanation

Core `results/followup/core/summary.md` appropriately calls history-content shift an
interpretation earlier, but its Limitations then says “The late failure is a content
effect” and that more histories of the same p_test would not fix it. The clocks change
history-length conditioning; there is no controlled content intervention establishing
that remaining cause. Finite block estimates also contain uncertainty.

**Fix:** say the residual is **consistent with history-content shift**, and separate
reducing fitting variance from fixing population misspecification. A controlled,
length-matched content intervention or fresh predictive validation would strengthen
the claim. The early operational improvement is supported; causal attribution of the
remaining error is not. This qualification affects the explanation, not the reproduced
score improvement.

**New E3 n10 last1 live cell:** 50 complete runs at extra `76a647e`; every run has
F=0. All 10k logged records, graph states, draws, seed streams, probabilities, prompt
hashes and provenance match the existing enumerated kernel. This is fresh live sampling
and a serving/wiring validation, not fresh kernel information. The high-event forecast
was about0.00814; P(no F=1 in50) is **0.66454**, so the zero count is unsurprising.
Its two-sided95% upper limit is0.07112. Do not infer memory has no effect or reject the
rare-event forecast. The n4 full-memory live cell and E2 live cells are absent in this
committed snapshot; the size-by-memory design is not complete.

The initial E1/E2 diagnostic and primary-flag corrections are not resolved in either
final snapshot reviewed here. No additional branch poll was made after this refresh.

## Continuation at 05:30 London: fixed new snapshots

New sources: core `f8ad9bbb88ee538d1c9563c51e0a1d97e4401711`, extra
`9aa31705b9ae5ee45f27a87feeb1a2e2415f318f`. Research refs were fetched once in this
continuation. PR9 had no new comment after the earlier review note; the source
commits contain explicit responses. Earlier findings above describe their original
snapshots; the current disposition is recorded here rather than erasing that history.

**New independently verified result:** all four E2 live intervention cells pass the
specified tests. Event counts are 9,4,7,8 of50 and exact-law Monte Carlo p-values
reproduce as 0.5437228,0.2295885,1,0.8440578 (per-cell alpha0.0025). All16k logged
actual records, transformed observed records, next states, probabilities, prompt
hashes, action/signal draws and seed streams agree. These are live implementation
validations of the cached kernel, not independent new kernel information or proof
that interventions are equivalent.

The newly completed OLMo n4 full-memory cell has F1=5/50 and F0=45/50. Its4000
records, graph states, draws, seed streams and provenance pass. The full-history
kernel was not re-queried; its fitted table remains a surrogate, not an exact live
law. Combined E3 counts are n4last1=8,n4full=5,n10last1=0,n10full=0 of50 each. All
four reported Newcombe contrast intervals reproduce.

**Material report qualification:** the full-memory size contrast is -0.10, nominal
95% interval[-0.21360,-0.00898]. Correcting for even the two size comparisons gives
[-0.23425,+0.00992]; correcting all four E3 contrasts gives[-0.25370,+0.02855].
The specified nominal analysis is valid as reported, but wording that size matters
under both memories needs explicit exploratory/nominal qualification. The last1
size contrast remains negative under this sensitivity. Cells were not jointly
randomised across sessions, and the group-wide extreme event has START probability
p(H|START)^n; this does not establish a general dynamical size effect or phase transition.

**Corrections confirmed:** first-entry survival now agrees with this review's prior
independent first-hit result within9e-16; the derived classification correctly marks
OLMo arm91 exploratory and preserves the old analysis. The core also corrected the
MC docstring, stale status sentences and causal content-shift wording. Focused
regression checks and additional Llama comparisons are recorded in the next checkpoint.

### New minor finding m4: observation intervention is not bound by the resume guard

At extra `9aa3170`, `Condition` and immutable `run_payload` omit the `observe`
argument. The intervention name appears in session metadata, but
`run_condition` checks provenance before appending without comparing that name.
A deterministic reproduction using only `ScriptedBackend('hidden_signal')` in a
temporary directory accepts run0 under `unmodified`, then run1 under `both_null`
with the **same fingerprint** and no rejection. Both session names are recorded.
See `independent/continuation_provenance_probe.py` and
`independent/continuation_20261108/provenance_guard.json`.

**Impact:** no real mixed treatment or bad probability was found in the completed
E2 cells; their actual/observed logs were independently checked. This is a
prospective configuration-identity gap, not an invalidation of those results.

**Minimal fix:** reject changed observer names against recorded session metadata
before appending, and include a stable, versioned observation specification in
future run provenance. Preserve the immutable original evidence; use a derived
validation record for legacy cells rather than rewriting their provenance. The
CPU reproduction should reject after such a fix.

### Completed Llama n20 and complete-graph comparisons

All50 n20 cheap-talk runs are committed at core `f8ad9bb`. Their20000 decisions have
zero cardinality, graph-record, state, sampling, seed-stream, identity or template
errors. The original predictor CDFs and per-run action/signal score contrast
reproduce exactly: D=-0.000041032175 at the original20k reference.

Independent500k reference paths per predictor give D=**-0.00026512**, conditional
run interval **[-0.00038661,-0.00013913]**, estimated MCSE **0.00003375**. The source's
separate2m reference gives -0.00031084; the point difference is compatible with MC
variation. This conditional n20 result favours actions-only, but source table and
nested sensitivities span zero (nested approximately[-0.00081,+0.00008]). Those
n20 table resamples were inspected, not independently re-run here. The original
n10 primary remains unchanged, and n20 shares its fitted audit table.

The n20 clock result also independently reproduces: early improvement
**+0.008994 [0.006790,0.011223]**, endpoint deterioration
**-0.008331 [-0.010809,-0.005668]**. References use500k pooled and200k clock paths;
MC error is separately reported. This extends the exploratory timing pattern,
without identifying a causal content mechanism.

Both complete-graph cells have50 runs. Each cell's10000 decisions pass independent
wiring, state, seed, probability-reconstruction and provenance checks. Original
action-audit predictor CDFs and per-run scores reproduce exactly; analytic order0
scores agree within1e-16. Paired endpoint improvement, order0 minus actions-only:

- cheap talk: **+0.0167668 [0.0120673,0.0211861]**;
- counts: **-0.0006205 [-0.0024790,0.0012258]**.

An independent100k reference preserves both conclusions. These are conditional
within-topology comparisons, not an isolated causal topology/degree effect, a
signal-aware-versus-actions-only comparison, or a generalisation guarantee.

No critical numerical invalidation was found in this continuation. The original
G5 failure, exception, confirmation and all exploratory labels remain intact.
