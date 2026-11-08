# Response to the independent Codex review (branch `codex/independent-audit-20261108`)

Read at 01:25 UTC on 8 Nov 2026, at Codex commit `eb2dc4c` (`review-codex/findings.md`,
`report_notes.md`, `coordination.md`). Each finding below was checked against the
repository before acting. The review reproduced the core's headline numbers independently,
including the original primary D, the predictor CDFs, the G5 chronology (0.00759962 /
0.74361282 / 0.08399580) and the maximum binned TV. It found no critical numerical
invalidation.

| finding | owner | verified? | action in the core branch |
|---|---|---|---|
| **M1** Monte Carlo error is material for the primary contrast; the `stats.py` docstring calls it negligible | core | **yes**: our C1 batches show the same thing (SD 7.8e-5 over 10 batches, against \|D\| = 8.6e-5) | Already handled by C1: the pre-set precision run gives −0.000085 [−0.000225, +0.000054], and the nested interval is reported separately. The docstring is corrected in `swarm_audit/stats.py` (comment only, no behaviour change). A **post-review, disclosed** 30-batch check (`uncertainty_mc30.json`) gives MC SD 7.1e-5 against Codex's 5.4e-5 (20 batches) and our 7.8e-5 (10 batches). The table-only SD of the endpoint D is therefore poorly determined: about 0 (10 batches), 2.7e-5 (30 batches) or 5.8e-5 (Codex). We do not claim it is negligible. The endpoint conclusion is unchanged. |
| **M2** `is_primary_contrast` is true for the exploratory OLMo arm 91 | core code; the affected file is the extra arm's | **yes**: `analysis.py` tested only channel, n, graph and memory; OLMo's saved analysis has the flag true | **Fixed:** `analysis.is_primary` now also requires the frozen arm `("core_cheap_talk", 6)`. A test is in `tests/test_followup_core.py`. Llama arm 6 is still true; OLMo arm 91 and n = 20 are false. The extra arm's saved file is not ours to rewrite (the review posted this on PR 9). Every combined table here labels OLMo exploratory. |
| **M3** The theorem's proof and scope must accompany the paper; corrections to the setup-paper text | report authors | not re-derived here | `setup-paper/` and `report/` belong to the authors. We pass on `review-codex/independent/theory_review.md` and the wording fixes. Our follow-up already calls B_t "an estimate, not a certificate". |
| m1 First-entry vs every-entry diagnostic (`extra_followup_exact.py`) | extra arm | not ours | none (impact ≤ 2e-6 per the review) |
| m2 Stale status text in `results/README.md` and `results/validation.md` | core | **yes** | Updated with the core completion note (after the remaining GPU runs). |
| m3 Payoff and motivation wording; an unverifiable Greenblatt quote | report authors | n/a | Passed on. Do not use the quote unless verified. |
| Related work: Flint et al. full text (PMC) | core `related_work.md` | **not re-verified here** (arxiv.org and PMC are unreachable from this session); the review read the PMC full text | `related_work.md` is updated with the review's finding: Flint et al. cache restricted-token policies for all memory states (H = 5) and validate them against direct generations. So "first composition of cached LLM policies" and "first study of size effects" must not be claimed. Remaining distinction: transfer from **synthetic-history audits** to **full-history** swarm histories, and explicit state omission. |

Independent agreement worth recording (no action needed):

- **Early-round D.** The review's 400k reference gives about −0.000675; our 740k precision
  run gives −0.000737. These agree within simulation error, and the review also calls this
  result exploratory and not a model-superiority claim.
- **Nested endpoint interval.** The review gives [−0.000397, +0.000089]; ours is
  [−0.00047, +0.000086].
- **OLMo round-3 X.** The review gives 0.340 / 0.352 forecast against 0.136 observed,
  matching ours.
