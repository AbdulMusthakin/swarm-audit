# Independent review: swarm-audit

The original primary and strongest OLMo calculations reproduce. No critical numerical
invalidation was found. Material corrections concern Monte Carlo uncertainty,
exploratory classification and causal wording. The original endpoint null survives.

- [Findings and concrete fixes](findings.md)
- [Claim-to-evidence table: 43 rows](claims.csv)
- [Reproduction, source commits and test results](reproduce.md)
- [Report wording, at most three figures and verified related work](report_notes.md)
- [Independent scripts and machine-readable outputs](independent/)
- [Complete finite-space proof and counterexamples to stronger interpretations](independent/theory_review.md)
- [Coordination notes](coordination.md)
- [Original task](task.txt)

Fixed sources: main `982822f`, initial core `f020fe2`, initial extra `4a81705`;
one final refresh reviewed core `3912f32` and extra `76a647e`. All exact source hashes
are in the reproduction record and claim table. Research branches and original
evidence were not changed. Only this isolated review branch was published, after
explicit user authorisation. No merge, paid compute or fresh model query occurred.

The scheduled continuation is complete. It reviewed core `f8ad9bb` and extra
`9aa3170` after one research-ref fetch, and independently checked the completed E2
live interventions, n4 full-memory OLMo cell, Llama n20 and both K10 arms. New
outputs are under `independent/continuation_20261108/`.

The earlier MC, primary-label, first-entry and causal-wording corrections were
verified in source. The live results reproduce; the report should retain nominal
interval/multiplicity qualifications. A new minor resume-identity gap is reproduced
with a scripted CPU backend; the real intervention logs are correct. The theorem
proof and setup wording still need incorporation by the report authors. No new
GPU work is required by this review.
