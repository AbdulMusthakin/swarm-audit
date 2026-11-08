# Final report verification — 8 November 2026

Reviewed `docs/handoff-report-scores.md` (commit `07314a6`) against the latest report
at `928d3b8` on `claude/gracious-bardeen-42sf3t`. Another agent had already implemented
the main hand-off in `c3ed9cb` and the reference updates in `42c52a0` and `e75e800`.
This pass preserves that implementation and adds corrections and independent checks.

## Hand-off coverage

- The 70B branch was merged in `b820abb`; §4.5, Table 2, the settings table, the
  abstract, discussion, conclusion and appendices include the exploratory rerun.
  The prespecified status remains exclusive to the original 8B endpoint comparison.
- `report_70b.py` reproduces message responsiveness, paired CRPS differences, the
  unchanged D contrast, bias direction, roundwise flags, calibration and contraction
  summaries. The original analysis supplies the published D intervals; the separate
  bootstrap uses a different seed and need not have identical endpoints.
- `report_persistence.py` reproduces the all-high start and 76-agent-round survival,
  including exact propagation over the 81 all-high message profiles. Later first
  entry and re-entry explain the small difference from the final-five-round event.
- The compression rationale, three contributions, settings table and shorter §4.6
  are retained. Figure 1 and all report figure files are unchanged.
- The two collusion references are retained. Fish's arXiv page confirms the authors,
  title and EC 2026 acceptance; Motwani's arXiv page confirms its authors and title.
- The new hand-off numbers agree with the report sections and Appendix B. The source
  pins distinguish original 8B/OLMo evidence from the later 70B evidence.

## Corrections in this pass

- The per-round-tail simulation rejected 19,939/20,000 batches (99.695%, rounded to
  99.7%), so the abstract and discussion no longer say **over** 99.7%. The persistence
  test's separate 99.97% rate is explicit.
- The 5% target is labelled nominal; achieved false-positive rates remain disclosed.
  Practitioner advice specifies joint calibration across rounds, avoiding the
  implication that independent 5% pointwise tests control the overall false-positive rate.
- Exact last-round transfer is attributed to exhaustive signal-aware tables, rather
  than to all audits or to model size. Calibration's per-run threshold is 0.01.
- The persistence explanation separates accumulated forecast error from the noisy
  early rounds that set the shared cutoff. It no longer claims the late discrepancy
  is barely visible in roundwise shares.
- Complete-graph settings and directly fitted actions-only tables are described
  consistently. Contraction upper bounds for 20 agents are labelled as such.
- The 70B script now preserves LF line endings in the claim map and parses with
  Python 3.10 syntax, matching the package's declared minimum version.
- The LLM usage statement records the final Codex review.

## Numerical verification

Commands run from the repository root (CPU only; no new model queries):

```bash
OPENBLAS_NUM_THREADS=1 python scripts/report_persistence.py
python scripts/report_70b.py
python interest/model-70b/compare.py
OPENBLAS_NUM_THREADS=1 python -I scripts/followup_core_check.py
OPENBLAS_NUM_THREADS=1 python -I scripts/extra_followup_check.py
OPENBLAS_NUM_THREADS=1 python scripts/report_fresh_cohort.py
OPENBLAS_NUM_THREADS=1 python scripts/report_sensitivity.py
OPENBLAS_NUM_THREADS=1 python scripts/report_final_check.py
git diff --check
```

Both existing independent re-derivations pass and reproduce their committed outputs.
The fresh-cohort, sensitivity, 70B and comparison outputs reproduce without numerical
drift; persistence changes only its explanatory note. `report_final_check.py` imports
no experiment or report-generation code: all **75 checks pass**, including independently
reconstructed survival, score differences, bootstrap intervals and the F_error sign.
Its deterministic record, `results/followup/report_final_check.json`, includes input
SHA-256 hashes. The order-zero F_error is checked against its exact scored CDF, not
the separately simulated roundwise mean.

## PDF and links

Built with `cd report && latexmk -pdf main.tex`, using pdfTeX/TeX Live and the report's
Carlito and Inconsolata fonts. The result has **16 pages**, with references starting
on page **8** (seven pages before references, within the eight-page cap). The abstract
has **207 whitespace-delimited words**, within 150–250. The final LaTeX log contains
no warnings, undefined citations/references or overfull/underfull boxes. All pages
were rendered and visually inspected; the source and generated PDF are committed together.

Repository and source links resolve. Four publisher landing pages (Kirchner, Flint,
Flint Ashery and Hourdin) returned HTTP 403 to automated requests; Crossref returned
HTTP 200 and matching DOI/title/author metadata for all four. The NeurIPS proceedings
site returned HTTP 503; the existing proceedings citation is retained and a working
arXiv link is now clickable alongside it. These external responses are documented
as access limitations, not treated as evidence of broken citations.

The content commit is followed only by the source-pin update and its rebuilt PDF,
following the repository's existing convention: the pin identifies the complete
content revision without requiring a commit to contain its own hash.
