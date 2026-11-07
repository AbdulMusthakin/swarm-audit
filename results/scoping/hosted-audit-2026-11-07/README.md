# Hosted inference scoping evidence

7 November 2026. This package records exploratory and calibration runs on Adaption and OpenRouter. The OpenRouter outcome concerns the counts-only control. A communicating holdout was not run on either service.

## Adaption

- The preflight returned 96 valid first responses and replayed two four-agent, four-round games. In two fixed histories, changing both neighbours' messages from low to high moved high bids from 10/16 to 15/16 and from 1/16 to 4/16.
- The revised prompt clarified that messages do not enter the payoff formula. The expanded audit accepted 587 draws over 116 histories. The first four draws per history formed a balanced screen. Across 18 matched histories, high bids occurred in 9/72 draws after two high messages and 5/72 after two low messages.
- Adaption returned 156 HTTP 429 responses in 743 logged attempts. Collection stopped during the audit after the user-level inference throttle reduced throughput. The preflight and expanded prompt samples remain separate.

## OpenRouter

Gemma 4 26B A4B was pinned to CoreWeave BF16 with no fallbacks. Counts-only outputs used two one-token labels. Communicating outputs used six labels encoding bid and message jointly. The audit averaged raw allowed-token probabilities before normalising. Live calls used the same prompts and sampled one token, retrying outside the same allowed set.

| Check | Result |
| --- | --- |
| Single measurement | All 116 histories had full label coverage. Median repeat total variation was 0.039 and the maximum was 0.304. |
| Pooled measurements | The 16-readout stability check failed. The 32-readout check passed its median and six-test-history gates, while the maximum split-half distance across all states remained 0.291. |
| Prospective counts calibration | Frozen forecast 14.63% high bids vs 17.97% observed across the final three rounds of 96 games. The CDF gap was 0.197 vs a 0.136 reference threshold. |
| Most frequent identical prompt | Reported high-bid probability was 12.51% during isolation and 18.72% during live play. Four further isolated blocks averaged 12.02% to 13.62%. |
| Retrospective substitution | Replacing only that history with its live readout moved the predicted share to 18.88% and reduced the CDF gap to 0.054. This calculation reuses calibration data. |

The logs verify matching prompts and identify where the frozen forecast missed. They leave the cause of the probability shift unresolved. The calibration gate stopped the hosted study before the communicating holdout. This is evidence for using the controlled serving and fresh calibration already specified in [the build handoff](../../../docs/handoff-build.md). The action-only versus signal-aware comparison requires its own matched held-out games.

## Evidence and verification

- `adaption-*` contains manifests, summaries and a per-history table of the balanced draw counts.
- `openrouter-*` contains provider screens, stability checks, the frozen policies and forecast distributions, the counts-only game records and calibration diagnostics.
- The pooled-32 run includes the first 16 measurements. Do not add the 16-readout and 32-readout totals.
- The public package uses inspectable aggregate records, synthetic counts-game trajectories, and the frozen forecast distributions. It excludes raw request logs, prompts, response text, credentials and provider request IDs.

Run `python3 results/scoping/hosted-audit-2026-11-07/verify.py` to recompute the balanced Adaption contrast, reconstruct OpenRouter’s counts-only history visits and CDF gap and verify the same-prompt probability comparison.
