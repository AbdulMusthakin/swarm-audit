# Extra-arm results imported for the combined report (read-only)

**Source:** branch `extra-arms` at commit `9aa3170` (8 Nov 2026, about 01:35 UTC),
`results/extra/followup/` (`mechanism_summary.md`, `claims.csv`, `audit_ablations.json`,
`observation_interventions.json`, `observation_live.json`, `size_memory.json`).

These files are **not** copied into the core branch. The extra session owns them, and they
reach `main` through PR 9. Labels are kept exactly as the extra session gave them. Every row
is **exploratory** (chosen after arm 9a was seen; specification frozen in its `plan.json`
before computing).

## Key imported numbers (OLMo-3-7B, n = 4 cycle, cheap talk, last-round history unless stated)

| quantity | value | evidence type (extra's label) | source key |
|---|---|---|---|
| P(F = 1), full signal-aware table | 0.1424 | exact (cached kernel) | `audit_ablations.json` models.olmo-3-7b.variants.both.P_F1 |
| P(F = 1), own last message kept | 0.1373 | exact | variants.own_only.P_F1 |
| P(F = 1), neighbour messages kept | 0.00007 | exact | variants.neighbours_only.P_F1 |
| P(F = 1), actions-only audit | 0.00008 | exact | variants.neither.P_F1 |
| Live F = 1, arm 9a | 8/50 | live | `size_memory.json` cells.n4_last1 |
| Live paired D, actions-only − signal-aware | +0.0252 [−0.0034, +0.0596] | live, whole-run bootstrap | variants.neither.live_crps_minus_both |
| K(H \| own H, 2 high neighbours, own "low") vs own "high"/"none" | 0.68–0.73 vs 0.9987–0.9996 | enumerated table | kernel_grid_P_H.H-2 |
| P(F = 1) with own / neighbour / both messages shown as "none" | 0.1412 / 0.1409 / 0.1413 | exact model-based counterfactual | `observation_interventions.json` processes.* |
| Size × memory, P(F = 1): n4 last-1 / n4 full / n10 last-1 / n10 full | 8/50, 5/50, 0/50, 0/50 | live (cells from different sessions) | `size_memory.json` cells |
| Size contrast within last-1 / full | −0.16 [−0.29, −0.06] / −0.10 [−0.21, −0.01] | live, Newcombe | contrasts.size_within_* |
| Llama control: P(F = 1) under every variant | ≤ 7e-11 | exact | models.llama-3.1-8b.variants |

## Integration notes

1. **One interval, two upper bounds.** The arm 9a live paired D appears as
   +0.025 [−0.003, **+0.054**] in `results/extra/arm9_summary.md` (also 0.0539 in the Codex
   review) and as [−0.0034, **+0.0596**] in the follow-up's `claims.csv`. The mean and lower
   bound agree to nine digits. Per-run D takes few distinct values, so two bootstrap
   draws with different seeds can share a lower quantile and differ at the upper one. This
   looks like bootstrap Monte Carlo variation, not an error, but the report should quote one
   version and say which (B = 2000, whole runs). Owner: extra session.
2. **How the two follow-ups fit together.** Both find that the audit's synthetic averaging
   injects states that the swarm rarely visits.
   - **Extra, OLMo n = 4 last-1:** an own-"low" message, weighted 1/3, destroys
     persistence.
   - **Core, Llama and OLMo n = 10 full history:** pooling over history lengths over-predicts
     early high bids, and the residual is consistent with a history-content mismatch.

   Both point to the same audit design question: what distribution over histories and
   observations the isolated tests should use. Neither shows intention or collusion.
3. **Classification.** The extra arm's saved analysis
   `results/extra/analysis/olmo_core_cheap_talk/cheap_talk_n10_cycle_full.json` carries
   `is_primary_contrast: true` from the old classifier (Codex review M2). The core
   classifier is fixed (`analysis.is_primary`). Treat that flag as **false**: the primary
   contrast is core arm 6 only.
