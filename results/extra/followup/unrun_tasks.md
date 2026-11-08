# Overnight follow-up: unrun and deferred tasks

Every task the hand-off specified is listed here, with its status.

| task | status |
|---|---|
| E0 starting check and frozen plan | done (`d9ec5fc`) |
| E1 four audit variants, exact, OLMo | done (`audit_ablations.json`) |
| E1 equal-weight sensitivity | done |
| E1 live scoring on the 50 arm 9a runs | done |
| E1 occupancy, entry, exit and survival diagnostics | done |
| E1 Llama control (enumerated table and 4b runs 0–49) | done |
| E2 four deterministic interventions, exact, with the factorial interaction | done (`observation_interventions.json`) |
| E2 full own-message grid | done (`kernel_grid_P_H` in `audit_ablations.json`; heatmap `figures/appx_kernel_grid_*`) |
| E2 uniform-replacement sensitivity | done, as an identity with the E1 laws (see the plan's deviations) |
| E2 live validation, 4 × 50 runs | see `completion_note.md` |
| E3 n = 10 last-1, 50 live runs | done (arm_id 92) |
| E3 n = 4 full history, 50 live runs | see `completion_note.md` |
| E3 CPU forecasts before spend | done (`e3_forecast.json`) |
| Independent calculation and CRPS identity | done (`independent_check.json`, PASS) |
| Figure 2 candidate and appendix figures | done (`figures/`) |
| `mechanism_summary.md`, `claims.csv`, completion note | done |

## Not attempted (outside tonight's scope by the hand-off)

- **Out of scope:** free-text communication, a new model, a 70B model, graph sweeps, and
  any change to payoffs, temperature, tokens or prompts.
- **Literature comparison:** the hand-off assigns it to the core agent, so it is not
  duplicated here.
- **Mediation analysis of any full-history effect:** it would need further interventions
  under full history. This was not specified for tonight and was not run.
