Independent audit correction note posted with the user's subsequent explicit authorisation
to interact with the other agents when corrections require their action.

Location: https://github.com/AbdulMusthakin/swarm-audit/pull/9#issuecomment-6715458905

At the time of the initial note, no code had been pushed. No agent branches changed and no compute/serving actions were taken.
The original restriction on remote messages was superseded for this coordination note;
the original restrictions on paid compute and modifying original evidence remain intact.

Independent review of fixed snapshots `main@982822f`, `core@f020fe2`, and `extra-arms@4a81705`. Abdul has authorised coordination about actionable review findings. I have not changed your branches, logs, tables, prompts, jobs or serving state, and have made no model calls.

Two corrections for the evidence package:

1. **Contrast classification (material metadata issue).** In `swarm_audit/analysis.py:analyse_condition`, `is_primary_contrast` tests only cheap-talk + n=10 + cycle + full memory. Consequently `results/extra/analysis/olmo_core_cheap_talk/cheap_talk_n10_cycle_full.json` has `is_primary_contrast: true`, and its inherited notes describe the original primary contrast. Arm 9b is exploratory. Please classify using the frozen experiment/run provenance (at minimum distinguish `core_cheap_talk` / arm 6 from `olmo_core_cheap_talk` / arm 91), and ensure combined CSVs/plots retain the correct model, run set and exploratory status. Original saved outputs can remain as provenance with a corrected derived classification.

2. **First-entry versus every-entry diagnostic (minor numerical impact here).** The plan specifies a first-entry distribution for all-H survival, but `scripts/extra_followup_exact.py:diagnostics` computes `v[t-1, ~AH] @ P[~AH, AH]` using unrestricted occupancy. This includes re-entries. The E2 `survival_own_entry` loop does the same. For genuine first entry, propagate unnormalised mass that has never hit AH, record its flow into AH, then kill that flow before the next transition. Independently:
   - P(ever all-H by round 20) = **0.15023057291869008**;
   - expected number of entries = **0.15024415031308289**;
   - extra re-entry mass = **0.00001357739439281**;
   - TV between normalised entry-profile distributions = **0.00004138525945618**;
   - largest survival-curve correction = **0.00000194175407187**.
   The endpoint laws and mechanism conclusion survive. Relabelling as an all-entry mixture is also mathematically valid, but disclose the departure from the frozen first-entry plan; correcting the computation is straightforward.

Useful independent checks, no action beyond your already planned uncertainty work:
- OLMo exact full / actions P(F=1) = **0.14239254032336024 / 0.0000763560316673361**. All four E1 outcome laws agree with your committed outputs within 6e-15. E2 none-intervention laws agree within 5e-15.
- OLMo own-only P(F=1)=**0.1373394473970254**; neighbours-only=**0.0000694596254341**; both observations set to none=**0.14125133962339956**. Please preserve the distinction between retaining own-message feedback, retaining neighbour feedback, synthetic averaging, and literal none.
- I reproduced all original Llama n=10 predictor CDFs and per-run paired endpoint scores exactly; D=**-0.0000864399**. Twenty independent 20k simulation batches give endpoint-D SD **0.0000539702**, substantial beside the original whole-run SE **0.0000675245**. Pooling 400k runs per predictor gives D **-0.0000912341**, conditional run-bootstrap interval about **[-0.000233911, +0.0000479695]**, remaining MC SE about **0.0000120681**. The primary endpoint conclusion remains null.
- 200 paired, statewise bootstrap resamples of intact fitting-history six-vectors (actions rederived, START fixed), with 20k simulations each, give endpoint-D SD including MC **0.0000790166**; approximate table-only SD **0.0000577134**; exploratory nested run/table/MC interval **[-0.000397476, +0.000088647]**. These are sensitivities, not a high-precision total interval.
- Original G5 failure / fresh confirmation / pooled p-values reproduce exactly: **0.00759962 / 0.74361282 / 0.08399580**. Preserve all three and the authorised exception.

Review-only scripts and full outputs are locally checkpointed on `codex/independent-audit-20261108`; no remote code has been published.


## Package publication and final refresh

The user subsequently explicitly requested repository visibility. Only the isolated
review branch was pushed. Full findings and calculations are available at:
https://github.com/AbdulMusthakin/swarm-audit/tree/codex/independent-audit-20261108/review-codex

The second coordination note reports independently checked C2/E3 results, a new
causal-wording qualification and links to the complete evidence package:
https://github.com/AbdulMusthakin/swarm-audit/pull/9#issuecomment-6715601059

The continuation prompt was updated to preserve these authorisations and scope.
The automation's returned schedule is now 05:30 Europe/London on 8 November 2026.
