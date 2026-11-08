"""Build the claim-to-evidence table from independent machine outputs."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
M='982822f554e547cc5c4468b70d5aaaae88356814'
E='4a817051abffbdf208eb38a2b09160ab9d49aa0f'
C='f020fe2a0c93c42177a48ccdf9141ead9c185d1a'

def main():
    exact=json.loads((ROOT/'independent/olmo_exact.json').read_text())
    primary=json.loads((ROOT/'independent/llama_primary.json').read_text())
    checks=json.loads((ROOT/'independent/secondary_checks.json').read_text())
    extra=json.loads((ROOT/'independent/extra_review.json').read_text())
    rows=[]
    def add(claim,status,result,condition,size,path,commit,checked,uncertainty,limitations,wording,review_output):
        rows.append(dict(claim=claim,evidence_status=status,result=result,condition=condition,sample_size=size,
            source_path=path,source_commit=commit,independently_checked=checked,uncertainty=uncertainty,
            limitations=limitations,recommended_wording=wording,review_output=review_output))
    ppath='results/analysis/core_cheap_talk/cheap_talk_n10_cycle_full.json'
    add('Original primary paired endpoint CRPS','original primary live + simulated forecasts','D=-0.0000864399; archived 95% CI [-0.0002210329,+0.0000353221]',
        'Llama cheap talk n10 cycle full history T20','50 independent runs; 20000 reference simulations per predictor',ppath,M,'yes: every predictor CDF and per-run paired score exactly',
        'whole-run paired bootstrap conditional on fitted tables and fixed simulated reference; 2000 draws',
        'does not include table or MC variability; interval crosses zero','No demonstrated incremental endpoint benefit from retaining signals in the designated primary condition.',
        'independent/llama_primary.json:original_paired; max_archived_D_per_run_error')
    add('Primary Monte Carlo variability','exploratory simulation sensitivity',str(primary['monte_carlo']['batch_SD']['endpoint']),
        'same fixed Llama tables and observed run set','20 independent batches of 20000 per predictor',
        'results/audit/llama-3.1-8b/cheap_talk_d2_signal.json; '+ppath,M,'yes: new implementation and seeds',
        'SD of D across simulated references; SD relative SE about 16%',
        'conditional on fitted table and observed runs','Simulation error is material relative to the tiny contrast; it does not reverse the endpoint null.',
        'independent/llama_primary.json:monte_carlo')
    add('Higher-precision primary endpoint comparison','exploratory simulation sensitivity',str(primary['monte_carlo']['pooled_D']['endpoint']),
        'same primary Llama condition','400000 simulations per predictor; same 50 live runs',ppath,M,'yes',
        'conditional run bootstrap; MC SE approximately 0.0000120681',
        'not a fresh experiment or a replacement preregistered result','Higher-precision references preserve the primary endpoint conclusion.',
        'independent/llama_primary.json:monte_carlo.pooled_D')
    add('Audit-table uncertainty','exploratory table sensitivity',str(primary['table_bootstrap']['summary']['endpoint']),
        'paired statewise resampling of intact signal-table fitting-history vectors; START fixed',
        '200 table resamples; 20000 simulated paths per predictor per resample',
        'results/audit/llama-3.1-8b/cheap_talk_d2_signal.json',M,'yes',
        'table+MC SD and nested run/table/MC interval; approximate table-only variance subtraction',
        '200 replicates give imprecise tails; excludes drift and model/prompt selection',
        'Original run intervals omit audit estimation; exploratory resampling does not change the endpoint null.',
        'independent/llama_primary.json:table_bootstrap')
    add('Primary early timing discrepancy','secondary trajectory statistic; independent check',
        'max binned TV 0.26145 actions / 0.28855 signal, round 2',
        'Llama n10 full-history cheap talk','50 runs; original 20k model references; independent 5000 null draws',ppath,M,'yes',
        'archived simultaneous thresholds 0.23758/0.23536; fresh thresholds about 0.23640/0.23395',
        'binned marginal TV is not full trajectory TV; adjustment covers rounds within one predictor',
        'Both finite-state audits miss early dynamics despite a null incremental endpoint score.',
        'independent/llama_primary.json:original_replication')
    add('Early-round CRPS comparison','exploratory temporal follow-up',str(primary['monte_carlo']['pooled_D']['early']),
        'mean per-round high-count CRPS rounds 2-7; actions minus signal',
        '50 live runs; 400k reference paths per predictor',ppath+'; scripts/followup_core_temporal.py',M+'; '+C,'yes: review implementation',
        'paired whole-run bootstrap conditional on table/reference; MC sensitivity separately',
        'post-outcome choice and multiplicity; small effect; no general predictor superiority',
        'In an exploratory early-round score, retaining signals does not repair this mismatch.',
        'independent/llama_primary.json:monte_carlo.pooled_D.early')
    add('Local defect and RHS estimate','secondary logged estimate',str(primary['local_errors']),
        'Llama n10 full history','10000 decisions clustered in 50 runs',ppath,M,'yes: local defects and B20',
        'population expectation estimated by run averages; any reported bands conditional on table',
        'binary and six-outcome TVs have different domains; B clips to 1',
        'Estimated local defects do not produce a nonvacuous certificate here.',
        'independent/llama_primary.json:local_errors')
    add('Defect-shift decomposition','secondary source result',
        '+0.01078159 = +0.00984951 within-state +0.00093208 frequency',
        'Llama n10 full-history actions-only table','50 runs; disjoint synthetic scoring histories',ppath,M,'source extraction; local aggregate independently agrees',
        'conditional on one fitted table and synthetic scoring set',
        'algebraic decomposition of one-step defect; not causal decomposition of group TV',
        'Within-state differences dominate this one-step defect shift, without identifying its cause.',
        'findings.md:reporting constraints')
    for label,status,indices in [('original','original calibration gate failure','0-49'),('confirmation','post-exception fresh confirmation','50-99'),('pooled','post-exception pooled confirmation','0-99')]:
        add('Llama G5 '+label,status,str(checks['llama_calibration'][label]),
            'Llama n4 last1 cheap talk; same enumerated signal kernel','runs '+indices,
            'results/gates/G5_cheap_talk'+('_confirmation' if label!='original' else '')+'.json; results/gates/G5_cheap_talk_exception.json',
            M,'yes: exact law, all calibration logs and original MC seed',
            '20k null draws; approximate MC p SE in independent output',
            'later confirmation does not erase earlier failure; pooled set is not independent',
            'Report the initial failure, diagnostics, authorised exception and subsequent confirmation separately.',
            'independent/secondary_checks.json:llama_calibration.'+label)
    op='results/extra/audit/olmo-3-7b/cheap_talk_d2_enumerated_signal.json'
    orun='results/extra/runs/olmo_calibration_cheap_talk/cheap_talk_n4_cycle_last1'
    add('OLMo full and actions-only final-five laws','exploratory exact-model calculation',
        'P(F=1)='+str(exact['variants']['both']['P_F1'])+' vs '+str(exact['actions_binary']['P_F1']),
        'OLMo n4 cycle last1 T20; full law on 1296 profiles; actions law on 16',
        '109 enumerated kernel states; no fitted-history sampling',op+'; '+op.replace('_signal','_actions'),E,
        'yes: reversed label order, independent transition/DP and killed-chain event check',
        'finite arithmetic, row/law checks at 3e-13; no sampling interval for cached law',
        'conditional on stationary constrained-label kernel, identity and product sampling',
        'The enumerated signal-aware law preserves a substantial late all-high component that the actions-only law largely loses.',
        'independent/olmo_exact.json:variants; actions_binary')
    add('OLMo observed late all-high event','exploratory live',
        '8/50; CI '+str(checks['olmo_live_event_CI_Clopper_Pearson_95']),
        'OLMo n4 cycle last1 rounds16-20','50 independent runs; 4000 decisions',orun,E,'yes: all rows, graph wiring, states and draws',
        'Clopper-Pearson two-sided 95% interval; event selected in exploratory analysis',
        'not instantaneous all-high or three-round consensus; interval is wide',
        'Eight of 50 runs bid high at every agent in every round16-20.',
        'independent/secondary_checks.json:olmo_live_event_CI_Clopper_Pearson_95')
    add('OLMo observed paired CRPS','exploratory live scoring',str(exact['paired_D']),
        'OLMo n4 last1; actions-only minus full','same 50 runs and exact forecast laws',orun+'; results/extra/analysis/olmo_calibration_cheap_talk/cheap_talk_n4_cycle_last1.json',E,'yes',
        'paired whole-run percentile bootstrap; review uses 50000 draws',
        'interval includes zero; exact-model contrast is distinct from uncertain observed advantage',
        'The observed score difference favours the full forecast but is not separated from zero at the reported interval.',
        'independent/olmo_exact.json:paired_D')
    add('Expected CRPS advantage under full reference','exploratory exact-model score identity',
        str(exact['expected_CRPS_difference_full_reference']),
        'expectation under enumerated full signal law','all 21 possible final-five counts',op,E,'yes: CDF-square and direct score expectation agree',
        'numerical tolerance, no live sampling interval',
        'proper-score identity does not independently validate the full law as truth',
        'Under the chosen full reference law, omitting signals incurs expected excess CRPS 0.020524.',
        'independent/olmo_exact.json:expected_CRPS_difference_full_reference')
    add('Own versus neighbour retention','exploratory exact E1 ablation',
        'own_only='+str(exact['variants']['own_only']['P_F1'])+'; neighbours_only='+str(exact['variants']['neighbours_only']['P_F1']),
        'OLMo n4 last1; autonomous chains; p_test averaging weights','same 109-state cached table',
        'results/extra/followup/audit_ablations.json',E,'yes: entire y laws within 6e-15',
        'exact conditional on cached kernel; no sampling interval',
        'weights are synthetic, retained variables include own-message feedback; no intent/safety claim',
        'Own-message retention nearly recovers this endpoint law; neighbour retention alone does not.',
        'independent/olmo_exact.json:variants; independent/secondary_checks.json:pipeline_comparison')
    add('Message-dependent transitions','exploratory kernel mechanism',str(exact['mechanism']),
        'H-2 slice and START; OLMo enumerated kernel','kernel entries rather than independent live samples',op,E,'yes',
        'recorded conditional probabilities; occupancy diagnostic is model-based',
        'do not factor actions and messages or multiply mutually incompatible hypothetical local states',
        'Under high-neighbour-message observations, own low has a sharply different next-action probability.',
        'independent/olmo_exact.json:mechanism')
    add('Actual model occupancy differs from averaging weights','exploratory oracle diagnostic',str(checks['occupancy_own_messages_rounds2_20']),
        'own message within all-high profiles, full chain, rounds2-20','exact-model occupancy, not swarm-fitted audit weights',
        'results/extra/followup/audit_ablations.json:models.olmo-3-7b.mechanism',E,'yes',
        'finite arithmetic',
        'oracle model occupancy cannot replace independent audit fitting without changing the method',
        'Own low is very rare in the reference chain all-high set but receives one-third synthetic weight.',
        'independent/secondary_checks.json:occupancy_own_messages_rounds2_20')
    add('None observation interventions','exploratory exact E2 counterfactual',str(extra['interventions']),
        'deterministic replacement of received own/neighbour message fields; n4 last1',
        'same cached kernel; no fresh live runs','results/extra/followup/observation_interventions.json',E,
        'yes: entire y laws within 5e-15','numerical tolerance',
        'none is meaningful content; cached counterfactual is not new live evidence',
        'Literal none preserves a near-full late all-high probability and differs sharply from synthetic averaging.',
        'independent/extra_review.json:interventions')
    add('Uniform replacement identity','exploratory model identity',
        'own randomisation = neighbours-only; neighbour randomisation = own-only; both = neither',
        'independent per-observer per-step uniform replacement',
        'finite products/sums over cached kernel','docs/overnight-extra-plan.md; results/extra/followup/observation_interventions.json',E,
        'analytically checked; endpoint averaging identity independently numerically checked',
        'identity conditional on independent noise','a shared replaced broadcast can correlate agent outputs and is different',
        'The uniform-replacement sensitivities reproduce E1 by construction, not as independent evidence.',
        'independent/olmo_exact.json:averaging_identity_max_y_error; independent/theory_review.md')
    add('First-entry survival correction','minor technical correction',
        'repeat-entry mass='+str(checks['entry_diagnostic']['mass_from_repeat_entries'])+'; max survival correction='+str(checks['entry_diagnostic']['survival']['both']['max_correction']),
        'OLMo all-high set by round20','1296-profile exact chain',
        'scripts/extra_followup_exact.py:diagnostics; results/extra/followup/plan.json',E,'yes: killed-chain first-hit calculation',
        'finite arithmetic','no endpoint or main mechanism change',
        'Use first-entry mass from paths that have not previously hit all-high, or relabel as an all-entry mixture.',
        'independent/secondary_checks.json:entry_diagnostic')
    add('OLMo full-history late outcome and timing','exploratory live + fitted surrogate forecasts',
        '50/50 y=0; observed round3 X=0.136; new model round3 means about0.340/0.352',
        'OLMo n10 cycle full history','50 live runs; review100k simulations per predictor',
        'results/extra/runs/olmo_core_cheap_talk/cheap_talk_n10_cycle_full; results/extra/audit/olmo-3-7b/cheap_talk_d2_signal.json',E,
        'yes: observed outcomes and new forecast simulations',
        'degenerate run bootstrap; zero event count has95% upper limit0.07112; model MC variability remains',
        'single fitted table; good late endpoint does not mean good trajectory',
        'Both surrogates forecast the late all-low outcome but are too slow early.',
        'independent/secondary_checks.json:olmo_n10_full_observed; independent/extra_review.json:olmo_n10_full_simulation')
    add('n4 full-history E3 CPU forecast','exploratory exact composition of fitted surrogate',
        str(extra['n4_full_history_surrogate_exact']),
        'OLMo n4 finite-state chain built from sampled full-history table','50 fit and25 score histories per nonSTART signal state',
        'results/extra/followup/e3_forecast.json:n4_full',E,'yes: P_F1=0.11159407613383451',
        'no propagation MC error; audit estimation and approximation remain',
        'not exact for the actual full-history swarm; not a live cell',
        'The fitted surrogate forecasts P(F=1)=0.11159; test against a separate live cell if available.',
        'independent/extra_review.json:n4_full_history_surrogate_exact')
    add('n10 last1 E3 CPU forecast','exploratory simulated forecast',str(extra['n10_last1_forecast']),
        'OLMo n10 enumerated last1 table; degree2 cycle','100000 review simulations per predictor',
        'results/extra/followup/e3_forecast.json:n10_last1',E,'yes: independent seeds; signal P_F1=0.00782, source0.00814 within MC variability',
        'signal MC SE about0.000279; zero-count action estimate needs binomial upper bound',
        'not a new live experiment; plug-in zero MC SE is not zero uncertainty',
        'The cached-kernel model forecasts a much rarer all-high event at n10; distinguish simulation from live validation.',
        'independent/extra_review.json:n10_last1_forecast')
    add('Counts controls','original secondary source results',
        'CRPS actions-only0.0423088 n10;0.0322966 n20','Llama cycle full history counts',
        '50 runs per cell','results/analysis/core_counts/counts_n10_cycle_full.json; results/analysis/core_counts/counts_n20_cycle_full.json',M,
        'source extraction and inventory only; no independent end-to-end replication',
        'run bootstrap conditional on fitted audit table',
        'does not prove exact transfer for full memory','Treat counts as a control with fitted-table error, not an exactness theorem application.',
        'independent/inventory.json')
    add('n100 scale arm','additional secondary source result','50 complete runs on one d4 random regular graph',
        'Llama counts n100 full history','50 runs; one fixed graph',
        'results/extra/arm8_numbers.json; results/extra/graph_rr4_n100.json',E,'inventory/source review only',
        'run-level sampling conditional on single graph and audit table',
        'size/degree/topology confounded relative to cycle n10/20','A scale/topology demonstration, not an isolated size effect or topology generalisation.',
        'independent/inventory.json')
    add('n20 cheap-talk completion','incomplete at initial core snapshot','25/50 committed logs; analysis absent',
        'Llama n20 cycle full history','committed run indices0-24',
        'results/runs/core_cheap_talk/cheap_talk_n20_cycle_full',C,'inventory only',
        'incomplete result; current job state not inspected',
        'do not use partial results as final 50-run cell','Mark partial until a complete committed snapshot and analysis are available.',
        'independent/inventory.json')
    add('TV bound and exact transfer','population mathematical statement','valid with explicit lifted-kernel proof',
        'finite histories; independent conditional outputs; action-compatible coarse wiring; history support',
        'not a sample-size claim','setup-paper/sections/2-setup.tex; setup-paper/sections/3-results.tex',M,
        'yes: complete proof and stronger-claim counterexamples','the population RHS is generally unknown',
        'finite estimates are not certificates; no verified novelty claim; binned TV ratios do not show tightness',
        'State the assumptions and distinguish theorem, population RHS and logged estimate.',
        'independent/theory_review.md')
    add('Exploratory classification','design chronology','OLMo trigger0.0553 exceeds0.05; extension chosen separately after results',
        'arm9a/9b and overnight E1-E3',
        'design/decision records, not random samples',
        'results/extra/arm9_trigger.json; results/extra/arm9_summary.md; docs/overnight-extra-plan.md',E,
        'records cross-checked; historical user approval not independently witnessed',
        'selection/multiplicity limitations remain',
        'is_primary_contrast boolean in arm9b output is incorrect',
        'Label the second model and follow-ups exploratory; later freezing does not retroactively preregister selection.',
        'independent/secondary_checks.json:OLMo_n10_archived_is_primary_contrast')
    fresh_path=ROOT/'independent/new_results_review.json'
    if fresh_path.exists():
        fresh=json.loads(fresh_path.read_text())
        fc=fresh['core_source_commit'];fe=fresh['extra_source_commit']
        add('Clock-indexed early improvement and late cost','exploratory C2 committed at final refresh','Review pooled-minus-clock CRPS: early+0.00597598; endpoint-0.00358139; early signed bias-0.0576836 to-0.0195103',
            'Llama n10 same50 runs; signal-aware pooled minus length-block clock audit','200k review simulations per predictor; source200k',
            'results/followup/core/clock/core_cheap_talk__cheap_talk_n10_cycle_full.json',fc,
            'yes: new time-inhomogeneous simulation; early/endpoint effects reproduce; middle differs by2.3 estimated combined MC SE',
            'conditional whole-run intervals; source nested table/run sensitivity read but not fully recomputed here',
            'same-run exploratory evaluation; block fitting counts9-27 per state; no empty cells; no causal content identification',
            'Length conditioning improves early forecasts but worsens later Llama scores; remaining failure is consistent with content shift, not proven to be caused by it.',
            'independent/new_results_review.json:C2_clock_comparison; signed_errors_by_block')
        add('New OLMo n10 last1 live cell','exploratory E3 live sample at final refresh','0/50 F1;50/50 F0; all saved probabilities and draws match the enumerated kernel',
            'OLMo n10 last1 cycle arm92; same enumerated kernel used at n4','50 live runs;10000 decisions',
            'results/extra/followup/runs/followup_olmo_n10_last1/cheap_talk_n10_cycle_last1',fe,
            'yes: every state/record/draw/seed/prompt/probability; all kernel/provenance checks pass',
            '0/50 F1; binomial95% upper0.07112; probability of0/50 about0.665 under p0.00814',
            'no new kernel information; low power for rare event; n4full and E2live absent in this snapshot',
            'The new live sample validates the enumerated-kernel serving and wiring path; zero events does not refute its rare-event forecast or complete the size-by-memory experiment.',
            'independent/new_results_review.json:E3_n10_last1_live; E3_provenance_check')
    continuation=ROOT/'independent/continuation_20261108/olmo.json'
    if continuation.exists():
        fresh=json.loads(continuation.read_text());fe=fresh['extra_source_commit']
        add('Completed E2 live observation-intervention validation','exploratory live; continuation',
            'F1 counts9/4/7/8 of50; MC p0.5437228/0.2295885/1/0.8440578;0 implementation mismatches',
            'OLMo n4 last1; unmodified/own-none/neighbour-none/both-none',
            '4 independent cells x50 runs;16000 decisions',
            'results/extra/followup/observation_live.json; results/extra/followup/runs/followup_obs_*',fe,
            'yes: every record, observed field, next state, probability, prompt hash, draw and seed; exact null p-values',
            'prespecified familywise alpha0.01/4; binomial intervals in output',
            'non-rejection is implementation consistency, not equivalence; uses existing enumerated prompts',
            'All four live intervention implementations agree with their cached-kernel laws at the specified checks.',
            'independent/continuation_20261108/olmo.json:E2_live')
        add('Completed n4 full-memory OLMo cell','exploratory new live cell; continuation',
            'F1=5/50;F0=45/50;P(F1)95%CI[0.0333,0.2181]',
            'OLMo n4 cycle full history arm93','50 runs;4000 decisions',
            'results/extra/followup/runs/followup_olmo_n4_full/cheap_talk_n4_cycle_full',fe,
            'yes: outcomes, graph/state wiring, draws, seeds and provenance',
            'Clopper-Pearson95%; fitted-surrogate forecast0.11159 has separate estimation/approximation uncertainty',
            'full-history kernel was not re-queried and is not exactly the fitted table',
            'The new live cell has five late all-high outcomes; distinguish it from the earlier CPU forecast.',
            'independent/continuation_20261108/olmo.json:E3.n4_full')
        add('Completed size-by-memory contrast table','exploratory live factorial cells; continuation',
            'F1 counts n4last1=8,n4full=5,n10last1=0,n10full=0, each/50',
            'OLMo cycle degree2 T20','4 cells x50 runs across separate sessions',
            'results/extra/followup/size_memory.json; results/extra/followup/size_memory.csv',fe,
            'yes: all outcomes and all four Newcombe intervals',
            'nominal95% Newcombe intervals; no joint session randomisation',
            'group-wide extreme event changes with n; no persistence observations from all-high starts at n10',
            'The estimated late all-high frequency is lower at n10, consistent with rarer all-high starts; memory effects are not established.',
            'independent/continuation_20261108/olmo.json:E3_contrasts')
        add('Multiplicity sensitivity of size contrast','exploratory reporting qualification; continuation',
            'Full-memory n10-n4=-0.10: nominal95%[-0.21360,-0.00898]; two-size-comparison correction[-0.23425,+0.00992]',
            'two size contrasts; four planned E3 contrasts as another sensitivity','50 independent runs per cell',
            'results/extra/followup/mechanism_summary.md; results/extra/followup/size_memory.json',fe,
            'yes: independently calculated Wilson/Newcombe intervals',
            'Bonferroni sensitivity, not a retroactive replacement of the specified nominal analysis',
            'full-memory size evidence is not robust to even two-comparison familywise correction',
            'Label the intervals nominal and the size pattern exploratory; avoid claiming both memory contrasts establish a corrected size effect.',
            'independent/continuation_20261108/olmo.json:E3_contrasts.size_within_full')
        add('Review first-entry correction verified','resolved minor finding m1; continuation',
            'P(ever allH)=0.15023057291869008; corrected survival curves agree within9e-16',
            'OLMo exact cached-kernel diagnostic','1296-profile chain',
            'scripts/extra_followup_exact.py; results/extra/followup/audit_ablations.json',fe,
            'yes: compared against independently calculated first-hit laws saved before the fix',
            'finite arithmetic','does not change endpoint law or main mechanism',
            'The specified first-entry diagnostic is now implemented correctly; retain the all-entry curve only as a labelled secondary result.',
            'independent/continuation_20261108/olmo.json:fixes.first_entry')
        add('Exploratory classification correction verified','resolved material finding M2; continuation',
            'OLMo arm91 corrected primary=false; original analysis retained as provenance',
            'explicit frozen arm identity plus original primary condition','classification/code review',
            'results/extra/analysis/classification.json; swarm_audit/analysis.py',fe,
            'yes: derived override checked; focused regression tests separately recorded',
            'not a sampling assertion','consumers must apply corrected classification rather than stale original JSON flag',
            'Use the derived classification map; the stored old analysis remains a historical record.',
            'independent/continuation_20261108/olmo.json:fixes.classification')
    with (ROOT/'claims.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print('Wrote',len(rows),'claim rows')

if __name__=='__main__':main()
