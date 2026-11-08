"""Check exploratory E2, E3 forecasts and OLMo n=10 using review-only code."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta
from exact_review import read_table,matrices,law
from primary_review import load_entries,compile_tables,simulate,binned

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--main',type=Path,required=True);ap.add_argument('--extra',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    er=args.extra/'results/extra';result={}
    tab=read_table(er/'audit/olmo-3-7b/cheap_talk_d2_enumerated_signal.json')
    pipe=json.loads((er/'followup/observation_interventions.json').read_text())['models']['olmo-3-7b']['processes']
    result['interventions']={}
    for mode,name in [('both','unmodified'),('none_own','own_null'),('none_neighbours','neighbour_null'),('none_both','both_null')]:
        P,v,h,_=matrices(tab,mode);l=law(P,v,h)
        result['interventions'][name]={'P_F1':l['P_F1'],'max_y_difference_from_pipeline':float(abs(np.array(l['y_law'])-pipe[name]['y_law']).max())}
    f={k:v['P_F1'] for k,v in result['interventions'].items()}
    result['factorial_interaction_P_F1']=f['unmodified']-f['own_null']-f['neighbour_null']+f['both_null']
    # The Llama control: independent exact laws, a distinct model and table.
    tab=read_table(args.main/'results/audit/llama-3.1-8b/cheap_talk_d2_enumerated_signal.json')
    result['llama_control']={}
    pp=json.loads((er/'followup/audit_ablations.json').read_text())['models']['llama-3.1-8b']['variants']
    for mode in ('both','neither','own_only','neighbours_only'):
        P,v,h,_=matrices(tab,mode);l=law(P,v,h)
        result['llama_control'][mode]={'P_F1':l['P_F1'],'E_F':l['E_F'],'max_y_difference_from_pipeline':float(abs(np.array(l['y_law'])-pp[mode]['y_law']).max())}
    table=load_entries(er/'audit/olmo-3-7b/cheap_talk_d2_signal.json');comp=compile_tables(table)
    archived=json.loads((er/'analysis/olmo_core_cheap_talk/cheap_talk_n10_cycle_full.json').read_text())
    obs=np.asarray(archived['X']['swarm']['mean']);result['olmo_n10_full_simulation']={}
    for j,mode in enumerate(('actions','signal')):
        counts=simulate(comp,mode,100000,np.random.default_rng(np.random.SeedSequence([20261108,1001,j])))
        y=counts[:,-5:].sum(1);m=counts.mean(0)/10
        key='actions_only' if mode=='actions' else 'signal_aware'
        result['olmo_n10_full_simulation'][mode]={'MC_runs':100000,'P_F0':float(np.mean(y==0)),
            'mean_F':float(np.mean(y/50)),'mean_F_MC_SE':float(np.std(y/50,ddof=1)/np.sqrt(len(y))),
            'CRPS_at_all_zero_observed':float(np.mean((np.cumsum(np.bincount(y,minlength=51))/len(y)-1)**2)*51/50),
            'EX_t':m.tolist(),'round3_mean_error_vs_observed':float(m[2]-obs[2]),
            'max_abs_EX_difference_from_archived_20k':float(abs(m-np.array(archived['X'][key]['mean'])).max())}
    # This is exact for a sampled surrogate table, not for an actual full-history swarm.
    tab=read_table(er/'audit/olmo-3-7b/cheap_talk_d2_signal.json');P,v,h,_=matrices(tab);l=law(P,v,h)
    result['n4_full_history_surrogate_exact']={k:v for k,v in l.items() if k not in ('y_law','EX_t')}
    result['n4_full_history_surrogate_exact']['label']='exact composition of a fitted finite-state surrogate; not exact for full-history LLM process'
    # Independent n=10 last-1 forecast; both inputs are enumerated.
    comp=compile_tables(load_entries(er/'audit/olmo-3-7b/cheap_talk_d2_enumerated_signal.json'))
    result['n10_last1_forecast']={}
    for j,mode in enumerate(('actions','signal')):
        counts=simulate(comp,mode,100000,np.random.default_rng(np.random.SeedSequence([20261108,1002,j])))
        y=counts[:,-5:].sum(1);p=float(np.mean(y==50))
        m=len(y); k=int((y==50).sum())
        ci=[float(beta.ppf(.025,k,m-k+1)) if k else 0.,float(beta.ppf(.975,k+1,m-k)) if k<m else 1.]
        result['n10_last1_forecast'][mode]={'MC_runs':100000,'P_F1':p,'P_F1_MC_SE':float(np.sqrt(p*(1-p)/len(y))),'mean_F':float(y.mean()/50),
            'P_F1_MC_binomial_95':ci,'MC_SE_note':'plug-in MC SE; a zero event count does not establish zero population probability'}
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=1))

if __name__=='__main__':main()
