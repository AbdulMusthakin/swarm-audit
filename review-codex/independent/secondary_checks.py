"""Independent gate chronology, pipeline comparison and first-entry correction."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta
from exact_review import read_table, matrices, law, verify_logs, action_matrix

def gate(law_y,ys,sims=20000,seed=5):
    py=np.asarray(law_y);G=np.cumsum(py/py.sum())
    obs=np.cumsum(np.bincount(ys,minlength=21))/len(ys)
    gap=float(abs(obs-G).max())
    rng=np.random.default_rng(seed)
    hist=rng.multinomial(len(ys),py/py.sum(),size=sims)
    null=abs(np.cumsum(hist,axis=1)/len(ys)-G).max(1)
    p=(1+(null>=gap-1e-12).sum())/(sims+1)
    return {'gap':gap,'MC_p':float(p),'MC_p_SE_approx':float(np.sqrt(p*(1-p)/sims)),
            'n_runs':len(ys),'mean_y':float(np.mean(ys)),'sims':sims}

def entry_distribution(P,start,AH,T=20,first=True):
    outside=start.copy();outside[AH]=0
    flows=[start[AH]]
    uncond=start.copy()
    for _ in range(1,T):
        if first:
            incoming=outside@P
            flows.append(incoming[AH]);outside=incoming;outside[AH]=0
        else:
            flows.append(uncond[~AH]@P[np.ix_(~AH,AH)])
            uncond=uncond@P
    flows=np.asarray(flows)
    return flows.sum(0)/flows.sum(),flows.sum(1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--main',type=Path,required=True);ap.add_argument('--extra',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    result={'source_main':'982822f554e547cc5c4468b70d5aaaae88356814','source_extra':'4a817051abffbdf208eb38a2b09160ab9d49aa0f'}
    tab=read_table(args.main/'results/audit/llama-3.1-8b/cheap_talk_d2_enumerated_signal.json')
    P,v,h,_=matrices(tab);ll=law(P,v,h)
    folder=args.main/'results/runs/calibration_cheap_talk/cheap_talk_n4_cycle_last1'
    integrity=verify_logs(folder,tab);ys=integrity.pop('y')
    result['llama_calibration_integrity']=integrity
    result['llama_calibration']={label:gate(ll['y_law'],part) for label,part in [('original',ys[:50]),('confirmation',ys[50:]),('pooled',ys)]}
    result['llama_calibration_exact_mean_y']=float(np.asarray(ll['y_law'])@np.arange(21))
    exdir=args.extra/'results/extra';tab=read_table(exdir/'audit/olmo-3-7b/cheap_talk_d2_enumerated_signal.json')
    P,v,h,profiles=matrices(tab);full=law(P,v,h);AH=h==4
    recorded=json.loads((exdir/'followup/audit_ablations.json').read_text())['models']['olmo-3-7b']
    result['pipeline_comparison']={}
    for mode in ('both','neither','own_only','neighbours_only'):
        Q,sv,hh,_=matrices(tab,mode);l=law(Q,sv,hh)
        old=recorded['variants'][mode]
        result['pipeline_comparison'][mode]={'max_y_law_difference':float(abs(np.asarray(l['y_law'])-old['y_law']).max()),'P_F1':l['P_F1']}
    pi_first,first_mass=entry_distribution(P,v,AH,first=True)
    pi_repeat,repeat_mass=entry_distribution(P,v,AH,first=False)
    result['entry_diagnostic']={'probability_ever_H_by20':float(first_mass.sum()),'expected_number_H_entries_by20':float(repeat_mass.sum()),
        'mass_from_repeat_entries':float(repeat_mass.sum()-first_mass.sum()),'TV_entry_profile_distribution':float(.5*abs(pi_first-pi_repeat).sum()),
        'first_entry_mass_per_round':first_mass.tolist(),'all_entries_mass_per_round':repeat_mass.tolist(),
        'survival':{}}
    for mode in ('both','neither','own_only','neighbours_only'):
        Q=matrices(tab,mode)[0][np.ix_(AH,AH)]
        a=pi_first.copy();b=pi_repeat.copy();ss=[];rr=[]
        for t in range(20):
            ss.append(float(a.sum()));rr.append(float(b.sum()));a=a@Q;b=b@Q
        old=recorded['mechanism']['survival_from_both_entry_distribution'][mode]
        result['entry_diagnostic']['survival'][mode]={'correct_first_entry':ss,'all_entries':rr,
            'max_correction':float(abs(np.array(ss)-rr).max()),'max_reproduce_old_error':float(abs(np.array(rr)-old).max())}
    # Occupancy diagnostic: conditioning is from reference model, never audit fitting.
    comp=np.zeros(3);total=0
    for t in range(1,21):
        if t>1:v=v@P
        if t>=2:
            for j in range(3):comp[j]+=float(v[AH]@((profiles[AH]%3)==j).mean(1))
            total+=float(v[AH].sum())
    result['occupancy_own_messages_rounds2_20']=dict(zip(('none','low','high'),(comp/total).tolist()))
    result['olmo_live_event_CI_Clopper_Pearson_95']=[float(beta.ppf(.025,8,43)),float(beta.ppf(.975,9,42))]
    result['zero_of50_event_CI_Clopper_Pearson_95']=[0.,float(beta.ppf(.975,1,50))]
    # Count outcomes directly for OLMo full-memory extension.
    n10folder=exdir/'runs/olmo_core_cheap_talk/cheap_talk_n10_cycle_full'
    outcomes=[]; x=[]
    for f in sorted(n10folder.glob('run_*.jsonl.gz')):
        rows=[json.loads(s) for s in gzip.open(f,'rt')]
        assert len(rows)==200
        outcomes.append(sum(r['action']=='H' for r in rows if r['round']>=16))
        x.append([sum(r['action']=='H' for r in rows if r['round']==t)/10 for t in range(1,21)])
    result['olmo_n10_full_observed']={'n_runs':len(outcomes),'y':outcomes,'EX_t':np.mean(x,axis=0).tolist(),
        'all_low_by_round':int(np.where((np.array(x)==0).all(0))[0][0]+1)}
    result['OLMo_n10_archived_is_primary_contrast']=json.loads((exdir/'analysis/olmo_core_cheap_talk/cheap_talk_n10_cycle_full.json').read_text())['is_primary_contrast']
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('llama_calibration','pipeline_comparison','occupancy_own_messages_rounds2_20')}))
    print(json.dumps({k:v for k,v in result['entry_diagnostic'].items() if k not in ('survival','first_entry_mass_per_round','all_entries_mass_per_round')}))
    print(json.dumps({k:v['max_correction'] for k,v in result['entry_diagnostic']['survival'].items()}))

if __name__=='__main__':main()
