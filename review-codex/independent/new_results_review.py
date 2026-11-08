"""One final refresh: independent C2 clock calculation and new E3 live-log audit."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta
from primary_review import load_entries,compile_tables,simulate,scores,summarize,read_runs
from provenance_checks import check

def clock_entries(table):
    blocks=[]; counts=[]
    for lo,hi in [(1,6),(7,12),(13,19)]:
        b={}; ns=[]
        for key,e in table.items():
            if key=='START':b[key]=e;continue
            fit=np.asarray(e['fit']);L=np.asarray(e['fit_L']);assert len(L)==len(fit)
            sel=(L>=lo)&(L<=hi);ns.append(int(sel.sum()))
            b[key]={'kappa':fit[sel].mean(0) if sel.any() else e['kappa']}
        blocks.append(b);counts.append(ns)
    return blocks,counts

def simulate_clock(comp,R,rng,n=10):
    # New implementation: categorical CDF draws with explicit ring neighbour rolls.
    prev=rng.choice(6,size=(R,n),p=comp[0][3])
    out=np.empty((R,20),np.int8);out[:,0]=(prev<3).sum(1)
    for t in range(2,21):
        J=comp[0 if t<=7 else 1 if t<=13 else 2][0]
        a=prev//3;m=prev%3
        nh=np.roll((a==0).astype(np.int8),1,1)+np.roll((a==0).astype(np.int8),-1,1)
        sh=np.roll((m==0).astype(np.int8),1,1)+np.roll((m==0).astype(np.int8),-1,1)
        sl=np.roll((m==1).astype(np.int8),1,1)+np.roll((m==1).astype(np.int8),-1,1)
        c=J[a,nh,sh,sl,m].cumsum(2);c/=c[:,:,-1:]
        prev=(rng.random((R,n))[:,:,None]>=c).sum(2).clip(max=5)
        out[:,t-1]=(prev<3).sum(1)
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--core',type=Path,required=True);ap.add_argument('--extra',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    root=args.core;tab=load_entries(root/'results/audit/llama-3.1-8b/cheap_talk_d2_signal.json')
    blocks,ns=clock_entries(tab);comp=[compile_tables(b) for b in blocks]
    obs,rows,errs=read_runs(root/'results/runs/core_cheap_talk/cheap_talk_n10_cycle_full')
    both=[];clock=[];batches=[]
    for j in range(5):
        rng=np.random.default_rng(np.random.SeedSequence([20261108,1101,j]))
        p=simulate(compile_tables(tab),'signal',40000,rng);c=simulate_clock(comp,40000,rng)
        ps=scores(p,obs);cs=scores(c,obs)
        batches.append({k:float((ps[k]-cs[k]).mean()) for k in ps});both.append(p);clock.append(c)
    p=scores(np.concatenate(both),obs);c=scores(np.concatenate(clock),obs)
    diffs={k:p[k]-c[k] for k in p};summary=summarize(diffs)
    archived=json.loads((root/'results/followup/core/clock/core_cheap_talk__cheap_talk_n10_cycle_full.json').read_text())
    old=archived['comparison']['signal_aware']['improvement_pooled_minus_clock']
    keymap={'endpoint':'endpoint','early':'2-7','middle':'8-13','late':'14-20'}
    comparison={}
    for k,ok in keymap.items():
        ours=summary[k]['mean'];se=float(np.std([b[k] for b in batches],ddof=1)/np.sqrt(len(batches)))
        pipeline=old[ok]['mean'];pipeline_se=old[ok]['mc_sd_of_pooled_estimate']
        comparison[k]={'review':summary[k],'review_MC_SE_approx':se,'source_mean':pipeline,
            'difference_in_combined_MC_SE':float((ours-pipeline)/np.sqrt(se*se+pipeline_se*pipeline_se))}
    signed={'pooled':[[],[],[]],'clock':[[],[],[]]}
    for r in rows:
        t=r['round']
        if t==1:continue
        b=0 if t<=7 else 1 if t<=13 else 2;k=r['state_sig']
        signed['pooled'][b].append(r['p_H']-np.asarray(tab[k]['kappa'])[0].sum())
        signed['clock'][b].append(r['p_H']-np.asarray(blocks[b][k]['kappa'])[0].sum())
    result={'core_source_commit':'3912f32915bea8f3850c30b779e3be50522f5108',
            'extra_source_commit':'76a647ec5f62b361f4048dbca05d30956e8e42de',
            'C2_clock_comparison':comparison,'clock_fit_count_min_max':[[min(x),max(x)] for x in ns],
            'empty_fit_cells':sum(x==0 for group in ns for x in group),
            'signed_errors_by_block':{k:[float(np.mean(x)) for x in v] for k,v in signed.items()}}
    # New n10 last-1 live cell: do not confuse its observations with cached forecasts.
    folder='results/extra/followup/runs/followup_olmo_n10_last1/cheap_talk_n10_cycle_last1'
    tablepath='results/extra/audit/olmo-3-7b/cheap_talk_d2_enumerated_signal.json'
    observed,rows,errors=read_runs(args.extra/folder);entries=load_entries(args.extra/tablepath)
    maxdiff=0.;nerrors=0
    for r in rows:
        p=r['p_H'];joint=np.stack([p*np.asarray(r['sig_after_H']),(1-p)*np.asarray(r['sig_after_L'])])
        maxdiff=max(maxdiff,float(abs(joint-np.asarray(entries[r['state_sig']]['kappa'])).max()))
        nerrors+=r['prompt_sha256']!=entries[r['state_sig']]['prompt_sha256']
    y=observed[:,-5:].sum(1);events=int((y==50).sum());R=len(y)
    result['E3_n10_last1_live']={'n_runs':R,'events_F1':events,'events_F0':int((y==0).sum()),'y':y.tolist(),
        'F1_Clopper_Pearson_95':[float(beta.ppf(.025,events,R-events+1)) if events else 0.,float(beta.ppf(.975,events+1,R-events)) if events<R else 1.],
        'mean_F':float(y.mean()/50),'EX_t':(observed.mean(0)/10).tolist(),'log_errors':errors,
        'max_probability_difference_from_enumerated_kernel':maxdiff,'prompt_hash_errors':nerrors,
        'probability_of_no_F1_in50_under_forecast_p00814':float((1-.00814)**R),
        'interpretation':'fresh live run sample validates the same enumerated last1 kernel on n10 wiring; no new kernel information; zero high outcomes has low power for a forecast near0.8%'}
    result['E3_provenance_check']=check(args.extra,folder,tablepath,True)
    result['unreviewed_new_core_items']=['full C1/C2 200-resample arrays were read at summary level, not every value independently recomputed','counts and OLMo clock controls not independently re-simulated','no new controlled content intervention validates content causation']
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=1))

if __name__=='__main__':main()
