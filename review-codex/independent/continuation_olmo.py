"""Independent continuation: completed E2/E3, first-entry fix and multiplicity.

No project modules are imported. Uses the earlier independently calculated exact
laws, conditional on a byte-identical enumerated kernel, and rebuilds live wiring.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta, norm
from secondary_checks import gate
from provenance_checks import check as provenance_check

CORE='f8ad9bbb88ee538d1c9563c51e0a1d97e4401711'
EXTRA='9aa31705b9ae5ee45f27a87feeb1a2e2415f318f'
HERE=Path(__file__).resolve().parent

def read(path):return json.loads(Path(path).read_text())
def cp(k,n,alpha=.05):
    return [float(beta.ppf(alpha/2,k,n-k+1)) if k else 0.,
            float(beta.ppf(1-alpha/2,k+1,n-k)) if k<n else 1.]

def proportions_difference(k1,n1,k2,n2,alpha=.05):
    # Wilson roots expressed via the quadratic's centre and radius.
    z=norm.isf(alpha/2)
    def roots(k,n):
        centre=(k+z*z/2)/(n+z*z)
        radius=z*np.sqrt(k*(n-k)/n+z*z/4)/(n+z*z)
        return centre-radius,centre+radius
    p1,p2=k1/n1,k2/n2;l1,u1=roots(k1,n1);l2,u2=roots(k2,n2);delta=p1-p2
    return {'difference':delta,'interval':[float(delta-np.hypot(p1-l1,u2-p2)),
                                         float(delta+np.hypot(u1-p1,p2-l2))],'alpha':alpha}

def transformed(record,mode):
    r=dict(record)
    if mode in ('own_null','both_null'):r['signal']='none'
    if mode in ('neighbour_null','both_null'):
        r.update(sig_high=0,sig_low=0,sig_none=r['n_high']+r['n_low'])
    return r

def state(record):
    return f"{record['action']}-{record['n_high']}-{record['sig_high']}-{record['sig_low']}-{record['sig_none']}-{record['signal']}"

def audit_cell(folder,enum=None,mode=None):
    design=read(folder/'provenance.json')['payload']['design'];n=design['n']
    errors={k:0 for k in ('cardinality','record_wiring','observed_record','state','action_draw','signal_draw','seed_replay','prompt','joint_kernel')}
    counts=[];outcomes=[];starts=[]; maxprob=0.;row_count=0;ids=[]
    for f in sorted(folder.glob('run_*.jsonl.gz')):
        rid=int(f.name.split('_')[1].split('.')[0]);ids.append(rid)
        rows=[json.loads(s) for s in gzip.open(f,'rt')];by={(r['round'],r['agent']):r for r in rows}
        errors['cardinality']+=len(rows)!=20*n or len(by)!=20*n
        rng=np.random.default_rng(np.random.SeedSequence([design['base_seed'],design['arm_id'],design['condition_id'],rid]))
        ct=np.zeros(20,int)
        for t in range(1,21):
            for i in range(n):
                r=by[t,i];row_count+=1;ct[t-1]+=r['action']=='H'
                nbs=[by[t,(i-1)%n],by[t,(i+1)%n]]
                actual={'action':r['action'],'n_high':sum(x['action']=='H' for x in nbs),'n_low':sum(x['action']=='L' for x in nbs),
                        'signal':r['signal'],**{'sig_'+m:sum(x['signal']==m for x in nbs) for m in ('high','low','none')}}
                errors['record_wiring']+=actual!=r['record']
                expected=transformed(actual,mode)
                if mode is not None:errors['observed_record']+=r.get('observed_record')!=expected
                previous=transformed(by[t-1,i]['record'],mode) if t>1 else None
                k=state(previous) if previous else 'START';ka='-'.join(k.split('-')[:2]) if previous else k
                errors['state']+=r['state_sig']!=k or r['state']!=ka
                errors['seed_replay']+=rng.random()!=r['u_action'];errors['seed_replay']+=rng.random()!=r['u_signal']
                errors['action_draw']+=('H' if r['u_action']<r['p_H'] else 'L')!=r['action']
                probs=np.asarray(r['sig_after_'+r['action']]);c=probs.cumsum();c/=c[-1]
                mi=min(int(np.searchsorted(c,r['u_signal'],side='right')),2)
                errors['signal_draw']+=('high','low','none')[mi]!=r['signal']
                if enum:
                    ent=enum['entries'][k];j=np.stack([r['p_H']*np.array(r['sig_after_H']),(1-r['p_H'])*np.array(r['sig_after_L'])])
                    diff=float(abs(j-np.asarray(ent['kappa'])).max());maxprob=max(maxprob,diff)
                    errors['joint_kernel']+=diff>1e-14;errors['prompt']+=r['prompt_sha256']!=ent['prompt_sha256']
        counts.append(ct.tolist());outcomes.append(int(ct[-5:].sum()));starts.append(ct[0]==n)
    assert sorted(ids)==list(range(50)),ids
    events=sum(x==5*n for x in outcomes);zeros=outcomes.count(0)
    return {'n':n,'n_runs':len(ids),'decisions':row_count,'errors':errors,'y':outcomes,
        'kernel_check':'exact enumerated comparison' if enum else 'full-history kernel not independently re-queried; fitted table is an approximation',
        'F1_count':events,'F0_count':zeros,'F1_CP95':cp(events,len(ids)),
        'mean_F':float(np.mean(outcomes)/(5*n)),'EX_t':(np.mean(counts,axis=0)/n).tolist(),
        'round1_all_H_count':int(sum(starts)),'F1_without_round1_all_H':int(sum(y==5*n and not s for y,s in zip(outcomes,starts))),
        'max_probability_error':maxprob,'provenance_design':design}

def outcomes_only(folder,n):
    ys=[];starts=[]
    for f in sorted(folder.glob('run_*.jsonl.gz')):
        rows=[json.loads(s) for s in gzip.open(f,'rt')]
        assert len(rows)==20*n
        ys.append(sum(r['action']=='H' for r in rows if r['round']>=16))
        starts.append(sum(r['action']=='H' for r in rows if r['round']==1)==n)
    return {'n':n,'n_runs':len(ys),'F1_count':ys.count(5*n),'F0_count':ys.count(0),'mean_F':float(np.mean(ys)/(5*n)),
        'F1_CP95':cp(ys.count(5*n),len(ys)),'round1_all_H_count':sum(starts),
        'F1_without_round1_all_H':sum(y==5*n and not s for y,s in zip(ys,starts)),
        'integrity_note':'prior-pass log audits not rerun; combined outcomes re-read'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    root=args.source;er=root/'results/extra';follow=er/'followup'
    table_path=er/'audit/olmo-3-7b/cheap_talk_d2_enumerated_signal.json'
    enum=read(table_path);old=read(HERE/'olmo_exact.json');oldcheck=read(HERE/'secondary_checks.json')
    assert hashlib.sha256(table_path.read_bytes()).hexdigest()==old['source_sha256']
    result={'extra_source_commit':EXTRA,'core_source_commit':CORE,'enumerated_kernel_sha256':old['source_sha256'],
            'E2_live':{},'E3':{},'fixes':{}}
    modes={'unmodified':'both','own_null':'none_own','neighbour_null':'none_neighbours','both_null':'none_both'}
    archived=read(follow/'observation_live.json')['cells']
    for mode,exact_mode in modes.items():
        rel=f'results/extra/followup/runs/followup_obs_{mode}/cheap_talk_n4_cycle_last1';folder=root/rel
        r=audit_cell(folder,enum,mode);g=gate(old['variants'][exact_mode]['y_law'],r['y'],seed=20261108)
        g['alpha_per_cell']=.01/4;g['passes_prespecified_threshold']=g['MC_p']>.01/4
        r['per_run']=g;r['source_p_difference']=abs(g['MC_p']-archived[mode]['per_run']['p_value'])
        r['exact_P_F1']=old['variants'][exact_mode]['P_F1']
        r['provenance_check']=provenance_check(root,rel,'results/extra/audit/olmo-3-7b/cheap_talk_d2_enumerated_signal.json',True)
        manifest=read(folder.parent/'manifest.json')['conditions'][folder.name]
        r['recorded_session_observe']=[s.get('observe') for s in manifest['sessions']]
        r['observation_in_immutable_provenance']='observe' in read(folder/'provenance.json')['payload']['design']
        result['E2_live'][mode]=r
        print('E2',mode,r['F1_count'],'/50','p',g['MC_p'],'errors',sum(r['errors'].values()),flush=True)
    oldfolders={'n4_last1':('results/extra/runs/olmo_calibration_cheap_talk/cheap_talk_n4_cycle_last1',4),
                'n10_last1':('results/extra/followup/runs/followup_olmo_n10_last1/cheap_talk_n10_cycle_last1',10),
                'n10_full':('results/extra/runs/olmo_core_cheap_talk/cheap_talk_n10_cycle_full',10)}
    for name,(rel,n) in oldfolders.items():result['E3'][name]=outcomes_only(root/rel,n)
    fullrel='results/extra/followup/runs/followup_olmo_n4_full/cheap_talk_n4_cycle_full'
    new=audit_cell(root/fullrel);new['provenance_check']=provenance_check(root,fullrel,'results/extra/audit/olmo-3-7b/cheap_talk_d2_signal.json',False)
    result['E3']['n4_full']=new
    source=read(follow/'size_memory.json');result['E3_contrasts']={}
    pairs={'memory_within_n4':('n4_full','n4_last1'),'memory_within_n10':('n10_full','n10_last1'),
           'size_within_last1':('n10_last1','n4_last1'),'size_within_full':('n10_full','n4_full')}
    for label,(a,b) in pairs.items():
        A,B=result['E3'][a],result['E3'][b];k1,n1,k2,n2=A['F1_count'],A['n_runs'],B['F1_count'],B['n_runs']
        nominal=proportions_difference(k1,n1,k2,n2)
        sourceci=source['contrasts'][label]['P_F1_a_minus_b']
        result['E3_contrasts'][label]={'nominal95':nominal,
            'source_interval_max_error':float(max(abs(nominal['interval'][0]-sourceci['lo']),abs(nominal['interval'][1]-sourceci['hi']))),
            'two_comparison_FWER_sensitivity':proportions_difference(k1,n1,k2,n2,.05/2),
            'four_comparison_FWER_sensitivity':proportions_difference(k1,n1,k2,n2,.05/4)}
    for name,x in result['E3'].items():
        s=source['cells'][name];assert x['F1_count']==s['F1_count'] and x['F0_count']==s['F0_count']
    fixed=read(follow/'audit_ablations.json')['models']['olmo-3-7b']['mechanism']
    result['fixes']['first_entry']={'new_P_ever':fixed['P_ever_all_H_by_T'],
        'P_ever_max_error':abs(fixed['P_ever_all_H_by_T']-oldcheck['entry_diagnostic']['probability_ever_H_by20']),
        'survival_max_error':max(float(abs(np.asarray(fixed['survival_from_both_entry_distribution'][m])-v['correct_first_entry']).max()) for m,v in oldcheck['entry_diagnostic']['survival'].items())}
    cls=read(er/'analysis/classification.json')['files'];key='results/extra/analysis/olmo_core_cheap_talk/cheap_talk_n10_cycle_full.json'
    result['fixes']['classification']=cls[key]
    summary=(follow/'mechanism_summary.md').read_text()
    result['scope_notes']=['E2 non-rejection validates implementation consistency, not equivalence or the absence of a treatment effect',
        'The n4-full memory result is a fresh live sample, not an exact swarm law',
        'Size contrast intervals are nominal95%; full-memory size contrast crosses0 under two- or four-contrast Bonferroni sensitivity',
        'Cells were collected in distinct sessions, and F=1 is a group-wide extreme event whose START probability is p_start^n']
    result['all_new_live_integrity_checks_pass']=all(sum(v['errors'].values())==0 for v in result['E2_live'].values()) and sum(new['errors'].values())==0
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print('E3 counts',{k:v['F1_count'] for k,v in result['E3'].items()},flush=True)
    print('Fix checks',result['fixes']['first_entry'],flush=True)
    print('Full-memory size contrast',result['E3_contrasts']['size_within_full'],flush=True)

if __name__=='__main__':main()
