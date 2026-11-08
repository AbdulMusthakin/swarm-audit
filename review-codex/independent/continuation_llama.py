"""Independent new Llama n20 and complete-graph comparisons; offline CPU only."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import binom
from primary_review import compile_tables,simulate,cdf,score
from new_results_review import clock_entries,simulate_clock
from provenance_checks import fp

SOURCE='f8ad9bbb88ee538d1c9563c51e0a1d97e4401711'

def read(p):return json.loads(Path(p).read_text())
def load_cell(root,rel,tablepath):
    folder=root/rel;prov=read(folder/'provenance.json');d=prov['payload']['design'];n=d['n'];cheap=d['channel']=='cheap_talk';graph=d['graph']
    table=read(root/tablepath);tp=table['meta']['provenance']
    errors={k:0 for k in ('cardinality','record','state','action','signal','seed','identity','template')}
    errors['identity']+=fp(prov['payload'])!=prov['fingerprint']
    errors['identity']+=fp(prov['payload']['kernel'])!=prov['kernel_fingerprint']
    errors['identity']+=tp['kernel_fingerprint']!=prov['kernel_fingerprint'] or fp(tp['kernel'])!=tp['kernel_fingerprint']
    errors['identity']+=fp({'kernel':tp['kernel'],'design':tp['design']})!=tp['fingerprint']
    for channel,digest in prov['payload']['kernel']['templates'].items():
        errors['template']+=hashlib.sha256((root/f'swarm_audit/prompts/{channel}.txt').read_bytes()).hexdigest()!=digest
    counts=[];ids=[];max_reconstruction=0.;min_valid=1.;N=0
    for file in sorted(folder.glob('run_*.jsonl.gz')):
        rid=int(file.name.split('_')[1].split('.')[0]);ids.append(rid)
        rows=[json.loads(s) for s in gzip.open(file,'rt')];by={(r['round'],r['agent']):r for r in rows};ct=np.zeros(20,int)
        errors['cardinality']+=len(rows)!=20*n or len(by)!=20*n
        rng=np.random.default_rng(np.random.SeedSequence([d['base_seed'],d['arm_id'],d['condition_id'],rid]))
        for t in range(1,21):
            for i in range(n):
                r=by[t,i];N+=1;ct[t-1]+=r['action']=='H'
                nb=[j for j in range(n) if j!=i] if graph=='complete' else [(i-1)%n,(i+1)%n]
                others=[by[t,j] for j in nb]
                expected={'action':r['action'],'n_high':sum(x['action']=='H' for x in others),'n_low':sum(x['action']=='L' for x in others)}
                if cheap:expected.update(signal=r['signal'],**{'sig_'+m:sum(x['signal']==m for x in others) for m in ('high','low','none')})
                errors['record']+=expected!=r['record']
                previous=by[t-1,i]['record'] if t>1 else None
                k=f"{previous['action']}-{previous['n_high']}" if previous else 'START'
                errors['state']+=r['state']!=k
                if cheap:
                    ks=k+f"-{previous['sig_high']}-{previous['sig_low']}-{previous['sig_none']}-{previous['signal']}" if previous else k
                    errors['state']+=r['state_sig']!=ks
                errors['seed']+=rng.random()!=r['u_action']
                errors['action']+=('H' if r['u_action']<r['p_H'] else 'L')!=r['action']
                lp=np.array([r['logprob_H'],r['logprob_L']]);p=np.exp(lp-lp.max());p/=p.sum()
                max_reconstruction=max(max_reconstruction,float(abs(p[0]-r['p_H'])));min_valid=min(min_valid,r['p_valid'])
                if cheap:
                    errors['seed']+=rng.random()!=r['u_signal']
                    probs=np.asarray(r['sig_after_'+r['action']]);c=probs.cumsum();c/=c[-1]
                    mi=min(int(np.searchsorted(c,r['u_signal'],side='right')),2)
                    errors['signal']+=('high','low','none')[mi]!=r['signal']
                    for a in ('H','L'):
                        lp=np.array(r['sig_logprobs'][a]);p=np.exp(lp-lp.max());p/=p.sum()
                        max_reconstruction=max(max_reconstruction,float(abs(p-r['sig_after_'+a]).max()))
                        min_valid=min(min_valid,r['sig_p_valid_'+a])
        counts.append(ct)
    assert sorted(ids)==list(range(50)),ids
    summary={'run_folder':rel,'n':n,'graph':graph,'R':len(ids),'decisions':N,'errors':errors,
             'y':[int(x[-5:].sum()) for x in counts],'mean_X':(np.mean(counts,axis=0)/n).tolist(),
             'probability_reconstruction_max_error':max_reconstruction,'min_valid_label_mass':min_valid,
             'kernel_fingerprint':prov['kernel_fingerprint']}
    return np.asarray(counts),summary,table

def get_scores(counts,obs,n):
    endpoint=score(cdf(counts[:,-5:].sum(1),5*n),obs[:,-5:].sum(1))
    rt=np.column_stack([score(cdf(counts[:,t],n),obs[:,t]) for t in range(20)])
    return {'endpoint':endpoint,'early':rt[:,1:7].mean(1),'middle':rt[:,7:13].mean(1),'late':rt[:,13:20].mean(1)}

def summ(values,seed=20261108):
    rng=np.random.default_rng(seed);R=len(next(iter(values.values())));idx=rng.integers(R,size=(20000,R))
    return {k:{'mean':float(x.mean()),'paired_run_bootstrap95':np.quantile(x[idx].mean(1),[.025,.975]).tolist(),
               'whole_run_SE':float(x.std(ddof=1)/np.sqrt(R))} for k,x in values.items()}

class Hist:
    def __init__(self,n):self.n=n;self.y=np.zeros(5*n+1,int);self.rt=np.zeros((20,n+1),int);self.R=0
    def add(self,ct):
        self.R+=len(ct);self.y+=np.bincount(ct[:,-5:].sum(1),minlength=len(self.y))
        for t in range(20):self.rt[t]+=np.bincount(ct[:,t],minlength=self.n+1)
    def scores(self,obs):
        ep=score(self.y.cumsum()/self.R,obs[:,-5:].sum(1))
        rt=np.column_stack([score(self.rt[t].cumsum()/self.R,obs[:,t]) for t in range(20)])
        return {'endpoint':ep,'early':rt[:,1:7].mean(1),'middle':rt[:,7:13].mean(1),'late':rt[:,13:20].mean(1)}

def binary_complete(table,R,rng,n=10):
    ent=table['entries'];K=np.array([[ent[f'{a}-{k}']['kappa'] for k in range(n)] for a in ('L','H')])
    previous=rng.random((R,n))<ent['START']['kappa'];out=np.empty((R,20),np.int16);out[:,0]=previous.sum(1)
    for t in range(1,20):
        nh=previous.sum(1,keepdims=True)-previous
        previous=rng.random((R,n))<K[previous.astype(int),nh]
        out[:,t]=previous.sum(1)
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--batches',type=int,default=10);ap.add_argument('--batch-size',type=int,default=50000);args=ap.parse_args();root=args.source
    rel='results/runs/core_cheap_talk/cheap_talk_n20_cycle_full';tabrel='results/audit/llama-3.1-8b/cheap_talk_d2_signal.json'
    obs,integrity,table=load_cell(root,rel,tabrel);compiled=compile_tables(table['entries']);arch=read(root/'results/analysis/core_cheap_talk/cheap_talk_n20_cycle_full.json')
    result={'source_commit':SOURCE,'n20':{'integrity':integrity},'complete_graph':{}}
    rng=np.random.default_rng(8);orig={};errors={}
    for name,mode in [('order0','order0'),('actions_only','actions'),('actions_only_equal_w','equal'),('signal_aware','signal')]:
        ct=simulate(compiled,mode,20000,rng,n=20);simulate(compiled,mode,20000,rng,n=20)
        orig[name]=get_scores(ct,obs,20)
        if name!='order0':errors[name]=float(abs(cdf(ct[:,-5:].sum(1),100)-arch['scores']['G'][name]).max())
    result['n20']['archived_CDF_max_errors']=errors
    result['n20']['archived_paired']=summ({k:orig['actions_only'][k]-orig['signal_aware'][k] for k in orig['actions_only']})
    saved=np.array(arch['scores']['crps_per_run']['actions_only'])-arch['scores']['crps_per_run']['signal_aware']
    result['n20']['archived_per_run_D_max_error']=float(abs(saved-(orig['actions_only']['endpoint']-orig['signal_aware']['endpoint'])).max())
    print('n20 original D',result['n20']['archived_paired']['endpoint'],'CDF errors',errors,flush=True)
    hist={mode:Hist(20) for mode in ('actions','signal')};batches=[];signal_score_means=[]
    for b in range(args.batches):
        scores={}
        for j,mode in enumerate(('actions','signal')):
            rng=np.random.default_rng(np.random.SeedSequence([20261108,1201,b,j]))
            ct=simulate(compiled,mode,args.batch_size,rng,n=20);hist[mode].add(ct);scores[mode]=get_scores(ct,obs,20)
        batches.append({k:float((scores['actions'][k]-scores['signal'][k]).mean()) for k in scores['actions']})
        signal_score_means.append({k:float(x.mean()) for k,x in scores['signal'].items()})
        print('n20 MC batch',b+1,'/',args.batches,flush=True)
    a,s=hist['actions'].scores(obs),hist['signal'].scores(obs)
    result['n20']['higher_precision']={'runs_per_predictor':args.batches*args.batch_size,'paired':summ({k:a[k]-s[k] for k in a}),
        'MC_batches':batches,'estimated_pooled_MC_SE':{k:float(np.std([b[k] for b in batches],ddof=1)/np.sqrt(len(batches))) for k in batches[0]}}
    # New clock-indexed n20 comparison: use the independent length-block implementation.
    blocks,ns=clock_entries(table['entries']);clock_comp=[compile_tables(b) for b in blocks];chist=Hist(20);clock_score_means=[]
    for j in range(5):
        rng=np.random.default_rng(np.random.SeedSequence([20261108,1203,j]))
        ct=simulate_clock(clock_comp,40000,rng,n=20);chist.add(ct);cs=get_scores(ct,obs,20)
        clock_score_means.append({k:float(v.mean()) for k,v in cs.items()})
    cs=chist.scores(obs);result['n20']['clock']={'pooled_runs':hist['signal'].R,'clock_runs':chist.R,
        'pooled_minus_clock':summ({k:s[k]-cs[k] for k in s}),'fit_count_range':[[min(v),max(v)] for v in ns],
        'combined_MC_SE_approx':{k:float(np.sqrt(np.var([b[k] for b in signal_score_means],ddof=1)/len(signal_score_means)+np.var([b[k] for b in clock_score_means],ddof=1)/len(clock_score_means))) for k in s}}
    # Source table sensitivity is inspected, explicitly not re-simulated in this pass.
    follow=read(root/'results/followup/core/temporal/core_cheap_talk__cheap_talk_n20_cycle_full.json')
    result['n20']['source_table_sensitivity']=follow['table_resampling']['summary']['paired']['actions_only - signal_aware']['endpoint']
    result['n20']['source_nested_sensitivity']=follow['table_resampling']['summary']['paired_nested']['actions_only - signal_aware']['endpoint']
    for channel in ('counts','cheap_talk'):
        rel=f'results/runs/second_topology_{channel}/{channel}_n10_complete_full';tr=f'results/audit/llama-3.1-8b/{channel}_d9_actions_actions.json'
        ob,integ,tab=load_cell(root,rel,tr);saved=read(root/f'results/analysis/second_topology_{channel}/{channel}_n10_complete_full.json')
        rng=np.random.default_rng(8);rng.random((20000,20,10));rng.random((20000,20,10))
        ct=binary_complete(tab,20000,rng);G=cdf(ct[:,-5:].sum(1),50)
        gerr=float(abs(G-saved['scores']['G']['actions_only']).max())
        ent=tab['entries'];p0=ent['START']['kappa']/20
        for action in ('H','L'):
            for k in range(10):p0+=19/20*.5*math.comb(9,k)/512*ent[f'{action}-{k}']['kappa']
        order0=score(binom.cdf(np.arange(51),50,p0),ob[:,-5:].sum(1));act=score(G,ob[:,-5:].sum(1))
        diff=order0-act
        rng=np.random.default_rng(np.random.SeedSequence([20261108,1204,channel=='cheap_talk']))
        high=binary_complete(tab,100000,rng);a_hi=score(cdf(high[:,-5:].sum(1),50),ob[:,-5:].sum(1))
        result['complete_graph'][channel]={'integrity':integ,'archived_action_CDF_max_error':gerr,'order0_p':float(p0),
            'archived_order0_per_run_score_max_error':float(abs(order0-np.array(saved['scores']['crps_per_run']['order0'])).max()),
            'archived_action_per_run_score_max_error':float(abs(act-np.array(saved['scores']['crps_per_run']['actions_only'])).max()),
            'original_paired_order0_minus_actions':summ({'endpoint':diff})['endpoint'],
            'higher_precision_100k_paired_order0_minus_actions':summ({'endpoint':order0-a_hi})['endpoint']}
        print('Complete graph',channel,result['complete_graph'][channel]['original_paired_order0_minus_actions'],flush=True)
    result['interpretation']=['n20 is a secondary result; original n10 primary is unchanged',
        'A conditional n20 score advantage for actions-only does not establish the sign after audit-table uncertainty',
        'Clock conditioning can improve early forecasts while worsening endpoint and late scores',
        'K10 comparisons are within-topology order0-vs-action contrasts; changing degree, topology and fitted tables cannot isolate a single causal graph effect']
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print('n20 higher precision',result['n20']['higher_precision']['paired'],flush=True)
    print('n20 clock',result['n20']['clock']['pooled_minus_clock'],flush=True)

if __name__=='__main__':main()
