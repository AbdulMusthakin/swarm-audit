"""Independent Llama replication and uncertainty review; no project imports.

Original random-stream consumption is reproduced only to check archived numbers.
New batches and table bootstraps use distinct documented seeds. Categorical draws
preserve action-message dependence by sampling the six-way joint distribution.
"""
import argparse
import gzip
import json
import math
from pathlib import Path
import numpy as np

def load_entries(path):
    return json.loads(Path(path).read_text())['entries']

def compile_tables(signal):
    J=np.zeros((2,3,3,3,3,6)); primary=np.zeros((2,3)); equal=np.zeros((2,3))
    for a in range(2):
        for k in range(3):
            for h in range(3):
                for l in range(3-h):
                    count=(h,l,2-h-l)
                    weight=2/math.prod(math.factorial(x) for x in count)/9
                    for m,name in enumerate(('high','low','none')):
                        key=f'{"HL"[a]}-{k}-{h}-{l}-{2-h-l}-{name}'
                        vec=np.asarray(signal[key]['kappa']).reshape(6)
                        J[a,k,h,l,m]=vec
                        primary[a,k]+=weight/3*vec[:3].sum()
                        equal[a,k]+=vec[:3].sum()/18
    start=np.asarray(signal['START']['kappa']).reshape(6)
    return J,primary,equal,start

def simulate(compiled,mode,R,rng,n=10):
    J,primary,equal,start=compiled
    if mode=='order0':
        p=.05*start[:3].sum()+.95*sum(primary[a,k]*.5*math.comb(2,k)/4 for a in range(2) for k in range(3))
        return (rng.random((R,20,n))<p).sum(2)
    out=np.zeros((R,20,n),np.int8)
    if mode in ('actions','equal'):
        K=primary if mode=='actions' else equal
        prev=rng.random((R,n))<start[:3].sum(); out[:,0]=prev
        for t in range(1,20):
            nh=np.roll(prev,1,axis=1).astype(np.int8)+np.roll(prev,-1,axis=1)
            prev=rng.random((R,n))<K[1-prev.astype(np.int8),nh];out[:,t]=prev
    else:
        c=np.cumsum(start); c/=c[-1]
        prev=(rng.random((R,n))[:,:,None]>=c).sum(2).clip(max=5)
        out[:,0]=prev<3
        for t in range(1,20):
            actions=prev//3; msgs=prev%3
            high=(actions==0).astype(np.int8)
            nh=np.roll(high,1,1)+np.roll(high,-1,1)
            hm=(msgs==0).astype(np.int8); lm=(msgs==1).astype(np.int8)
            h=np.roll(hm,1,1)+np.roll(hm,-1,1); l=np.roll(lm,1,1)+np.roll(lm,-1,1)
            probs=J[actions,nh,h,l,msgs]
            c=np.cumsum(probs,axis=2); c/=c[:,:,-1:]
            prev=(rng.random((R,n))[:,:,None]>=c).sum(2).clip(max=5)
            out[:,t]=prev<3
    return out.sum(2)

def cdf(samples,maximum):
    return np.cumsum(np.bincount(samples,minlength=maximum+1))/len(samples)

def score(G,observed):
    return ((G[None,:]-(observed[:,None]<=np.arange(len(G))))**2).sum(1)/(len(G)-1)

def scores(counts,observed):
    end=score(cdf(counts[:,-5:].sum(1),50),observed[:,-5:].sum(1))
    byround=np.column_stack([score(cdf(counts[:,t],10),observed[:,t]) for t in range(20)])
    return {'endpoint':end,'early':byround[:,1:7].mean(1),
            'middle':byround[:,7:13].mean(1),'late':byround[:,13:20].mean(1)}

def summarize(scores,seed=20261108,B=20000):
    rng=np.random.default_rng(seed);idx=rng.integers(50,size=(B,50))
    return {k:{'mean':float(v.mean()),'bootstrap_95':np.quantile(v[idx].mean(1),[.025,.975]).tolist(),
               'whole_run_SE':float(v.std(ddof=1)/np.sqrt(len(v)))} for k,v in scores.items()}

def binned(counts,n=10):
    # Fractions [0,.2),[.2,.4),[.4,.6),[.6,.8),[.8,1].
    ix=np.minimum(counts//(n//5),4)
    return np.array([np.bincount(ix[:,t],minlength=5)/len(counts) for t in range(20)])

def read_runs(folder):
    arrays=[]; rowsall=[]; errors={'record':0,'state':0,'action_draw':0,'signal_draw':0,'seed_replay':0}
    design=json.loads((Path(folder)/'provenance.json').read_text())['payload']['design']
    for file in sorted(Path(folder).glob('run_*.jsonl.gz')):
        rows=[json.loads(x) for x in gzip.open(file,'rt')];rowsall.extend(rows)
        by={(x['round'],x['agent']):x for x in rows};n=rows[0]['n']; T=max(x['round'] for x in rows)
        assert len(by)==T*n
        A=np.zeros((T,n),int); rid=rows[0]['run']
        replay=np.random.default_rng(np.random.SeedSequence([design['base_seed'],design['arm_id'],design['condition_id'],rid]))
        for t in range(1,T+1):
            for i in range(n):
                r=by[t,i]; A[t-1,i]=r['action']=='H'
                # Original run stream: action and then signal per agent.
                errors['seed_replay']+=(replay.random()!=r['u_action'])
                errors['seed_replay']+=(replay.random()!=r['u_signal'])
                nbs=[by[t,(i-1)%n],by[t,(i+1)%n]]
                rec={'action':r['action'],'signal':r['signal'],'n_high':sum(x['action']=='H' for x in nbs),
                     'n_low':sum(x['action']=='L' for x in nbs),
                     **{'sig_'+m:sum(x['signal']==m for x in nbs) for m in ('high','low','none')}}
                errors['record']+=rec!=r['record']
                if t==1: s=ss='START'
                else:
                    p=by[t-1,i]['record'];s=f"{p['action']}-{p['n_high']}"
                    ss=s+f"-{p['sig_high']}-{p['sig_low']}-{p['sig_none']}-{p['signal']}"
                errors['state']+=s!=r['state'] or ss!=r['state_sig']
                errors['action_draw']+=('H' if r['u_action']<r['p_H'] else 'L')!=r['action']
                probs=np.asarray(r['sig_after_'+r['action']]); c=np.cumsum(probs); c/=c[-1]
                drawn=int(np.searchsorted(c,r['u_signal'],side='right').clip(max=2))
                errors['signal_draw']+=('high','low','none')[drawn]!=r['signal']
        arrays.append(A.sum(1))
    return np.asarray(arrays),rowsall,errors

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--table-resamples',type=int,default=200);ap.add_argument('--mc-batches',type=int,default=20)
    args=ap.parse_args();root=args.source
    table=load_entries(root/'results/audit/llama-3.1-8b/cheap_talk_d2_signal.json')
    compiled=compile_tables(table)
    observed,rows,errors=read_runs(root/'results/runs/core_cheap_talk/cheap_talk_n10_cycle_full')
    archived=json.loads((root/'results/analysis/core_cheap_talk/cheap_talk_n10_cycle_full.json').read_text())
    result={'source_commit':'982822f554e547cc5c4468b70d5aaaae88356814','n':10,'R':len(observed),
            'log_errors':errors,'observed_y':observed[:,-5:].sum(1).tolist(),'original_replication':{},'monte_carlo':{},'table_bootstrap':{}}
    rng=np.random.default_rng(8);original={}
    for name,mode in [('order0','order0'),('actions_only','actions'),('actions_only_equal_w','equal'),('signal_aware','signal')]:
        samples=simulate(compiled,mode,20000,rng); nullsamples=simulate(compiled,mode,20000,rng)
        original[name]=scores(samples,observed)
        if mode!='order0':
            maxG=float(abs(cdf(samples[:,-5:].sum(1),50)-archived['scores']['G'][name]).max())
            delta=.5*abs(binned(samples)-binned(observed)).sum(1)
            # 5000 null resamples of whole runs against this fixed reference.
            nr=np.random.default_rng(173+len(name)); bix=np.minimum(nullsamples//2,4); ref=binned(samples)
            maxnull=[]
            for _ in range(5000):
                pick=bix[nr.integers(20000,size=50)]
                law=np.array([np.bincount(pick[:,t],minlength=5)/50 for t in range(20)])
                maxnull.append(float((.5*abs(law-ref).sum(1)).max()))
            result['original_replication'][name]={'max_archived_G_difference':maxG,
                'scores':summarize(original[name]),'delta':delta.tolist(),
                'max_delta':float(delta.max()),'max_delta_round':int(delta.argmax()+1),
                'null_max_95':float(np.quantile(maxnull,.95)),
                'null_tail_p':float((1+sum(x>=delta.max()-1e-12 for x in maxnull))/5001)}
    paired={k:original['actions_only'][k]-original['signal_aware'][k] for k in original['actions_only']}
    result['original_paired']=summarize(paired)
    saved=np.asarray(archived['scores']['crps_per_run']['actions_only'])-archived['scores']['crps_per_run']['signal_aware']
    result['max_archived_D_per_run_error']=float(abs(paired['endpoint']-saved).max())
    batches=[]; sum_counts={mode:np.zeros((20,11),int) for mode in ('actions','signal')};sum_y={mode:np.zeros(51,int) for mode in ('actions','signal')}
    for batch in range(args.mc_batches):
        sc={}
        for mi,mode in enumerate(('actions','signal')):
            r=np.random.default_rng(np.random.SeedSequence([20261108,901,batch,mi]))
            pool=simulate(compiled,mode,20000,r);sc[mode]=scores(pool,observed)
            sum_y[mode]+=np.bincount(pool[:,-5:].sum(1),minlength=51)
            for t in range(20):sum_counts[mode][t]+=np.bincount(pool[:,t],minlength=11)
        batches.append({k:float((sc['actions'][k]-sc['signal'][k]).mean()) for k in sc['actions']})
    batch_sds={k:float(np.std([b[k] for b in batches],ddof=1)) for k in batches[0]}
    result['monte_carlo']={'n_model_per_batch':20000,'n_batches':args.mc_batches,'batch_D':batches,'batch_SD':batch_sds,
        'SE_of_batch_SD_relative_approx':1/math.sqrt(2*(args.mc_batches-1))}
    pooled={}
    for mode in ('actions','signal'):
        G=np.cumsum(sum_y[mode])/sum_y[mode].sum();end=score(G,observed[:,-5:].sum(1))
        rt=np.column_stack([score(np.cumsum(sum_counts[mode][t])/sum_counts[mode][t].sum(),observed[:,t]) for t in range(20)])
        pooled[mode]={'endpoint':end,'early':rt[:,1:7].mean(1),'middle':rt[:,7:13].mean(1),'late':rt[:,13:20].mean(1)}
    result['monte_carlo']['pooled_D']=summarize({k:pooled['actions'][k]-pooled['signal'][k] for k in pooled['actions']})
    result['monte_carlo']['pooled_n_model']=20000*args.mc_batches
    result['monte_carlo']['pooled_MC_SE']={k:v/math.sqrt(args.mc_batches) for k,v in batch_sds.items()}
    print('Original:',json.dumps(result['original_paired']),flush=True)
    print('Monte Carlo:',json.dumps({k:v for k,v in result['monte_carlo'].items() if k!='batch_D'}),flush=True)
    # Recompute local defect and bound straight from logs and independent table arrays.
    local=[];local_sig=[]
    for r in rows:
        if r['state']=='START':ka=compiled[3][:3].sum()
        else:
            a,k=r['state'].split('-');ka=compiled[1]['HL'.index(a),int(k)]
        local.append(abs(r['p_H']-ka))
        j=np.r_[r['p_H']*np.asarray(r['sig_after_H']),(1-r['p_H'])*np.asarray(r['sig_after_L'])]
        local_sig.append(.5*abs(j-np.asarray(table[r['state_sig']]['kappa']).reshape(6)).sum())
    result['local_errors']={'binary_mean':float(np.mean(local)),'joint_mean':float(np.mean(local_sig)),
                            'binary_B20':float(np.mean(local)*200),'joint_B20':float(np.mean(local_sig)*200)}
    # Paired table resampling: preserve six-vector and derive actions anew.
    boot=[]; nested=[]
    for b in range(args.table_resamples):
        rng=np.random.default_rng(np.random.SeedSequence([20261108,902,b]))
        bt={}
        for key,entry in table.items():
            a=np.asarray(entry['fit']); est=entry['kappa'] if key=='START' else a[rng.integers(len(a),size=len(a))].mean(0)
            bt[key]={'kappa':est}
        comp=compile_tables(bt)
        ac=scores(simulate(comp,'actions',20000,rng),observed);sg=scores(simulate(comp,'signal',20000,rng),observed)
        diffs={k:ac[k]-sg[k] for k in ac};idx=rng.integers(50,size=50)
        boot.append({k:float(v.mean()) for k,v in diffs.items()});nested.append({k:float(v[idx].mean()) for k,v in diffs.items()})
        if (b+1)%25==0:print('table resamples',b+1,flush=True)
    result['table_bootstrap']={'resamples':len(boot),'n_model':20000,'rows':boot,
                              'description':'statewise resampling of intact fitting-history six-vectors; START fixed; actions derived from each paired resample; contains MC variability'}
    result['table_bootstrap']['summary']={}
    if boot:
        for k in boot[0]:
            vals=np.array([b[k] for b in boot]);sd=float(vals.std(ddof=1))
            result['table_bootstrap']['summary'][k]={'mean':float(vals.mean()),'SD_including_MC':sd,
                'percentile_95':np.quantile(vals,[.025,.975]).tolist(),
                'table_only_SD_approx':float(np.sqrt(max(0,sd**2-batch_sds[k]**2))),
                'nested_run_table_MC_95':np.quantile([b[k] for b in nested],[.025,.975]).tolist()}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print('Table bootstrap:',json.dumps(result['table_bootstrap'].get('summary')),flush=True)

if __name__=='__main__': main()
