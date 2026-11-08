"""Independent finite-chain calculation. No swarm_audit imports; NumPy only.

States here encode (L,none),(L,low),(L,high),(H,none),(H,low),(H,high)
in that order, deliberately different from the repository's order. Transition
probabilities are products over agents, with ring neighbours (i-1,i+1).
"""
import argparse
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np

SIGNALS = ('none', 'low', 'high')
OUTPUTS = tuple(itertools.product(('L', 'H'), SIGNALS))

def read_table(path):
    raw = json.loads(Path(path).read_text())
    return {k: np.asarray(v['kappa'], float) for k,v in raw['entries'].items()}

def source_joint(table, action, nh, counts, own):
    key = f'{action}-{nh}-{counts[0]}-{counts[1]}-{counts[2]}-{own}'
    return table[key]

def local_law(table, action, nh, counts, own, mode='both', scheme='primary'):
    if mode in ('none_own','none_both'):
        own = 'none'
    if mode in ('none_neighbours','none_both'):
        counts = (0,0,2)
    own_options = [(own,1.)]
    count_options = [(counts,1.)]
    if mode in ('neighbours_only','neither'):
        own_options = [(x,1/3) for x in SIGNALS]
    if mode in ('own_only','neither'):
        count_options = []
        for h in range(3):
            for l in range(3-h):
                c = (h,l,2-h-l)
                w = math.factorial(2)/math.prod(math.factorial(x) for x in c)/9
                count_options.append((c, w if scheme == 'primary' else 1/6))
    joint = sum(wc*wo*source_joint(table,action,nh,c,o)
                for c,wc in count_options for o,wo in own_options)
    # Table source order: actions H,L; signals high,low,none.
    return joint[::-1, ::-1].reshape(6)

def matrices(table, mode='both', scheme='primary'):
    profiles = list(itertools.product(range(6), repeat=4))
    arr = np.asarray(profiles)
    acts = arr//3  # 0 L, 1 H
    msgs = arr%3   # none,low,high
    high = acts.sum(1)
    start = table['START'][::-1,::-1].reshape(6)
    v = start[arr].prod(1)
    P = np.ones((len(arr),len(arr)))
    for i in range(4):
        neighbours = [(i-1)%4,(i+1)%4]
        nh = acts[:,neighbours].sum(1)
        sh = (msgs[:,neighbours]==2).sum(1)
        sl = (msgs[:,neighbours]==1).sum(1)
        laws = np.asarray([local_law(table, 'H' if a else 'L', int(k),
                          (int(h),int(l),int(2-h-l)),SIGNALS[int(m)],mode,scheme)
                          for a,k,h,l,m in zip(acts[:,i],nh,sh,sl,msgs[:,i])])
        P *= laws[:,arr[:,i]]
    assert np.max(abs(P.sum(1)-1)) < 2e-14
    assert abs(v.sum()-1) < 2e-14
    assert P.min() >= 0
    return P,v,high,arr

def law(P, initial, high):
    v = initial.copy()
    ex = []
    round15 = None
    for t in range(1,21):
        if t>1: v = v @ P
        ex.append(float(v @ high /4))
        if t==15: round15=v.copy()
    # Probability-generating-function propagation, rounds 16..20.
    coeff = np.zeros((21,len(v))); coeff[0]=round15
    survival = round15.copy()
    for t in range(5):
        incoming = coeff @ P
        coeff[:] = 0
        for j,h in enumerate(high):
            coeff[h:,j] = incoming[:21-h,j]
        survival = (survival @ P)*(high==4)
    y = coeff.sum(1)
    assert abs(y.sum()-1)<3e-13
    assert abs(y[-1]-survival.sum())<2e-14
    return {'y_law':y.tolist(),'P_F1':float(y[-1]),'P_F0':float(y[0]),
            'E_F':float(y @ np.arange(21)/20),'EX_t':ex,
            'survival_16_20':float(survival.sum()),
            'max_row_normalization_error':float(abs(P.sum(1)-1).max())}

def action_matrix(table):
    arr=np.asarray(list(itertools.product((0,1),repeat=4)))
    P=np.ones((16,16)); start=float(table['START'])
    v=np.prod(np.where(arr,start,1-start),axis=1)
    for i in range(4):
        nh=arr[:,[(i-1)%4,(i+1)%4]].sum(1)
        ps=np.array([float(table[f'{"H" if a else "L"}-{k}']) for a,k in zip(arr[:,i],nh)])
        P*=np.where(arr[None,:,i],ps[:,None],1-ps[:,None])
    return P,v,arr.sum(1)

def crps_scores(p, ys):
    F=np.cumsum(p)[:-1]
    return ((F[None,:]-(np.arange(len(F))[None,:]>=np.asarray(ys)[:,None]))**2).sum(1)/len(F)

def verify_logs(directory, table):
    ys=[]; errors={'state':0,'record_wiring':0,'draw_action':0,'draw_signal':0,
                  'probability':0,'prompt':0}; maxdiff=0.; seen=set(); hashes={}
    us=[]; vs=[]
    for file in sorted(Path(directory).glob('run_*.jsonl.gz')):
        rows=[json.loads(x) for x in gzip.open(file,'rt')]
        assert len(rows)==80
        by={(r['round'],r['agent']):r for r in rows}
        assert len(by)==80
        ys.append(sum(r['action']=='H' for r in rows if r['round']>=16))
        for r in rows:
            t,i=r['round'],r['agent']; rec=r['record']
            nbs=[by[t,(i-1)%4],by[t,(i+1)%4]]
            expected={'action':r['action'],'n_high':sum(x['action']=='H' for x in nbs),
                      'n_low':sum(x['action']=='L' for x in nbs),'signal':r['signal'],
                      'sig_high':sum(x['signal']=='high' for x in nbs),
                      'sig_low':sum(x['signal']=='low' for x in nbs),
                      'sig_none':sum(x['signal']=='none' for x in nbs)}
            errors['record_wiring']+=rec!=expected
            if t==1: key='START'
            else:
                p=by[t-1,i]['record']
                key=f"{p['action']}-{p['n_high']}-{p['sig_high']}-{p['sig_low']}-{p['sig_none']}-{p['signal']}"
            errors['state']+=key!=r['state_sig']; seen.add(key)
            a=r['action']; p=r['p_H']
            j=np.stack([p*np.asarray(r['sig_after_H']),(1-p)*np.asarray(r['sig_after_L'])])
            diff=float(abs(j-table[key]).max()); maxdiff=max(maxdiff,diff)
            errors['probability']+=diff>1e-14
            errors['draw_action']+=(('H' if r['u_action']<p else 'L')!=a)
            dist=np.asarray(r['sig_after_'+a]); draw=np.searchsorted(np.cumsum(dist)/dist.sum(),r['u_signal'],side='right')
            errors['draw_signal']+=('high','low','none')[min(draw,2)]!=r['signal']
            if key in hashes: errors['prompt']+=hashes[key]!=r['prompt_sha256']
            hashes[key]=r['prompt_sha256']; us.append(r['u_action']); vs.append(r['u_signal'])
    return {'n_runs':len(ys),'y':ys,'event_count':ys.count(20),'state_count':len(seen),
            'errors':errors,'max_joint_probability_difference':maxdiff,
            'action_signal_draw_correlation':float(np.corrcoef(us,vs)[0,1])}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--followups',action='store_true'); args=ap.parse_args()
    root=args.source; folder=root/'results/extra/audit/olmo-3-7b'
    source=folder/'cheap_talk_d2_enumerated_signal.json'; tab=read_table(source)
    result={'source_commit':'4a817051abffbdf208eb38a2b09160ab9d49aa0f',
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'encoding':{'actions':['L','H'],'signals':list(SIGNALS)},'variants':{}}
    modes=['both','neither']
    if args.followups: modes+=['own_only','neighbours_only','none_own','none_neighbours','none_both']
    for mode in modes:
        P,v,h,arr=matrices(tab,mode)
        result['variants'][mode]=law(P,v,h)
        if mode=='both':
            allhigh=h==4
            # Three all-high profiles with homogeneous messages; one-round action survival.
            result['mechanism']={}
            for m in SIGNALS:
                idx=list(map(tuple,arr)).index((3+SIGNALS.index(m),)*4)
                result['mechanism'][f'P_all_H_next_given_H_{m}']=float(P[idx,allhigh].sum())
            result['mechanism']['K_H_given_H2_neighbours_high']={m:float(tab[f'H-2-2-0-0-{m}'][0].sum()) for m in SIGNALS}
            result['mechanism']['start_joint_vs_product_TV']=float(.5*abs(tab['START']-np.outer(tab['START'].sum(1),tab['START'].sum(0))).sum())
    act=read_table(folder/'cheap_talk_d2_enumerated_actions.json')
    P,v,h=action_matrix(act); result['actions_binary']=law(P,v,h)
    result['averaging_identity_max_y_error']=float(np.max(abs(np.array(result['actions_binary']['y_law'])-result['variants']['neither']['y_law'])))
    derived={}
    for a in ('H','L'):
        for k in range(3): derived[f'{a}-{k}']=float(local_law(tab,a,k,(0,0,2),'none','neither')[3:].sum())
    result['derived_action_max_error']=max(abs(derived[k]-act[k]) for k in derived)
    result['mechanism']['K_action_H2']=float(act['H-2'])
    result['mechanism']['P_all_H_next_actions']=float(act['H-2']**4)
    obs=verify_logs(root/'results/extra/runs/olmo_calibration_cheap_talk/cheap_talk_n4_cycle_last1',tab)
    result['observations']=obs
    full=np.asarray(result['variants']['both']['y_law']); action=np.asarray(result['actions_binary']['y_law'])
    d=crps_scores(action,obs['y'])-crps_scores(full,obs['y'])
    rng=np.random.default_rng(20261108); boots=d[rng.integers(len(d),size=(50000,len(d)))].mean(1)
    result['paired_D']={'mean':float(d.mean()),'percentile_95':np.quantile(boots,[.025,.975]).tolist(),'B':50000}
    result['expected_CRPS_difference_full_reference']=float(np.mean((np.cumsum(action)[:-1]-np.cumsum(full)[:-1])**2))
    # Evaluate the expectation by integrating scores as a separate check.
    result['expected_CRPS_difference_by_summing_scores']=float(full@(crps_scores(action,range(21))-crps_scores(full,range(21))))
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('paired_D','expected_CRPS_difference_full_reference','derived_action_max_error','averaging_identity_max_y_error','mechanism')}))
    print(json.dumps({k:{m:v for m,v in x.items() if m in ('P_F1','P_F0','E_F')} for k,x in result['variants'].items()}))

if __name__=='__main__': main()
