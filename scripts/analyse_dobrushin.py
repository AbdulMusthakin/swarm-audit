#!/usr/bin/env python3
"""CPU-only Dobrushin comparison using frozen swarm-audit tables and logs.
Requires numpy, scipy, matplotlib. No model calls, fitting, or network access.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import binom
os.environ.setdefault('MPLCONFIGDIR', '/tmp/swarm-audit-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/swarm-audit-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def product_laws(p):
    """Independent Bernoulli laws; columns use the same binary order for all rows."""
    p = np.atleast_2d(p)
    out = np.ones((len(p), 1))
    for j in range(p.shape[1]):
        out = np.concatenate((out * (1 - p[:, j, None]), out * p[:, j, None]), axis=1)
    return out


def product_tv(p, q, batch=8):
    """Exact TV of product laws (float64 enumeration), not a sum/union bound."""
    p, q = np.atleast_2d(p), np.atleast_2d(q)
    ans = np.empty(len(p))
    for i in range(0, len(p), batch):
        a, b = product_laws(p[i:i+batch]), product_laws(q[i:i+batch])
        ans[i:i+batch] = np.abs(a - b).sum(axis=1) / 2
    return ans


def box_delta(lo, hi, n):
    """Analytic upper bound, evaluated in float64, on global TV delta.

    Max TV over product Bernoulli rows with all parameters in [lo,hi].
    Separate convexity pushes each p_i and q_i to interval endpoints. Coordinates
    where the endpoints coincide can be replaced by differing endpoints without
    decreasing TV (data processing). Permutation symmetry leaves n+1 orientations.
    Within each of the two orientation groups, binomial counts are sufficient.
    """
    best = 0.0
    for k in range(n + 1):
        l = n - k
        a = np.outer(binom.pmf(np.arange(k+1), k, lo),
                     binom.pmf(np.arange(l+1), l, hi))
        b = np.outer(binom.pmf(np.arange(k+1), k, hi),
                     binom.pmf(np.arange(l+1), l, lo))
        best = max(best, np.abs(a-b).sum()/2)
    return float(best)


def adjacency(n, kind):
    a = np.ones((n,n), dtype=int) - np.eye(n, dtype=int) if kind == 'complete' else np.zeros((n,n),dtype=int)
    if kind == 'cycle':
        for i in range(n):
            a[i,(i-1)%n] = a[i,(i+1)%n] = 1
    return a


def transition(K, adj):
    n = len(adj)
    profiles = ((np.arange(2**n)[:,None] >> np.arange(n)) & 1).astype(int)
    probs = K[1-profiles, profiles @ adj.T]
    return profiles, product_laws(probs)


def discount(e, rate):
    out = np.zeros_like(e, dtype=float)
    last = np.zeros(e.shape[:-1])
    for t in range(e.shape[-1]):
        last = e[...,t] + rate * last
        out[...,t] = last
    return out


def band(per_run, rng, B=2000):
    ix = rng.integers(len(per_run), size=(B,len(per_run)))
    z = per_run[ix].mean(1)
    return np.quantile(z, [0.025,0.975], axis=0).tolist()


def analyse(repo, path, rng):
    start = time.time()
    source = json.loads(path.read_text())
    cond, inputs = source['condition'], source['inputs']
    n, T = cond['n'], cond['T']
    tabpath = repo / inputs['tables'] / (inputs['tag'] + '_actions.json')
    tab = json.loads(tabpath.read_text())
    lookup = {s: e['kappa'] for s,e in tab['entries'].items()}
    d = tab['d']
    K = np.array([[lookup[f'{own}-{k}'] for k in range(d+1)] for own in 'HL'])
    a = adjacency(n, cond['graph'])
    own = float(np.abs(K[0]-K[1]).max())
    neigh = float(np.abs(np.diff(K,axis=1)).max())
    alpha = own + d * neigh
    upper = box_delta(K.min(), K.max(), n)
    P = profiles = None
    if n <= 10:
        profiles, P = transition(K,a)
        dist = pdist(P, metric='cityblock')
        delta = float(dist.max()/2)
        delta_kind = 'exact enumeration of all transition-row pairs'
        assert delta <= upper + 1e-10
    else:
        delta = upper
        delta_kind = 'uniform parameter-box upper bound (not exact coefficient)'
    # Round-1 START law is separate from the homogeneous round>=2 transition.
    runs = sorted((repo / inputs['runs']).glob('run_*.jsonl.gz'))
    assert len(runs) == cond['runs']
    p = np.full((len(runs),T,n), np.nan)
    q = np.full_like(p,np.nan)
    acts = np.full_like(p,np.nan)
    source_hashes = {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest(),
                     str(tabpath.relative_to(repo)): hashlib.sha256(tabpath.read_bytes()).hexdigest()}
    for r,f in enumerate(runs):
        source_hashes[str(f.relative_to(repo))] = hashlib.sha256(f.read_bytes()).hexdigest()
        with gzip.open(f,'rt') as fh:
            for line in fh:
                row=json.loads(line); t=row['round']-1; i=row['agent']
                assert np.isnan(p[r,t,i]), 'duplicate log row'
                p[r,t,i],q[r,t,i],acts[r,t,i] = row['p_H'],lookup[row['state']],row['action']=='H'
    assert np.isfinite(p).all() and np.isfinite(q).all() and np.isfinite(acts).all()
    errors = np.abs(p-q)
    eta = product_tv(p.reshape(-1,n),q.reshape(-1,n)).reshape(len(runs),T)
    eps = errors.sum(2)
    assert np.all(eta <= eps+1e-12)
    theorem = np.cumsum(eps,axis=1)
    assert np.allclose(theorem.mean(0),source['defects']['actions_only']['bound']['B'])
    same_defects = discount(eps,delta)
    sharp = discount(eta,delta)
    local = discount(errors.mean(2),alpha)
    mean_no_contraction = np.cumsum(errors.mean(2),axis=1)
    # Compute the predictor's roundwise mean exactly when n<=10; otherwise retain
    # the published 20,000-trajectory MC mean and disclose this approximation.
    if P is not None:
        law = product_laws(np.full((1,n),lookup['START']))[0]
        model_mean = []
        for t in range(T):
            if t: law = law @ P
            model_mean.append(float(law @ profiles.mean(1)))
    else:
        model_mean = source['X']['actions_only']['mean']
    swarm_mean = acts.mean((0,2))
    tv_observed = source['scores']['delta']['actions_only']
    if isinstance(tv_observed,dict):
        print('Unexpected delta dict:',tv_observed.keys(),flush=True)
        raise ValueError('delta format')
    floor = source['scores']['floor']['actions_only']
    out = dict(condition=cond, analysis=str(path.relative_to(repo)), table=str(tabpath.relative_to(repo)),
               delta=delta,delta_kind=delta_kind,delta_box_upper=upper,
               c_self=own,c_neighbour=neigh,alpha=alpha,rounds=list(range(1,T+1)),
               theorem1=theorem.mean(0).tolist(),
               joint_defect_without_contraction=np.cumsum(eta,axis=1).mean(0).tolist(),
               tv_contraction_same_local_defects=same_defects.mean(0).tolist(),
               tv_contraction_joint_defect=sharp.mean(0).tolist(),
               tv_contraction_joint_defect_ci=band(sharp,rng),
               eta_joint_mean=eta.mean(0).tolist(),eps_sum_mean=eps.mean(0).tolist(),
               mean_local_bound=local.mean(0).tolist(),mean_local_bound_ci=band(local,rng),
               mean_without_contraction=mean_no_contraction.mean(0).tolist(),
               observed_binned_tv=tv_observed,observed_tv_null=floor,
               observed_mean_error=np.abs(swarm_mean-np.array(model_mean)).tolist(),
               swarm_mean=swarm_mean.tolist(),model_mean=list(model_mean),
               model_mean_source='exact propagation' if P is not None else 'published 20,000-run Monte Carlo',
               source_sha256=source_hashes)
    print(f"{path.stem}: delta={delta:.6f}, alpha={alpha:.6f}, B20={out['theorem1'][-1]:.3f}, "
          f"same20={same_defects.mean(0)[-1]:.3f}, sharp20={sharp.mean(0)[-1]:.3f}, "
          f"local20={local.mean(0)[-1]:.3f} ({time.time()-start:.1f}s)",flush=True)
    return out


def plots(results,out,model_label='Llama 3.1 8B'):
    colors = dict(thm='#747d8c',same='#d99320',sharp='#006f87',obs='#25324a',local='#168367')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','axes.labelcolor':'#25324a','text.color':'#25324a',
                         'savefig.facecolor':'white','font.family':'DejaVu Sans'})
    full = [r for r in results if r['condition']['n']>4]
    for kind in ['tv','mean']:
        fig,axs=plt.subplots(2,3,figsize=(14.5,8),sharex=True,sharey=True)
        for ax,r in zip(axs.flat,full):
            c=r['condition']; t=np.array(r['rounds'])
            title=f"{'Cheap talk' if c['channel']=='cheap_talk' else 'Counts'} · {'cycle' if c['graph']=='cycle' else 'complete graph'} · n = {c['n']}"
            ax.set_title(title,fontsize=11,pad=24)
            coeff=(f"δ = {r['delta']:.3f}" if c['n']<=10 else f"δ ≤ {r['delta']:.3f}") if kind=='tv' else f"α = {r['alpha']:.3f}"+('  (no local contraction)' if r['alpha']>=1 else '')
            ax.text(.5,1.025,coeff,transform=ax.transAxes,ha='center',fontsize=10)
            ax.plot(t,np.minimum(r['theorem1'],1),color=colors['thm'],ls='--',lw=2)
            if kind=='tv':
                ax.plot(t,np.minimum(r['tv_contraction_same_local_defects'],1),color=colors['same'],ls=':',lw=2)
                ax.plot(t,np.minimum(r['tv_contraction_joint_defect'],1),color=colors['sharp'],lw=2.3)
                lo,hi=np.minimum(r['tv_contraction_joint_defect_ci'],1)
                ax.fill_between(t,lo,hi,color=colors['sharp'],alpha=.13)
                ax.plot(t,r['observed_binned_tv'],color=colors['obs'],lw=1.3,marker='o',ms=3)
            else:
                ax.plot(t,np.minimum(r['mean_without_contraction'],1),color=colors['same'],ls=':',lw=2)
                ax.plot(t,np.minimum(r['mean_local_bound'],1),color=colors['local'],lw=2.3)
                lo,hi=np.minimum(r['mean_local_bound_ci'],1)
                ax.fill_between(t,lo,hi,color=colors['local'],alpha=.13)
                ax.plot(t,r['observed_mean_error'],color=colors['obs'],lw=1.3,marker='o',ms=3)
            ax.set_ylim(-.025,1.055); ax.set_xlim(1,20); ax.set_xticks([1,5,10,15,20]); ax.grid(alpha=.16)
        for ax in axs[-1]: ax.set_xlabel('Round')
        for ax in axs[:,0]: ax.set_ylabel('TV distance / bound' if kind=='tv' else 'Absolute mean error / bound')
        lines=[Line2D([0],[0],color=colors['thm'],ls='--',lw=2),
               Line2D([0],[0],color=colors['same'],ls=':',lw=2),
               Line2D([0],[0],color=colors['sharp' if kind=='tv' else 'local'],lw=2.3),
               Line2D([0],[0],color=colors['obs'],marker='o',ms=3,lw=1.3)]
        labels=['Theorem 1 (capped)', 'TV contraction: same summed local defects',
                'TV contraction + exact joint one-step defect', 'Observed binned TV (50 runs)'] if kind=='tv' else [
                'Theorem 1 (capped)', 'Mean-specific accumulation, no contraction',
                'Local Dobrushin bound on mean / normalized Hamming', 'Observed absolute mean error (50 runs)']
        fig.suptitle('Dobrushin contraction versus Theorem 1: current-round TV' if kind=='tv' else
                     'Local contraction gives tighter bounds for the mean high-bid rate',fontsize=18,x=.5,y=.98)
        fig.legend(lines,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.055),fontsize=10)
        fig.text(.5,.025,f'Actions-only audit · {model_label} · pointwise 95% whole-run bootstrap shading, fitted tables fixed\n'
                 'Estimated bound curves are not confidence certificates. Empirical gaps include sampling error; all bounds capped at 1.',
                 ha='center',fontsize=9,color='#5c6778')
        fig.subplots_adjust(left=.065,right=.985,bottom=.21,top=.85,hspace=.5,wspace=.2)
        fig.savefig(out/f'dobrushin_{kind}.png',dpi=190)
        fig.savefig(out/f'dobrushin_{kind}.pdf')
        plt.close(fig)


def checks():
    p=np.array([[.1,.5,.9],[.3,.4,.6]])
    assert np.allclose(product_laws(p).sum(1),1)
    assert np.allclose(product_tv(p,p),0)
    assert np.allclose(product_tv(np.array([[.2]]),np.array([[.7]])),.5)
    assert np.allclose(product_tv(np.zeros((1,3)),np.ones((1,3))),1)
    # Exhaustively validate the box maximum in a non-symmetric interval for n<=4.
    for n in range(1,5):
        corners=((np.arange(2**n)[:,None]>>np.arange(n))&1)*.63+.13
        brute=pdist(product_laws(corners),'cityblock').max()/2
        assert abs(brute-box_delta(.13,.76,n))<1e-12
    e=np.array([[.1,.2,.3]])
    assert np.allclose(discount(e,1),np.cumsum(e,axis=1))
    assert np.allclose(discount(e,0),e)
    print('Mathematical implementation checks passed.',flush=True)


# The six full-history conditions in figure order, then the two calibration arms.
# 'core' is the 8B submission analysis (the default, unchanged); 'm70' is the Llama 3.1 70B
# runs for interest (interest/model-70b, configs/model-70b.yaml), with the same arm ids.
PRESETS = {
    'core': dict(root='results/analysis', label='Llama 3.1 8B', specs=[
        ('core_cheap_talk','cheap_talk_n10_cycle_full'),('core_cheap_talk','cheap_talk_n20_cycle_full'),
        ('second_topology_cheap_talk','cheap_talk_n10_complete_full'),
        ('core_counts','counts_n10_cycle_full'),('core_counts','counts_n20_cycle_full'),
        ('second_topology_counts','counts_n10_complete_full'),
        ('calibration_cheap_talk','cheap_talk_n4_cycle_last1_runs0-99'),
        ('calibration_counts','counts_n4_cycle_last1')]),
    'm70': dict(root='interest/model-70b/analysis', label='Llama 3.1 70B', specs=[
        ('m70_core_cheap_talk','cheap_talk_n10_cycle_full'),
        ('m70_core_cheap_talk_n20','cheap_talk_n20_cycle_full'),
        ('m70_second_topology_cheap_talk','cheap_talk_n10_complete_full'),
        ('m70_core_counts','counts_n10_cycle_full'),('m70_core_counts_n20','counts_n20_cycle_full'),
        ('m70_second_topology_counts','counts_n10_complete_full'),
        ('m70_calibration_cheap_talk','cheap_talk_n4_cycle_last1'),
        ('m70_calibration_counts','counts_n4_cycle_last1')]),
}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--preset',choices=sorted(PRESETS),default='core',
                    help="which analysed conditions: 'core' (8B, default) or 'm70' (70B, for interest)")
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);checks()
    preset=PRESETS[args.preset]; specs=preset['specs']
    rng=np.random.default_rng(20261108)
    res=[analyse(args.repo,args.repo/preset['root']/arm/(name+'.json'),rng) for arm,name in specs]
    sha=subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD'],text=True).strip()
    artifact={'repository':'https://github.com/AbdulMusthakin/swarm-audit','commit':sha,
              'scope':'actions-only predictor; marginal TV and mean, not full-trajectory contraction',
              'bootstrap':'2000 whole-run resamples; conditional on fixed audit tables and delta; pointwise 95%',
              'results':res}
    (args.out/'dobrushin_results.json').write_text(json.dumps(artifact,indent=2)+'\n')
    rows=[]
    for r in res:
        c=r['condition']
        for t in range(c['T']):
            rows.append(dict(channel=c['channel'],n=c['n'],graph=c['graph'],last_k=c['last_k'],round=t+1,
                             delta_used=r['delta'],delta_type=r['delta_kind'],alpha=r['alpha'],
                             **{k:r[k][t] for k in ['theorem1','tv_contraction_same_local_defects',
                             'tv_contraction_joint_defect','mean_local_bound','mean_without_contraction',
                             'observed_binned_tv','observed_mean_error','eta_joint_mean','eps_sum_mean']}))
    with (args.out/'dobrushin_curves.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    plots(res,args.out,preset['label'])
    print('Completed:',args.out,flush=True)

if __name__=='__main__':main()
