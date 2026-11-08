"""Check against exact hidden-signal calibration dynamics, with no sampled runs."""
import sys
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import pdist
sys.path.insert(0,str(Path(__file__).parent))
from analyse_dobrushin import product_laws,product_tv,transition,adjacency,checks

repo=Path(sys.argv[1]); out=Path(sys.argv[2])
checks()
# Optional third argument: the audit-table folder (default: the core's 8B tables).
root=repo/(sys.argv[3] if len(sys.argv)>3 else 'results/audit/llama-3.1-8b')
actions=json.loads((root/'cheap_talk_d2_enumerated_actions.json').read_text())['entries']
signal=json.loads((root/'cheap_talk_d2_enumerated_signal.json').read_text())['entries']
n=4; adj=adjacency(n,'cycle'); states=np.indices((6,)*n).reshape(n,-1).T
act=(states//3==0).astype(int); msg=states%3
k=act@adj.T; sh=(msg==0).astype(int)@adj.T; sl=(msg==1).astype(int)@adj.T
J=np.empty((len(states),n,6)); Q=np.empty((len(states),n))
for r in range(len(states)):
 for i in range(n):
  a='H' if act[r,i] else 'L'; m=['high','low','none'][msg[r,i]]
  key=f'{a}-{k[r,i]}-{sh[r,i]}-{sl[r,i]}-{2-sh[r,i]-sl[r,i]}-{m}'
  J[r,i]=np.array(signal[key]['kappa']).reshape(6)
  Q[r,i]=actions[f'{a}-{k[r,i]}']['kappa']
Ptrue=np.ones((len(states),len(states)))
for i in range(n): Ptrue*=J[:,i,states[:,i]]
assert np.allclose(Ptrue.sum(1),1)
K=np.array([[actions[f'{a}-{k}']['kappa'] for k in range(3)] for a in 'HL'])
profiles,Pmodel=transition(K,adj); delta=pdist(Pmodel,'cityblock').max()/2
alpha=np.abs(K[0]-K[1]).max()+2*np.abs(np.diff(K,axis=1)).max()
startsig=np.array(signal['START']['kappa']).reshape(6)
mu=np.prod(startsig[states],axis=1)
nu=product_laws(np.full((1,n),actions['START']['kappa']))[0]
index=act@(2**np.arange(n)); true_p=J[:,:,:3].sum(2)
eta_by_state=product_tv(true_p,Q)
e_by_state=np.abs(true_p-Q).mean(1)
actual=[]; bounds=[]; mean_actual=[]; mean_bounds=[]; prev_d=prev_m=0.
for t in range(20):
 if t:
  eta=float(mu@eta_by_state); eps=float(mu@e_by_state)
  prev_d=eta+delta*prev_d; prev_m=eps+alpha*prev_m
  mu=mu@Ptrue; nu=nu@Pmodel
 marginal=np.bincount(index,weights=mu,minlength=16)
 tv=np.abs(marginal-nu).sum()/2
 mean=abs(marginal@profiles.mean(1)-nu@profiles.mean(1))
 assert tv<=prev_d+1e-12,(t,tv,prev_d)
 assert mean<=prev_m+1e-12,(t,mean,prev_m)
 actual.append(tv);bounds.append(prev_d);mean_actual.append(mean);mean_bounds.append(prev_m)
res={'check':'exact 1296-state hidden-signal calibration chain vs 16-state actions-only audit',
 'pass':True,'rounds':20,'delta':float(delta),'alpha':float(alpha),'tv_actual':actual,
 'tv_bound_exact_expected_defects':bounds,'mean_error_actual':mean_actual,'mean_bound_exact_expected_defects':mean_bounds,
 'notes':'No live-run sampling: population expectations are computed from the enumerated full chain. START mismatch is zero.'}
out.write_text(json.dumps(res,indent=2)+'\n')
print('All 20 exact-population TV and mean inequalities pass. Round20:',actual[-1],bounds[-1],mean_actual[-1],mean_bounds[-1])
