"""Offline validation of saved kernel identities and enumerated prompt records."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

def fp(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()

def check(root,folder,table_path,enumerated):
    prov=json.loads((root/folder/'provenance.json').read_text())
    tab=json.loads((root/table_path).read_text());tp=tab['meta']['provenance']; payload=prov['payload']
    errors={'run_fingerprint':int(fp(payload)!=prov['fingerprint']),
            'run_kernel_fingerprint':int(fp(payload['kernel'])!=prov['kernel_fingerprint']),
            'table_fingerprint':int(fp({'kernel':tp['kernel'],'design':tp['design']})!=tp['fingerprint']),
            'table_kernel_fingerprint':int(fp(tp['kernel'])!=tp['kernel_fingerprint']),
            'table_run_kernel_match':int(tp['kernel']!=payload['kernel']),
            'template_hashes':0,'table_prompt':0,'table_raw_joint':0,'draw_seed_replay':0}
    for channel,h in payload['kernel']['templates'].items():
        errors['template_hashes']+=hashlib.sha256((root/f'swarm_audit/prompts/{channel}.txt').read_bytes()).hexdigest()!=h
    max_norm=0.;minpv=1.;N=0;max_prob_from_logprob=0
    for key,e in tab['entries'].items():
        j=np.asarray(e['kappa']);max_norm=max(max_norm,abs(j.sum()-1))
        if 'raw' in e:
            r=e['raw'];re=np.stack([r['p_H']*np.asarray(r['sig']['H']),(1-r['p_H'])*np.asarray(r['sig']['L'])])
            errors['table_raw_joint']+=float(abs(re-j).max())>1e-14
    design=payload['design']
    for f in sorted((root/folder).glob('run_*.jsonl.gz')):
        rid=int(f.name.split('_')[1].split('.')[0]);seed=np.random.default_rng(np.random.SeedSequence([design['base_seed'],design['arm_id'],design['condition_id'],rid]))
        rows=sorted([json.loads(s) for s in gzip.open(f,'rt')],key=lambda r:(r['round'],r['agent']))
        for r in rows:
            N+=1
            errors['draw_seed_replay']+=seed.random()!=r['u_action']
            errors['draw_seed_replay']+=seed.random()!=r['u_signal']
            minpv=min(minpv,r['p_valid'],r['sig_p_valid_H'],r['sig_p_valid_L'])
            if enumerated:errors['table_prompt']+=r['prompt_sha256']!=tab['entries'][r['state_sig']]['prompt_sha256']
            # Independent renormalization of the saved log-probabilities.
            lp=np.asarray([r['logprob_H'],r['logprob_L']]);p=np.exp(lp-lp.max());p/=p.sum()
            max_prob_from_logprob=max(max_prob_from_logprob,abs(p[0]-r['p_H']))
            for a in ('H','L'):
                lp=np.asarray(r['sig_logprobs'][a]);p=np.exp(lp-lp.max());p/=p.sum()
                max_prob_from_logprob=max(max_prob_from_logprob,float(abs(p-r['sig_after_'+a]).max()))
    return {'folder':folder,'table':table_path,'errors':errors,'decisions':N,
            'kernel_fingerprint':prov['kernel_fingerprint'],'raw_design':design,
            'max_table_probability_sum_error':float(max_norm),
            'max_probability_reconstruction_error_from_logs':float(max_prob_from_logprob),
            'min_logged_valid_label_mass':minpv,'run_creation_commit':prov['git_commit']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--main',type=Path,required=True);ap.add_argument('--extra',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    jobs=[(args.main,'results/runs/calibration_cheap_talk/cheap_talk_n4_cycle_last1','results/audit/llama-3.1-8b/cheap_talk_d2_enumerated_signal.json',True),
        (args.main,'results/runs/core_cheap_talk/cheap_talk_n10_cycle_full','results/audit/llama-3.1-8b/cheap_talk_d2_signal.json',False),
        (args.extra,'results/extra/runs/olmo_calibration_cheap_talk/cheap_talk_n4_cycle_last1','results/extra/audit/olmo-3-7b/cheap_talk_d2_enumerated_signal.json',True),
        (args.extra,'results/extra/runs/olmo_core_cheap_talk/cheap_talk_n10_cycle_full','results/extra/audit/olmo-3-7b/cheap_talk_d2_signal.json',False)]
    result=[check(*job) for job in jobs]
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=1))

if __name__=='__main__':main()
