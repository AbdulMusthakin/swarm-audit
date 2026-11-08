"""Inventory fixed snapshots, source hashes and completeness, without running jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',action='append',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    result=[]
    for root in args.root:
        commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        entry={'root':str(root),'snapshot_commit':commit,'conditions':[],'source_hashes':{}}
        for file in sorted((root/'results').rglob('manifest.json')):
            d=json.loads(file.read_text())
            for name,c in d.get('conditions',{}).items():
                folder=file.parent/name;logs=sorted(folder.glob('run_*.jsonl.gz'));ids=[int(p.name.split('_')[1].split('.')[0]) for p in logs]
                entry['conditions'].append({'manifest':str(file.relative_to(root)),'condition':name,'log_count':len(logs),
                    'run_min':min(ids) if ids else None,'run_max':max(ids) if ids else None,
                    'missing_within_range':sorted(set(range(max(ids)+1))-set(ids)) if ids else [],
                    'summary':c.get('summary'), 'kernel_fingerprint':c.get('provenance',{}).get('kernel_fingerprint')})
        for pattern in ['results/audit/**/*.json','results/extra/audit/**/*.json','results/gates/*.json','results/extra/followup/*.json','configs/*.yaml','swarm_audit/prompts/*.txt']:
            for file in sorted(root.glob(pattern)):
                entry['source_hashes'][str(file.relative_to(root))]=hashlib.sha256(file.read_bytes()).hexdigest()
        entry['provenance_without_logs']=[str(f.parent.relative_to(root)) for f in root.rglob('provenance.json') if not list(f.parent.glob('run_*.jsonl.gz'))]
        result.append(entry)
    args.out.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
