"""Offline reproduction of a missing intervention binding in resume provenance.

This is the one review script that intentionally imports the project runner under
test. It uses a scripted CPU backend and writes only to a temporary directory.
It never uses the LLM backend, a server, credentials, or existing run evidence.
"""
import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    sys.path.insert(0,str(args.source))
    from swarm_audit.policies import ScriptedBackend
    from swarm_audit.swarm import Condition,run_condition,read_run
    from swarm_audit.followup import record_observer
    c=Condition(arm='review_provenance_probe',arm_id=9001,condition_id=4,channel='cheap_talk',n=4,last_k=1,runs=1)
    backend=ScriptedBackend('hidden_signal')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='review-only-provenance-',dir=args.out.parent) as tmp:
        root=Path(tmp)
        folder=run_condition(c,backend,root,observe=record_observer('unmodified'),extra_manifest={'observe':'unmodified'},log=lambda _:None)
        first=json.loads((folder/'provenance.json').read_text())
        accepted=True;error=None
        try:
            run_condition(replace(c,runs=2),backend,root,observe=record_observer('both_null'),extra_manifest={'observe':'both_null'},log=lambda _:None)
        except Exception as exc:
            accepted=False;error=f'{type(exc).__name__}: {exc}'
        second=json.loads((folder/'provenance.json').read_text())
        manifest=json.loads((folder.parent/'manifest.json').read_text())['conditions'][folder.name]
        result={'source_commit':'9aa31705b9ae5ee45f27a87feeb1a2e2415f318f',
            'backend':'scripted:hidden_signal; offline CPU only','changed_observation_on_resume_accepted':accepted,
            'fingerprint_before':first['fingerprint'],'fingerprint_after':second['fingerprint'],
            'fingerprint_unchanged':first['fingerprint']==second['fingerprint'],
            'immutable_design_has_observe':'observe' in first['payload']['design'],
            'session_observe_names':[s.get('observe') for s in manifest['sessions']],
            'runs_logged':manifest['summary']['runs_logged'],'rejection':error,
            'actual_intervention_logs_checked_in_this_review':'all four real E2 cells have a single correct treatment; no real mixed evidence found',
            'scope':'prospective resume safeguard gap; does not invalidate the checked completed experiments'}
        if accepted:
            r0=read_run(folder/'run_000.jsonl.gz');r1=read_run(folder/'run_001.jsonl.gz')
            result['first_run_observations_equal_actual']=all(r['observed_record']==r['record'] for r in r0)
            result['second_run_all_observed_message_fields_none']=all(r['observed_record']['signal']=='none' and r['observed_record']['sig_none']==2 for r in r1)
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=1))

if __name__=='__main__':main()
