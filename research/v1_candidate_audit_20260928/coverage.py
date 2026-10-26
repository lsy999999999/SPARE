"""Read-only source-log coverage; writes only new diagnostic artifacts."""
import gzip
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_candidate_audit_20260928.common import *
from research.v1_candidate_audit_20260928.candidates import enumerate_actions


def main():
    summaries=[]
    for folder in sorted((OLD/'groups').glob('*')):
        if (folder/'exclusion.json').exists(): continue
        for path in sorted((folder/'runs').glob('*_r0.json')):
            run=read(path)
            if run.get('arm') not in ('current','points_rate'): continue
            rp=OLD/run['rounds_path'];assert digest(rp)==run['rounds_sha256']
            with gzip.open(rp,'rt') as f: rows=json.load(f)
            counts=Counter();unselected=0;examples=[]
            for row in rows:
                c=enumerate_actions(row)
                counts['rounds']+=1
                if not c['actions']:continue
                counts['normal_rounds']+=1
                counts['k_effective']+=c['k_effective'];counts['k_matched']+=c['k_matched']
                counts['with_new_plans']+=c['new_plans']>0
                counts['legal_plans']+=len(c['actions']);counts['new_plans']+=c['new_plans']
                counts['proposals']+=len(c['proposals'])
                counts['legal_proposals']+=sum(p['legal'] for p in c['proposals'])
                unselected+=sum(a['tokens']!=row['scheduled'] for a in c['actions'])
                if c['new_plans'] and c['k_matched'] and len(examples)<3:
                    examples.append(dict(round=row['round'],free=row['free_before'],actions=c['actions']))
            summary=dict(group=folder.name,run=path.stem,source_sha=digest(rp),**counts,
                         unexecuted_legal_plans=unselected,examples=examples)
            summaries.append(summary)
            print(folder.name,path.stem,dict(counts),flush=True)
    write(OUT/'coverage.json',dict(scope='Existing r0 current and points_rate trajectories; descriptive, not action timing labels',runs=summaries))


if __name__=='__main__':main()
