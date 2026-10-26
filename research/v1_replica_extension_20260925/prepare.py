"""Freeze a separate, uniformly warmed 1/2/4 replica series; no tuning."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.v1_replica_extension_20260925.common import *


def main():
    assert not (OUT/'design.json').exists()
    old=ROOT/'artifacts/v1_mechanism_models_replicas_20260925_r1'
    previous=read(old/'design.json')
    for case in previous['cases']:
        assert digest(old/'inputs'/f'{case}.json')==previous['trace_hashes'][case]
        write(OUT/'inputs'/f'{case}.json',read(old/'inputs'/f'{case}.json'))
    write(OUT/'histories.json',read(old/'histories.json'))
    models={k:previous['models'][k] for k in ('qwen7b','smol360m')}
    for model in models.values():
        path=Path(model['path'])
        assert digest(path/'config.json')==model['config_sha256']
        for weight in model['weights']:
            assert digest(path/weight['name'])==weight['sha256']
    tasks=[]
    # Four replicas first, then fresh matched 1/2-replica controls.
    for capacity in (768,4096,2048):
        for replicas in (4,1,2):
            for model in models:
                tasks.append(dict(key=f'{model}_c{capacity}_r{replicas}',model=model,
                    capacity=capacity,replicas=replicas,cases=previous['cases'],
                    loads=['fixed_total'] if replicas==1 else ['fixed_total','per_replica']))
    count=lambda t:len(t['cases'])*len(t['loads'])*len(ARMS)*2
    design=dict(previous,created=time.time(),models=models,tasks=tasks,
        max_hours=None,gpu_preference=[1,2,6,7],replica_counts=[1,2,4],
        planned_group_tasks=len(tasks),planned_global_episodes=sum(map(count,tasks)),
        planned_replica_episodes=sum(count(t)*t['replicas'] for t in tasks),
        parent_design_sha256=digest(old/'design.json'),
        amendment='User explicitly authorizes four independent replicas and removes total finish deadline.',
        calibration='Every extension group gets one full SRP calibration warmup before the unchanged two A/A runs and 5% gate. Failed original group remains excluded; no automatic reruns.',
        comparability='Fresh 1/2/4 groups share the extension warmup protocol. Do not pool with the original series.',
        histories_sha256=digest(OUT/'histories.json'),
        trace_hashes={c:digest(OUT/'inputs'/f'{c}.json') for c in previous['cases']})
    write(OUT/'design.json',design)
    print({k:design[k] for k in ('planned_group_tasks','planned_global_episodes','planned_replica_episodes')},flush=True)


if __name__=='__main__':main()
