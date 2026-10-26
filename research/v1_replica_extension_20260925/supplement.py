"""One prespecified replacement attempt for externally interrupted four-replica groups.

Frozen runner/policies remain unchanged. This entrypoint is also imported by
spawned workers, so both runner globals and shared input helpers use the new OUT.
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_replica_extension_20260925 import common as c,run as runner
from research.v1_state_aware_20260923 import run as infra

PARENT=c.OUT
OUT=c.ROOT/'artifacts/v1_replica_extension_20260926_supplement_r1'
c.OUT=OUT
runner.OUT=OUT
KEYS=['qwen7b_c768_r4','smol360m_c768_r4']
GPUS=[1,2,6,7]


def prepare():
    assert not OUT.exists(),'New namespace required; never overwrite measurements'
    design=c.read(PARENT/'design.json');manifest=c.read(PARENT/'manifest.json')
    assert c.sources()==manifest['sources']
    evidence={}
    for key in KEYS:
        f=PARENT/'groups'/key
        assert c.read(f/'calibration.json')['passed']
        assert c.read(f/'controller_failure.json')['returncode']==-15
        evidence[key]=c.read(f/'contamination.json')
    design.update(tasks=[t for t in design['tasks'] if t['key'] in KEYS],created=time.time(),
        planned_group_tasks=2,planned_global_episodes=96,planned_replica_episodes=384,
        amendment='One replacement attempt for BOTH externally interrupted 768-block four-replica groups; same frozen protocol, no performance-based selection.',
        parent_results=str(PARENT),interruption_evidence=evidence)
    OUT.mkdir(parents=True)
    shutil.copytree(PARENT/'inputs',OUT/'inputs')
    shutil.copyfile(PARENT/'histories.json',OUT/'histories.json')
    shutil.copyfile(PARENT/'cpu_validation.json',OUT/'cpu_validation.json')
    c.write(OUT/'design.json',design)
    manifest.update(design_sha256=c.digest(OUT/'design.json'),created=time.time(),
        supplemental_entrypoint_sha256=c.digest(__file__))
    c.write(OUT/'manifest.json',manifest)
    shutil.copytree(PARENT/'source',OUT/'source')
    shutil.copyfile(__file__,OUT/'source/supplement.py')
    c.write(OUT/'status.json',dict(state='prepared',pending=KEYS,complete=[],failed=[]))


def report():
    from research.v1_replica_extension_20260925 import report as reporter
    reporter.OUT=OUT
    reporter.main()


def controller():
    lock=infra.acquire(OUT/'controller.lock');complete=[];failed=[]
    manifest=c.read(OUT/'manifest.json')
    assert c.digest(__file__)==manifest['supplemental_entrypoint_sha256']
    for index,key in enumerate(KEYS):
        while True:
            c.write(OUT/'status.json',dict(state='waiting_for_four_idle_gpus',pid=os.getpid(),updated=time.time(),
                deadline=None,pending=KEYS[index:],complete=complete,failed=failed))
            hardware=infra.hardware()
            if all(infra.idle(hardware[g]) for g in GPUS):break
            time.sleep(15)
        assert shutil.disk_usage(OUT).free>15*1024**3
        f=OUT/'groups'/key;f.mkdir(parents=True,exist_ok=True)
        with (f/'worker.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-B','-u',__file__,'--group',key],cwd=c.ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        start=time.time()
        while p.poll() is None:
            c.write(OUT/'status.json',dict(state='running',pid=os.getpid(),updated=time.time(),deadline=None,
                active=dict(key=key,pid=p.pid,gpus=GPUS),pending=KEYS[index+1:],complete=complete,failed=failed))
            outsiders=[]
            for gpu in GPUS:
                for pid in infra.compute_pids(manifest['hardware'][str(gpu)]['uuid']):
                    try:own=os.getpgid(pid)==p.pid
                    except ProcessLookupError:continue
                    if not own:outsiders.append(pid)
            if outsiders or time.time()-start>24000:
                c.write(f/('contamination.json' if outsiders else 'interrupted.json'),
                    dict(pids=outsiders,reason='external_gpu_process' if outsiders else 'group_watchdog',updated=time.time()))
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
            else:time.sleep(3)
        okay=p.returncode==0 and (f/'status.json').exists() and c.read(f/'status.json')['state']=='complete'
        okay=okay and not (f/'contamination.json').exists() and not (f/'interrupted.json').exists()
        (complete if okay else failed).append(key)
        if not okay:c.write(f/'controller_failure.json',dict(returncode=p.returncode,updated=time.time()))
        report()
    c.write(OUT/'status.json',dict(state='complete' if not failed else 'finished_with_gaps',
        updated=time.time(),pending=[],active=None,complete=complete,failed=failed,deadline=None))
    lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true')
    parser.add_argument('--controller',action='store_true');parser.add_argument('--group');args=parser.parse_args()
    if args.group:runner.group(args.group,GPUS)
    elif args.controller:
        try:controller()
        except BaseException:
            c.write(OUT/'status.json',dict(state='failed',error=traceback.format_exc(),updated=time.time()));raise
    elif args.launch:
        prepare()
        with (OUT/'controller.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-B','-u',__file__,'--controller'],cwd=c.ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        c.write(OUT/'launch.json',dict(pid=p.pid,created=time.time(),deadline=None))
        print('Supplement launched',p.pid,flush=True)
    else:parser.error('Select --launch / --controller / --group')
