"""Automatic reports and bounded GPU probes after the main comparison releases GPUs."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_mechanism_models_replicas_20260925.common import *
from research.v1_state_aware_20260923 import run as infra


def report():
    with (OUT/'analysis/report.log').open('a') as log:
        result=subprocess.run([sys.executable,'-B',str(HERE/'report.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    assert result.returncode==0,'Report/mechanism audit failed; inspect analysis/report.log'


def pipeline():
    lock=infra.acquire(OUT/'analysis/pipeline.lock')
    deadline=read(OUT/'launch.json')['deadline']
    report()
    while read(OUT/'status.json')['state']=='running' and time.time()<deadline+30:
        write(OUT/'analysis/status.json',dict(state='monitoring_main_gpu',updated=time.time(),pid=os.getpid()))
        time.sleep(30);report()
    if time.time()>=deadline:
        write(OUT/'analysis/status.json',dict(state='finished_main_deadline_no_probe_budget',updated=time.time()));return
    validation=read(OUT/'probes/cpu_validation.json')
    assert validation['passed']
    for relative,sha in validation['source_hashes'].items():assert digest(ROOT/relative)==sha
    pending=read(OUT/'probes/design.json')['tasks'];active={};complete=[];failed=[]
    manifest=read(OUT/'manifest.json');gpus=[6,7]
    try:
        while (pending or active) and time.time()<deadline:
            used={v['gpu'] for v in active.values()}
            hardware=infra.hardware()
            for gpu in gpus:
                if not pending:break
                if gpu in used or not infra.idle(hardware[gpu]):continue
                task=pending.pop(0);key=f"{task['model']}_c{task['capacity']}"
                folder=OUT/'probes'/key;folder.mkdir(parents=True,exist_ok=True)
                log=(folder/'worker.log').open('a')
                process=subprocess.Popen([sys.executable,'-B','-u',str(HERE/'probes.py'),'--model',task['model'],
                    '--capacity',str(task['capacity']),'--gpu',str(gpu)],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[key]=dict(gpu=gpu,proc=process,log=log)
            for key,entry in list(active.items()):
                process=entry['proc'];outsiders=[]
                for pid in infra.compute_pids(manifest['hardware'][str(entry['gpu'])]['uuid']):
                    try:own=os.getpgid(pid)==process.pid
                    except ProcessLookupError:continue
                    if not own:outsiders.append(pid)
                if outsiders and process.poll() is None:
                    write(OUT/'probes'/key/'contamination.json',dict(pids=outsiders,updated=time.time()))
                    os.killpg(process.pid,signal.SIGTERM)
                if process.poll() is not None:
                    entry['log'].close();folder=OUT/'probes'/key
                    okay=process.returncode==0 and (folder/'status.json').exists() and read(folder/'status.json')['state']=='complete' and not (folder/'contamination.json').exists()
                    (complete if okay else failed).append(key);active.pop(key)
                    if not okay:write(folder/'controller_failure.json',dict(returncode=process.returncode,updated=time.time()))
                    report()
            write(OUT/'analysis/status.json',dict(state='running_same_state_probes',updated=time.time(),pid=os.getpid(),
                active={k:dict(gpu=v['gpu'],pid=v['proc'].pid) for k,v in active.items()},pending=pending,complete=complete,failed=failed,deadline=deadline))
            time.sleep(2)
    finally:
        for key,entry in active.items():
            p=entry['proc']
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
            entry['log'].close();write(OUT/'probes'/key/'interrupted.json',dict(reason='deadline_or_pipeline_failure',updated=time.time()))
        report()
    main=read(OUT/'status.json')
    write(OUT/'analysis/status.json',dict(state='complete' if main['state']=='complete' and len(complete)==6 else 'finished_with_gaps',
        main_state=main['state'],probe_complete=complete,probe_failed=failed,probe_pending=pending,updated=time.time()))
    lock.close()


def launch():
    assert not (OUT/'analysis/launch.json').exists()
    (OUT/'analysis').mkdir(exist_ok=True)
    with (OUT/'analysis/pipeline.log').open('a') as log:
        p=subprocess.Popen([sys.executable,'-B','-u',str(HERE/'pipeline.py')],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(OUT/'analysis/launch.json',dict(pid=p.pid,detached=True,created=time.time()))
    print('Detached analysis/probe pipeline',p.pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');args=p.parse_args()
    if args.launch:launch()
    else:
        try:pipeline()
        except BaseException:
            write(OUT/'analysis/status.json',dict(state='failed',error=traceback.format_exc(),updated=time.time()));raise
