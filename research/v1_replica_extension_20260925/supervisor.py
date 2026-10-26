"""Audited takeover of cached-deadline controllers; finite queue without wall cutoff.

GPU groups run in their own sessions. Only the two verified Python controllers
are stopped/replaced; their GPU groups are adopted, never restarted or overwritten.
All subsequent GPU groups are serialized to avoid CPU-affinity overlap across series.
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
from research.v1_replica_extension_20260925.common import *
from research.v1_state_aware_20260923 import run as infra

OLD=ROOT/'artifacts/v1_mechanism_models_replicas_20260925_r1'
OLD_CODE=ROOT/'research/v1_mechanism_models_replicas_20260925'


def process_info(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':return None
        return dict(start=fields[19],state=fields[0],
                    command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode())
    except (FileNotFoundError,ProcessLookupError):return None


def running(pid,start):
    info=process_info(pid)
    return info is not None and info['start']==start


def acquire_after_exit(path):
    # SIGKILL delivery and release of process-held flock are asynchronous.
    for attempt in range(300):
        try:return infra.acquire(path)
        except BlockingIOError:
            if attempt==299:raise
            time.sleep(.1)


def takeover():
    assert not (OUT/'takeover.json').exists(),'Takeover is one-shot; preserve evidence'
    controllers=[(read(OLD/'launch.json')['pid'],'run.py --controller'),
                 (read(OLD/'analysis/launch.json')['pid'],'pipeline.py')]
    verified=[]
    for pid,suffix in controllers:
        info=process_info(pid)
        assert info and str(OLD_CODE/suffix) in info['command'],(pid,info)
        assert Path(f'/proc/{pid}').stat().st_uid==os.getuid()
        verified.append(dict(pid=pid,**info))
    stopped=[]
    try:
        for item in verified:
            os.kill(item['pid'],signal.SIGSTOP);stopped.append(item)
        for item in stopped:
            for _ in range(100):
                info=process_info(item['pid'])
                if info and info['state']=='T':break
                time.sleep(.01)
            assert info and info['state']=='T'
        status=read(OLD/'status.json');analysis=read(OLD/'analysis/status.json')
        assert analysis['state']=='monitoring_main_gpu','Do not orphan running probe children'
        adopted=[]
        # Read actual processes as well as status to cover a controller launch/status race.
        tasks={t['key']:t for t in read(OLD/'design.json')['tasks']}
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            info=process_info(int(proc.name))
            if not info:continue
            words=info['command'].split()
            if str(OLD_CODE/'run.py') not in words or '--group' not in words:continue
            key=words[words.index('--group')+1]
            gpus=list(map(int,words[words.index('--gpus')+1].split(',')))
            assert key in tasks and os.getpgid(int(proc.name))==int(proc.name)
            adopted.append(dict(kind='original',key=key,gpus=gpus,pid=int(proc.name),start=info['start']))
        for key,item in status['active'].items():
            info=process_info(item['pid'])
            if info:assert any(a['pid']==item['pid'] for a in adopted)
        record=dict(created=time.time(),reason='User removes total deadline and authorizes 4 replicas.',
            old_launch=read(OLD/'launch.json'),old_status=status,old_analysis_status=analysis,
            controllers=verified,adopted=adopted,deadline=None)
        write(OUT/'takeover.json',record)
        write(OLD/'deadline_amendment.json',dict(created=time.time(),deadline=None,
            supervisor_pid=os.getpid(),record=str(OUT/'takeover.json'),original_launch_preserved=True))
        # SIGKILL targets ONLY controller PIDs, not sessions/groups; prevents their
        # finally blocks from terminating the independently running GPU groups.
        for item in stopped:os.kill(item['pid'],signal.SIGKILL)
        return adopted
    except BaseException:
        for item in stopped:
            if running(item['pid'],item['start']):os.kill(item['pid'],signal.SIGCONT)
        raise


def folder(entry):
    root=OUT if entry['kind']=='extension' else OLD
    return root/('probes' if entry['kind']=='probe' else 'groups')/entry['key']


def status_series(root,active):
    design=read(root/'design.json');complete=[];failed=[];pending=[]
    kind='extension' if root==OUT else 'original'
    for task in design['tasks']:
        f=root/'groups'/task['key'];s=read(f/'status.json') if (f/'status.json').exists() else {}
        bad=any((f/n).exists() for n in ('contamination.json','interrupted.json','controller_failure.json'))
        if s.get('state')=='complete' and not bad:complete.append(task['key'])
        elif s.get('state')=='failed' or bad:failed.append(task['key'])
        elif not any(a['kind']==kind and a['key']==task['key'] for a in active):pending.append(task['key'])
    current={a['key']:{k:a[k] for k in ('gpus','pid')} for a in active if a['kind']==kind}
    state='running' if pending or current else 'finished_with_gaps' if failed else 'complete'
    write(root/'status.json',dict(state=state,updated=time.time(),deadline=None,supervisor_pid=os.getpid(),
        pending=pending,active=current,complete=complete,failed=failed,planned_groups=len(design['tasks'])))


def report(root):
    code=HERE if root==OUT else OLD_CODE
    (root/'analysis').mkdir(exist_ok=True)
    with (root/'analysis/report.log').open('a') as log:
        result=subprocess.run([sys.executable,'-B',str(code/'report.py')],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,timeout=300)
    assert result.returncode==0,f'Report failed: {root}'


def run():
    lock=infra.acquire(OUT/'supervisor.lock')
    from research.v1_replica_extension_20260925.run import initialize
    initialize()  # Verify frozen source hashes before touching current controllers.
    from research.v1_mechanism_models_replicas_20260925.common import sources as original_sources
    assert original_sources()==read(OLD/'manifest.json')['sources']
    if (OUT/'takeover.json').exists():
        assert not (OUT/'queue.json').exists(),'This path only recovers pre-queue setup failures'
        record=read(OUT/'takeover.json')
        assert all(not running(c['pid'],c['start']) for c in record['controllers'])
        active=[a for a in record['adopted'] if running(a['pid'],a['start'])]
        write(OUT/'setup_recovery.json',dict(created=time.time(),pid=os.getpid(),adopted=active))
        amendment=read(OLD/'deadline_amendment.json')
        amendment.update(supervisor_pid=os.getpid(),updated=time.time())
        write(OLD/'deadline_amendment.json',amendment)
    else:
        active=takeover()
    old_lock=acquire_after_exit(OLD/'controller.lock')
    pipeline_lock=acquire_after_exit(OLD/'analysis/pipeline.lock')
    old_design=read(OLD/'design.json');extension=read(OUT/'design.json')
    probe_validation=read(OLD/'probes/cpu_validation.json')
    assert probe_validation['passed']
    for name,sha in probe_validation['source_hashes'].items():assert digest(ROOT/name)==sha
    original=[dict(kind='original',key=t['key'],replicas=t['replicas']) for t in old_design['tasks']
              if not (OLD/'groups'/t['key']/'status.json').exists()
              and not any(a['key']==t['key'] for a in active)]
    extended=[dict(kind='extension',key=t['key'],replicas=t['replicas']) for t in extension['tasks']]
    probes=[dict(kind='probe',key=f"{t['model']}_c{t['capacity']}",replicas=1,**t)
            for t in read(OLD/'probes/design.json')['tasks']]
    # Prioritize the requested two four-replica groups, then finish original
    # comparisons/mechanism probes, then the remaining extension controls.
    pending=extended[:2]+original+probes+extended[2:]
    write(OUT/'queue.json',dict(created=time.time(),queue=pending,adopted=active,deadline=None))
    complete=[];failed=[];handles={};last_report=0
    while pending or active:
        assert shutil.disk_usage(OUT).free>15*1024**3,'Disk reserve below 15 GiB'
        for entry in list(active):
            proc=handles.get(entry['pid'])
            alive=proc.poll() is None if proc is not None else running(entry['pid'],entry['start'])
            f=folder(entry)
            if alive:
                manifest=read((OUT if entry['kind']=='extension' else OLD)/'manifest.json')
                outsiders=[]
                for gpu in entry['gpus']:
                    for pid in infra.compute_pids(manifest['hardware'][str(gpu)]['uuid']):
                        try:own=os.getpgid(pid)==entry['pid']
                        except ProcessLookupError:continue
                        if not own:outsiders.append(pid)
                if outsiders:
                    write(f/'contamination.json',dict(pids=outsiders,updated=time.time()))
                    os.killpg(entry['pid'],signal.SIGTERM)
                # Local watchdog bounds hung groups; it is NOT an overall deadline.
                if time.time()>entry.get('watchdog',float('inf')):
                    write(f/'interrupted.json',dict(reason='group_watchdog',updated=time.time()))
                    os.killpg(entry['pid'],signal.SIGKILL)
                continue
            s=read(f/'status.json') if (f/'status.json').exists() else {}
            okay=s.get('state')=='complete' and not any((f/n).exists() for n in ('contamination.json','interrupted.json'))
            if proc is not None:okay=okay and proc.returncode==0
            (complete if okay else failed).append(dict(kind=entry['kind'],key=entry['key']))
            if not okay:write(f/'controller_failure.json',dict(returncode=proc.returncode if proc else None,updated=time.time()))
            active.remove(entry)
            report(OUT if entry['kind']=='extension' else OLD)
        if not active and pending:
            entry=pending[0];preference=[1,2,6,7] if entry['kind']=='extension' else [6,7]
            hardware=infra.hardware();available=[g for g in preference if infra.idle(hardware[g])]
            if len(available)>=entry['replicas']:
                pending.pop(0);entry=dict(entry,gpus=available[:entry['replicas']]);f=folder(entry)
                assert not (f/'status.json').exists(),'Never overwrite measured group'
                f.mkdir(parents=True,exist_ok=True)
                if entry['kind']=='probe':
                    command=[str(OLD_CODE/'probes.py'),'--model',entry['model'],'--capacity',str(entry['capacity']),'--gpu',str(entry['gpus'][0])]
                    watchdog=7200
                else:
                    code=HERE if entry['kind']=='extension' else OLD_CODE
                    command=[str(code/'run.py'),'--group',entry['key'],'--gpus',','.join(map(str,entry['gpus']))]
                    watchdog=24000
                with (f/'worker.log').open('a') as log:
                    proc=subprocess.Popen([sys.executable,'-B','-u',*command],cwd=ROOT,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                entry.update(pid=proc.pid,start=process_info(proc.pid)['start'],watchdog=time.time()+watchdog)
                handles[proc.pid]=proc;active.append(entry)
        for root in (OLD,OUT):status_series(root,active)
        state=dict(state='running',pid=os.getpid(),updated=time.time(),deadline=None,
                   active=active,pending=pending,complete=complete,failed=failed)
        write(OUT/'supervisor_status.json',state)
        write(OLD/'analysis/status.json',dict(state='supervised_without_deadline',supervisor_pid=os.getpid(),
            active=[a for a in active if a['kind']=='probe'],pending=[a for a in pending if a['kind']=='probe'],
            complete=[a for a in complete if a['kind']=='probe'],failed=[a for a in failed if a['kind']=='probe'],updated=time.time()))
        if time.time()-last_report>180:
            for root in (OLD,OUT):report(root)
            last_report=time.time()
        time.sleep(3)
    for root in (OLD,OUT):report(root)
    # Include failures that occurred before takeover in the terminal summary.
    prior_failed=read(OLD/'status.json')['failed']
    state.update(state='finished_with_gaps' if failed or prior_failed else 'complete',updated=time.time(),
                 prior_original_failures=prior_failed)
    write(OUT/'supervisor_status.json',state)
    write(OLD/'analysis/status.json',dict(state=state['state'],updated=time.time(),
        probe_complete=[a for a in complete if a['kind']=='probe'],probe_failed=[a for a in failed if a['kind']=='probe']))
    lock.close();old_lock.close();pipeline_lock.close()


def launch():
    assert not (OUT/'launch.json').exists()
    with (OUT/'supervisor.log').open('a') as log:
        p=subprocess.Popen([sys.executable,'-B','-u',str(HERE/'supervisor.py'),'--run'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(OUT/'launch.json',dict(pid=p.pid,created=time.time(),deadline=None,detached=True))
    print('Detached no-total-deadline supervisor',p.pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--run',action='store_true');args=p.parse_args()
    if args.launch:launch()
    elif args.run:
        try:run()
        except BaseException:
            write(OUT/'supervisor_status.json',dict(state='failed',updated=time.time(),pid=os.getpid(),error=traceback.format_exc()))
            raise
    else:p.error('Select --launch or --run')
