"""Technical recovery of one G3 group; preserves frozen methods and all attempts."""
import os,sys,json,time,subprocess,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.www_capacity_load_20261007 import run as base
from research.www_capacity_load_20261007.common import *
gpu=1

def recover(cell):
    base.verify_design();manifest=read(OUT/'manifest.json')
    existing=read(OUT/'groups'/f'{cell}.json')
    if existing['state']=='complete':return
    amendment=dict(reason='Original bounded retries exhausted; observed A/A TTFT variation exceeded unchanged 5% gate; extra technical attempts registered before rerunning',additional_attempts=[3,4,5],methods_changed=False,acceptance_changed=False,created=time.time(),pid=os.getpid())
    write(OUT/f'recovery_{cell}_amendment_20261007.json',amendment)
    write(OUT/'recovery_status_20261007.json',dict(state='running',pid=os.getpid(),cell=cell,gpu=gpu))
    for attempt in [3,4,5]:
        while not base.idle(base.hardware()[gpu]):
            if time.time()>read(OUT/'launch.json')['deadline']:raise TimeoutError('original batch deadline')
            time.sleep(15)
        folder=f'attempts/{cell}/a{attempt}';p=OUT/folder;p.mkdir(parents=True,exist_ok=True)
        assert not (p/'worker.log').exists()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',VLLM_USE_V1='1',VLLM_ENABLE_V1_MULTIPROCESSING='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        start=time.time();error=None
        with (p/'worker.log').open('w') as log:
            child=subprocess.Popen([str(PYTHON),'-B','-u',str(HERE/'run.py'),'--worker',cell,'--gpu',str(gpu),'--folder',folder],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                external=base.compute_pids(manifest['hardware'][str(gpu)]['uuid'])-base.descendants(child.pid)
                if external or time.time()-start>12*3600:
                    error='external_gpu_process' if external else 'watchdog'
                    os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=30);break
                time.sleep(3)
        status=read(p/'status.json') if (p/'status.json').exists() else {}
        passed=error is None and child.returncode==0 and status.get('state')=='complete'
        record=dict(cell=cell,gpu=gpu,attempt=attempt,folder=folder,state='complete' if passed else 'failed',error=error or status.get('error'),started=start,ended=time.time())
        write(p/'attempt_status.json',record)
        if passed:
            write(OUT/'groups'/f'{cell}.json',record)
            return record
    write(OUT/'recovery_status_20261007.json',dict(state='failed',cell=cell))

def main():
    import fcntl
    lock=open('/tmp/www-g4-recovery-controller.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    original=read(OUT/'launch.json')['pid']
    write(OUT/'recovery_status_20261007.json',dict(state='waiting_for_original_controller',pid=os.getpid(),original_pid=original))
    while Path(f'/proc/{original}').exists():
        if time.time()>read(OUT/'launch.json')['deadline']:raise TimeoutError('original batch deadline')
        time.sleep(15)
    records=[]
    for cell in read(OUT/'design.json')['cells']:
        key=cell['key'];group=read(OUT/'groups'/f'{key}.json')
        if group['state']!='complete':
            records.append(recover(key))
    groups=[read(p) for p in (OUT/'groups').glob('*.json')]
    complete=all(g['state']=='complete' for g in groups) and len(groups)==132
    if complete:subprocess.run([str(PYTHON),'-B',str(HERE/'analyze.py')],cwd=ROOT,check=True)
    status=read(OUT/'status.json');status.update(state='complete' if complete else 'incomplete',updated=time.time(),finished=groups,active={})
    write(OUT/'status.json',status)
    write(OUT/'recovery_status_20261007.json',dict(state='complete' if complete else 'failed',pid=os.getpid(),records=records))

if __name__=='__main__':main()
