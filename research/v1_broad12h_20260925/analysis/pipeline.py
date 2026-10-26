"""Independent session, 12h GPU deadline, automatic reporting even on incomplete matrix."""
import argparse,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from research.v1_broad12h_20260925.common import *
GPU_PYTHON='.venv/bin/python'

def main():
    from research.v1_state_aware_20260923.run import acquire
    lock=acquire(OUT/'pipeline.lock')
    write(OUT/'analysis/status.json',dict(state='running_gpu_and_simulator',pid=os.getpid(),updated=time.time()))
    with (OUT/'controller.log').open('a') as log:
        result=subprocess.run([GPU_PYTHON,'-B','-u',str(HERE/'run.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    write(OUT/'analysis/status.json',dict(state='auditing',controller_exit=result.returncode,updated=time.time()))
    with (OUT/'analysis/report.log').open('a') as log:
        audit=subprocess.run([sys.executable,'-B','-u',str(HERE/'analysis/report.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    state=read(OUT/'status.json')['state'];sim=read(OUT/'simulator_process.json') if (OUT/'simulator_process.json').exists() else {}
    final='complete' if result.returncode==0 and audit.returncode==0 and state=='complete' and sim.get('complete') else 'finished_with_gaps' if audit.returncode==0 else 'audit_failed'
    write(OUT/'analysis/status.json',dict(state=final,controller_exit=result.returncode,audit_exit=audit.returncode,gpu_state=state,simulator=sim,updated=time.time()))
    lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
    (OUT/'analysis').mkdir(exist_ok=True,parents=True)
    if a.launch:
        from research.v1_state_aware_20260923.run import acquire
        lock=acquire(OUT/'launch.lock')
        assert not (OUT/'launch.json').exists(),'Already launched; do not duplicate'
        v=read(OUT/'cpu_validation.json')
        assert v['passed'] and v['sources']==source_hashes() and v['design_sha256']==digest(OUT/'design.json')
        assert v['theory_checks_sha256']==digest(OUT/'theory_checks.json')
        start=time.time();write(OUT/'launch.json',dict(time=start,deadline=start+12*3600,detached=True,pid=None))
        with (OUT/'analysis/pipeline.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(OUT/'launch.json',dict(pid=child.pid,time=start,deadline=start+12*3600,detached=True))
        print('Detached 12h pipeline',child.pid,'deadline',time.strftime('%Y-%m-%d %H:%M:%S %z',time.localtime(start+12*3600)))
        lock.close()
    else:
        try:main()
        except Exception:write(OUT/'analysis/status.json',dict(state='failed',error=traceback.format_exc(),updated=time.time()));raise
