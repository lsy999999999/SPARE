"""Detached server-owned pipeline; terminal/SSH disconnect does not own child lifetime."""
import argparse,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_arrival_kv_papers_20260925.common import *
from research.v1_state_aware_20260923.run import acquire
GPU_PYTHON='.venv/bin/python'

def main():
 lock=acquire(OUT/'pipeline.lock')
 write(OUT/'analysis/status.json',dict(state='running_gpu_and_simulator',pid=os.getpid(),updated=time.time()))
 with (OUT/'controller.log').open('a') as log:
  result=subprocess.run([GPU_PYTHON,'-B','-u',str(HERE/'run.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 write(OUT/'analysis/status.json',dict(state='auditing',controller_exit=result.returncode,pid=os.getpid(),updated=time.time()))
 with (OUT/'analysis/report.log').open('a') as log:
  audit=subprocess.run([sys.executable,'-B','-u',str(HERE/'report.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 state=read(OUT/'status.json')['state'];sim=read(OUT/'simulator_process.json') if (OUT/'simulator_process.json').exists() else {}
 final='complete' if result.returncode==0 and audit.returncode==0 and state=='complete' and sim.get('complete') and sim.get('returncode')==0 else 'finished_with_gaps' if audit.returncode==0 else 'audit_failed'
 write(OUT/'analysis/status.json',dict(state=final,controller_exit=result.returncode,audit_exit=audit.returncode,gpu_state=state,simulator=sim,updated=time.time()))
 lock.close()

def launch():
 (OUT/'analysis').mkdir(exist_ok=True,parents=True)
 lock=acquire(OUT/'launch.lock')
 assert not (OUT/'launch.json').exists(),'Already launched; do not duplicate'
 v=read(OUT/'cpu_validation.json');d=read(OUT/'design.json')
 assert v['passed'] and v['sources']==source_hashes() and v['design_sha256']==digest(OUT/'design.json')
 start=time.time();deadline=start+d['deadline_hours']*3600
 write(OUT/'launch.json',dict(time=start,deadline=deadline,detached=True,pid=None))
 with (OUT/'analysis/pipeline.log').open('a') as log:
  child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve())],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 write(OUT/'launch.json',dict(pid=child.pid,time=start,deadline=deadline,detached=True))
 print('Detached pipeline',child.pid,'GPU deadline',time.strftime('%Y-%m-%d %H:%M:%S %z',time.localtime(deadline)),flush=True)
 lock.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
 if a.launch:launch()
 else:
  try:main()
  except Exception:write(OUT/'analysis/status.json',dict(state='failed',error=traceback.format_exc(),updated=time.time()));raise
