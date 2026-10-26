"""Stop only this experiment, preserving completed groups and triggering its final audit."""
import json,os,signal,time
from pathlib import Path
ROOT=Path('.')
OUT=ROOT/'artifacts/v1_arrival_kv_papers_20260925_r1'
RUN=str(ROOT/'research/v1_arrival_kv_papers_20260925/run.py')
def read(p):return json.loads(p.read_text())
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.stop.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def info(pid):
 try:
  path=Path('/proc')/str(pid);fields=(path/'stat').read_text().split(') ',1)[1].split()
  return dict(pid=pid,ppid=int(fields[1]),state=fields[0],start=fields[19],args=(path/'cmdline').read_bytes().decode().split('\0'))
 except (FileNotFoundError,ProcessLookupError):return None
def processes():return [r for p in Path('/proc').iterdir() if p.name.isdigit() for r in [info(int(p.name))] if r]
def same(r):
 now=info(r['pid']);return now and now['start']==r['start'] and now['args']==r['args'] and now['state']!='Z'
def sig(r,s,group=False):
 if same(r):
  if group:
   assert os.getpgid(r['pid'])==r['pid'];os.killpg(r['pid'],s)
  else:os.kill(r['pid'],s)
def main():
 launch=read(OUT/'launch.json');pid=launch['pid'];pipeline=info(pid)
 assert pipeline and str(ROOT/'research/v1_arrival_kv_papers_20260925/pipeline.py') in pipeline['args']
 controllers=[r for r in processes() if r['ppid']==pid and RUN in r['args'] and '--worker' not in r['args']]
 if not controllers:print('Controller already exited; pipeline may be auditing.');return
 assert len(controllers)==1;controller=controllers[0];now=time.time()
 assert not (OUT/'user_stop.json').exists(),'Stop already recorded'
 write(OUT/'user_stop.json',dict(requested_at=now,reason='User requested smaller per-cell scope and a basic result now; stop further GPU execution.',previous_launch=launch,controller=controller,pipeline=pid,state='stopping'))
 sig(controller,signal.SIGSTOP)
 workers=[r for r in processes() if RUN in r['args'] and '--worker' in r['args'] and r['ppid']==controller['pid']]
 for r in workers:sig(r,signal.SIGTERM,True)
 end=time.monotonic()+15
 while any(same(r) for r in workers) and time.monotonic()<end:time.sleep(.2)
 for r in workers:sig(r,signal.SIGKILL,True)
 # Controller is frozen: completed groups cannot race with cutoff bookkeeping.
 m=read(OUT/'manifest.json');interrupted=[]
 for r in workers:
  taskpath=Path(r['args'][r['args'].index('--worker')+1]);task=read(taskpath);g=task['group']
  assert OUT in taskpath.parents
  p=OUT/'groups'/f'{g}.json'
  if p.exists():continue
  folder=taskpath.parent
  entry=dict(group=g,gpu=int(r['args'][r['args'].index('--gpu')+1]),attempt=task['attempt'],folder=task['folder'],cell=task['cell'],
   state='stopped_by_user',error='user_requested_smaller_basic_result',started=taskpath.stat().st_mtime,ended=time.time())
  write(folder/'attempt_status.json',entry);write(p,entry);interrupted.append(g)
 groups=[read(p) for p in (OUT/'groups').glob('*.json')];complete=[g for g in groups if g['state']=='complete']
 prev=read(OUT/'status.json');finished={g['group'] for g in groups}
 write(OUT/'status.json',dict(state='stopped_by_user',updated=time.time(),deadline=now,planned_groups=len(m['tasks']),
  counts=dict(complete=len(complete),failed=sum(g['state']=='failed' for g in groups),stopped=len(interrupted),retried=prev.get('counts',{}).get('retried',0)),
  unfinished_groups=[t['group'] for t in m['tasks'] if t['group'] not in finished],user_stop=True))
 sim=OUT/'simulator/full_complete.json';assert sim.exists() and read(sim)['passed']
 write(OUT/'simulator_process.json',dict(returncode=0,complete=True,note='Completed before user-requested stop.'))
 stop=read(OUT/'user_stop.json');stop.update(state='gpu_stopped',stopped_at=time.time(),complete_groups=len(complete),interrupted_groups=interrupted,workers=[r['pid'] for r in workers]);write(OUT/'user_stop.json',stop)
 # Pending TERM is delivered when CONT wakes the frozen controller. Its parent
 # then runs the existing independent report, with the cutoff already on disk.
 sig(controller,signal.SIGTERM);sig(controller,signal.SIGCONT)
 time.sleep(1)
 assert not same(controller),'Controller did not stop'
 remaining=[r for r in processes() if RUN in r['args'] and '--worker' in r['args'] and r['state']!='Z']
 assert not remaining,remaining
 print(json.dumps(dict(stopped=True,complete_groups=len(complete),interrupted_groups=interrupted,pipeline_pid=pid,automatic_audit=True),ensure_ascii=False))
if __name__=='__main__':main()
