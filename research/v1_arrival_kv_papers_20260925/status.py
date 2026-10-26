"""Read-only progress; standard library only, run with watch from any SSH session."""
import os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_arrival_kv_papers_20260925.common import *

def load(name):
 p=OUT/name
 return read(p) if p.exists() else {}

def main():
 d=load('design.json');launch=load('launch.json');status=load('status.json');pipeline=load('analysis/status.json');sim=load('simulator/status.json')
 print(time.strftime('%Y-%m-%d %H:%M:%S %z'),'旧式+A / KV / paper-rule comparison')
 pid=launch.get('pid');alive=False
 if pid:
  try:os.kill(pid,0);alive=True
  except ProcessLookupError:pass
 print('Pipeline:',pipeline.get('state','not launched'),'PID:',pid,'alive:',alive)
 print('GPU controller:',status.get('state','not launched'))
 groups=[read(p) for p in (OUT/'groups').glob('*.json')];done=[g for g in groups if g['state']=='complete'];failed=[g for g in groups if g['state']!='complete']
 nr=len(done)*len(d.get('methods',[]))*d.get('repeats',0)
 print(f"Complete groups: {len(done)}/{len(d.get('cells',[]))}; accepted formal runs: {nr}/{d.get('expected_formal_runs',0)}; failed groups: {len(failed)}")
 print('Retries:',status.get('counts',{}).get('retried',0),'disabled GPU/capacity:',status.get('disabled_gpu_capacity',[]))
 for p in sorted(OUT.glob('worker_g*.json')):
  w=read(p);print(p.stem,w.get('state'),w.get('job',''),f"updated {time.time()-w.get('updated',time.time()):.0f}s ago")
 print('Simulator:',sim.get('state','not started'),f"{sim.get('completed',0)}/{sim.get('expected',len(d.get('cells',[]))*len(d.get('methods',[])))}")
 ap=load('analysis/audit_progress.json')
 if ap:print('Independent final audit:',ap.get('groups'),'groups',ap.get('rounds'),'rounds')
 if launch:
  elapsed=time.time()-launch['time'];left=launch['deadline']-time.time()
  print(f'Elapsed: {elapsed/3600:.2f} h; GPU deadline remaining: {left/3600:.2f} h')
  if done and elapsed>600 and len(done)<len(d['cells']):print(f"Rough ETA at observed throughput: {(len(d['cells'])-len(done))*elapsed/len(done)/3600:.1f} h (depends on free GPUs)")
  print('GPU deadline:',time.strftime('%Y-%m-%d %H:%M:%S %z',time.localtime(launch['deadline'])))
 if failed:print('Latest failure:',failed[-1]['group'],failed[-1].get('error'))
 print('Results:',HERE/'ANALYSIS_CN.md')
if __name__=='__main__':main()
