"""Audited paired comparisons, cohort-level uncertainty and automatic result document."""
import argparse,csv,gzip,json,math,os,random,statistics as st,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_arrival_kv_papers_20260925.common import *
from research.v1_budget_20260922 import run as ref
from research.v1_broad12h_20260925.analysis import report as previous
ref.OUT=OUT;previous.OUT=OUT

def audit(r):
 if r['job']['method']=='threshold':return previous.audit(r)
 ref.audit_result(r)
 assert math.isclose(st.fmean(c['finish_ms']-c['arrival_ms'] for c in r['clocks'].values()),r['metrics']['mean_e2e_ms'],abs_tol=1e-6)
 with gzip.open(OUT/r['rounds_path'],'rt') as f:rows=json.load(f)
 bins=dict(rounds=0,free_lt_10pct=0,free_gt_90pct=0,prefill_dominant=0,decode_dominant=0,recomputing=0)
 high={};recomputed=0
 for row in rows:
  state={s['rid']:s for s in row['state']};cu=row['total_blocks']-1
  assert row['free_before']+sum(s['blocks'] for s in state.values())==cu
  assert set(row['scheduled'])<=set(state)
  details={s['id']:s for s in row['allocations']};assert set(details)==set(row['scheduled'])
  rec=0
  for rid,s in state.items():
   h=max(high.get(rid,0),s['computed']);a=details.get(rid)
   if a:
    assert a['tokens']==row['scheduled'][rid]
    assert a['blocks_after']>=math.ceil((a['before']+a['tokens'])/16)
    rec+=min(a['tokens'],max(0,h-a['before']));high[rid]=max(h,a['before']+a['tokens'])
   else:high[rid]=h
  assert rec==row['recomputed_tokens'];recomputed+=rec
  if row['verified']:
   growth=sum(max(0,(state[rid]['computed']+n+15)//16-state[rid]['blocks']) for rid,n in row['scheduled'].items())
   assert growth==row['free_before']-row['free_after'] and row['preemptions']==0
  bins['rounds']+=1;bins['free_lt_10pct']+=row['free_before']/cu<.1;bins['free_gt_90pct']+=row['free_before']/cu>.9
  bins['prefill_dominant']+=sum(s['blocks'] for s in state.values() if s['outputs']==0)/cu>.5
  bins['decode_dominant']+=sum(s['blocks'] for s in state.values() if s['outputs']>0)/cu>.5
  bins['recomputing']+=rec>0
 assert recomputed==r['metrics']['recomputed_tokens']
 return dict(rounds=len(rows),state_coverage=bins)

def write_csv(name,rows):
 if not rows:return
 target=OUT/'analysis'/name;target.parent.mkdir(parents=True,exist_ok=True)
 with target.with_suffix('.csv.tmp').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
 target.with_suffix('.csv.tmp').replace(target)

def ci_cluster(rows):
 # Resample independent windows within each provenance source. Shapes stay together.
 cohorts={}
 for r in rows:cohorts.setdefault((r['source'],r['window']),[]).append(r['gain_pct'])
 sources={}
 for (s,w),xs in cohorts.items():sources.setdefault(s,[]).append(st.fmean(xs))
 if any(len(v)<2 for v in sources.values()):return None,None,len(cohorts)
 rng=random.Random(20260925);draws=[]
 for _ in range(2000):
  draws.append(st.fmean(st.fmean(rng.choices(v,k=len(v))) for v in sources.values()))
 draws.sort();return draws[49],draws[1949],len(cohorts)

def main(quick=False):
 d=read(OUT/'design.json');m=read(OUT/'manifest.json');assert digest(OUT/'design.json')==m['design_sha256']
 assert digest(OUT/'histories.json')==d['histories_sha256']
 for n,h in m['sources'].items():assert digest(OUT/'source'/n)==h
 for n,h in d['trace_hashes'].items():assert object_digest(read(OUT/f'inputs/{n}.json'))==h
 planned={t['group']:t for t in m['tasks']};accepted=[];failed=[];flat=[];pairs=[];total_rounds=0;calibrated=set();audited=0
 for p in sorted((OUT/'groups').glob('*.json')):
  entry=read(p);g=entry['group'];assert g in planned and entry['cell']==planned[g]['cell']
  if entry['state']!='complete':failed.append(entry);continue
  folder=OUT/entry['folder'];assert not (folder/'contamination.json').exists()
  assert read(folder/'attempt_status.json')==entry and read(folder/'worker_complete.json')['passed']
  task=read(folder/'task.json');expected={j['key']:j for j in task['jobs']}
  assert task['jobs']==planned[g]['jobs']
  runs=[read(p) for p in sorted((folder/'runs').glob('*.json'))]
  assert len(runs)==len(ARMS)*d['repeats'] and {r['job']['key'] for r in runs}==set(expected)
  ck=(entry['gpu'],entry['cell']['capacity']);cp=OUT/f'calibration/g{ck[0]}_c{ck[1]}'
  assert read(cp/'status.json')['passed']
  if ck not in calibrated:
   cr=[read(p) for p in (cp/'runs').glob('*.json')];assert len(cr)==2
   if not quick:
    for r in cr:total_rounds+=audit(r)['rounds']
   calibrated.add(ck)
  by={}
  for r in runs:
   j=r['job'];assert j==expected[j['key']] and r['gpu']==entry['gpu'] and r['status']=='complete'
   assert object_digest(r['requests'])==d['trace_hashes'][j['case']]
   assert j['method']==ARMS[j['arm']]['method'] and j['config']==ARMS[j['arm']]['config']
   assert r['saved_at']>=d['design_frozen_at']
   a={} if quick else audit(r)
   if not quick:total_rounds+=a['rounds'];audited+=1
   by.setdefault(j['arm'],{})[j['repeat']]=r
   flat.append(dict(group=g,**entry['cell'],arm=j['arm'],repeat=j['repeat'],gpu=r['gpu'],**r['metrics'],
                    **{f'coverage_{k}':v for k,v in a.get('state_coverage',{}).items()}))
  assert set(by)==set(ARMS) and all(set(rs)==set(range(d['repeats'])) for rs in by.values())
  for refarm in ARMS:
   if refarm=='arrival':continue
   ours=[by['arrival'][i]['metrics']['mean_ttft_ms'] for i in range(d['repeats'])]
   theirs=[by[refarm][i]['metrics']['mean_ttft_ms'] for i in range(d['repeats'])]
   gains=[100*(1-a/b) for a,b in zip(ours,theirs)]
   pair=dict(group=g,**entry['cell'],reference=refarm,gpu=entry['gpu'],arrival_ttft_ms=st.fmean(ours),reference_ttft_ms=st.fmean(theirs),
    gain_pct=100*(1-st.fmean(ours)/st.fmean(theirs)),arrival_spread_pct=100*(max(ours)-min(ours))/st.fmean(ours),
    reference_spread_pct=100*(max(theirs)-min(theirs))/st.fmean(theirs),positive_repeats=sum(v>0 for v in gains),
    gain_repeat_min=min(gains),gain_repeat_max=max(gains),**{f'gain_r{i}_pct':v for i,v in enumerate(gains)})
   for name in ('preemptions','recomputed_tokens','interventions','p95_ttft_ms','mean_tbt_ms','mean_e2e_ms','throughput_tokens_s','schedule_ms_mean'):
    for prefix,arm in [('arrival','arrival'),('reference',refarm)]:
     vals=[by[arm][i]['metrics'][name] for i in range(d['repeats'])]
     pair[f'{prefix}_{name}']=st.fmean(vals) if all(v is not None for v in vals) else None
   pairs.append(pair)
  accepted.append(g)
  if not quick and len(accepted)%10==0:
   write(OUT/'analysis/audit_progress.json',dict(groups=len(accepted),runs=audited,rounds=total_rounds,updated=time.time()))
   print('Audited',len(accepted),'groups',total_rounds,'rounds',flush=True)
 aggregates=[]
 for family in ('measured_lengths','synthetic'):
  for cap in CAPS:
   for refarm in ARMS:
    if refarm=='arrival':continue
    rs=[r for r in pairs if r['family']==family and r['capacity']==cap and r['reference']==refarm]
    if not rs:continue
    lo,hi,clusters=ci_cluster(rs) if family=='measured_lengths' else (None,None,0)
    gain=st.fmean(r['gain_pct'] for r in rs)
    assessment='证据不足/差异较小'
    if lo is not None and lo>0 and gain>=2:assessment='本范围平均改善证据'
    if hi is not None and hi<0 and gain<=-2:assessment='本范围平均退化证据'
    a=dict(family=family,capacity=cap,reference=refarm,cells=len(rs),cohorts=clusters,gain_pct=gain,ci95_low=lo,ci95_high=hi,
     worst_gain_pct=min(r['gain_pct'] for r in rs),better_over_2pct=sum(r['gain_pct']>2 for r in rs),worse_over_2pct=sum(r['gain_pct']<-2 for r in rs),
     all_repeats_positive=sum(r['positive_repeats']==4 for r in rs),assessment=assessment)
    for metric in ('preemptions','recomputed_tokens'):
     x=sum(r[f'arrival_{metric}'] for r in rs);b=sum(r[f'reference_{metric}'] for r in rs)
     a[f'{metric}_reduction_pct']=100*(1-x/b) if b else None
    aggregates.append(a)
 missing=sorted(set(planned)-set(accepted));sim=read(OUT/'simulator/full_complete.json') if (OUT/'simulator/full_complete.json').exists() else {}
 summary=dict(updated=time.time(),complete_matrix=not missing,accepted_groups=len(accepted),planned_groups=len(planned),formal_runs=len(flat),
  audit=dict(passed=not quick,rounds=total_rounds),failed_groups=failed,missing_groups=missing,calibration_runs=2*len(calibrated),simulator=sim,aggregates=aggregates,
  note='Quick reports do not run independent round audits.' if quick else 'All accepted runs and used calibrations independently audited.')
 for name,rows in [('runs.csv',flat),('comparisons.csv',pairs),('aggregates.csv',aggregates)]:write_csv(name,rows)
 write(OUT/'analysis/summary.json',summary)
 report(summary,pairs)
 if not quick:plot(aggregates)
 print(json.dumps({k:summary[k] for k in ('accepted_groups','planned_groups','formal_runs','audit')},ensure_ascii=False))

def report(s,pairs):
 lines=['# 旧式＋A：KV 容量与论文规则同条件比较','',
 f"完整组 {s['accepted_groups']}/{s['planned_groups']}；正式运行 {s['formal_runs']}；独立逐轮审计 {'通过' if s['audit']['passed'] else '未执行（快速预览）'}，{s['audit']['rounds']:,} 轮。",'',
 '**矩阵完整。**' if s['complete_matrix'] else '**矩阵不完整，以下为已完成组，不能宣布全范围优胜；所有缺失/失败列于 summary.json。**','',
 '只验证旧式＋A；未运行 D 变体。固定公式、同组同卡、每方法4重复。主指标 mean TTFT，先平均重复再对场景相对改善等权平均。正值表示 A 更好。',
 '每容量48个实测长度单元来自12个互不重叠请求窗口；4种形状共用窗口，容量也共用请求，不能当作48套独立样本。置信区间按来源分层、以窗口为簇bootstrap 2000次；仅12个窗口，属于描述性不确定性区间，未作多重比较校正。',
 '4种合成负载单独列出。无在线调参，不因结果好坏重跑或增删场景。收益≥2%且区间下界>0仅标记本范围平均改善证据，不是保证每个负载适用。','',
 'Sarathi/Past-Future是原有论文规则在同一vLLM V1上的适配；不是原系统复现。common使用1024/256/32，frozen仅chunk改为1024，其余相同。native保留原生请求排序及分配并加入统计，不能称为未插桩生产vLLM。SRP是同排序固定预算消融。',
 '所有实际推理均为Qwen2.5-7B FP16、block16、最大长度8176，固定token ID100回放长度；不是四个模型、原始文本或原始时间尺度实验。模拟器仅检验原生分配器与调度行为，不提供GPU TTFT预测。','',
 '| 类别 | KV块 | 对照 | 单元 | A改善 | 95%区间 | 最差单元 | 退化>2%单元 | 判断 |',
 '|---|---:|---|---:|---:|---|---:|---:|---|']
 for a in s['aggregates']:
  ci='—' if a['ci95_low'] is None else f"[{a['ci95_low']:+.2f}, {a['ci95_high']:+.2f}]"
  lines.append(f"| {a['family']} | {a['capacity']} | {a['reference']} | {a['cells']} | {a['gain_pct']:+.2f}% | {ci} | {a['worst_gain_pct']:+.2f}% | {a['worse_over_2pct']} | {a['assessment']} |")
 lines+=['','每组四次重复的原值、极差/均值、配对改善、p95 TTFT、吞吐、TBT、E2E、抢占和重算保存在 analysis/runs.csv 与 comparisons.csv。失败/超时整组不混入完整配对均值，必须连同失败率审阅，不能将条件成功均值作为全面优胜证据。',
 '策略调度、诊断和审计开销保留在端到端计时中，各方法开销另外记录；未扣除有利于任何一方的开销。',
 '统计描述局部负载下的容量响应，不是线上按容量切换策略的验证。宽松区零干预时，小计时差异不能解释为策略收益。',
 '', '[冻结协议](PROTOCOL_CN.md) · [进度](PROGRESS_CN.md) · [结果数据](../../artifacts/v1_arrival_kv_papers_20260925_r1/analysis/summary.json)']
 for file in ('RESULTS_CN.md','ANALYSIS_CN.md'):(HERE/file).write_text('\n'.join(lines)+'\n')

def plot(rows):
 os.environ.setdefault('MPLCONFIGDIR','/tmp/arrival-kv-papers-mpl')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(11,6))
 for refarm in ('old','srp','native','sarathi_common','past_future_common'):
  rs=[r for r in rows if r['family']=='measured_lengths' and r['reference']==refarm]
  if not rs:continue
  ax.plot([r['capacity'] for r in rs],[r['gain_pct'] for r in rs],marker='o',label=refarm)
 ax.axhline(0,color='black',linewidth=.8);ax.set(xlabel='KV blocks (16 tokens/block)',ylabel='Mean TTFT improvement of old+A (%)',title='Paired GPU comparison; equal weighting of completed cells')
 ax.legend();ax.grid(alpha=.25);fig.tight_layout();(OUT/'figures').mkdir(exist_ok=True)
 for ext in ('png','pdf'):fig.savefig(OUT/f'figures/kv_curve.{ext}',dpi=160)
 plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');a=p.parse_args();main(a.quick)
