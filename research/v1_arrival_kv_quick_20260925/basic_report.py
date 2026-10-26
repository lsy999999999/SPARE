"""Small descriptive comparison from existing completed groups; no new GPU runs."""
import csv,json,math,statistics as st,sys,time
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from research.v1_arrival_kv_papers_20260925.common import OUT,read,write,digest
DEST=ROOT/'research/v1_arrival_kv_quick_20260925'
RESULT=ROOT/'artifacts/v1_arrival_kv_quick_20260925'
ARMS=['arrival','old','srp','native','sarathi_common','past_future_common']
NAMES={'arrival':'旧式＋A','old':'旧式','srp':'SRP','native':'vLLM原生调度（插桩）','sarathi_common':'Sarathi规则适配','past_future_common':'Past-Future规则适配'}
def csvout(name,rows):
 with (RESULT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def main():
 DEST.mkdir(parents=True,exist_ok=True);RESULT.mkdir(parents=True,exist_ok=True)
 groups=[read(p) for p in sorted((OUT/'groups').glob('*.json')) if read(p)['state']=='complete']
 rows=[];pairs=[]
 for g in groups:
  data=[read(p) for p in (OUT/g['folder']/'runs').glob('*.json')]
  by={a:{r['job']['repeat']:r for r in data if r['job']['arm']==a} for a in ARMS}
  assert all(set(rs)=={0,1,2,3} for rs in by.values())
  for a,rs in by.items():
   for rep,r in rs.items():
    assert r['status']=='complete'
    clocks=r['clocks'];ttft=st.fmean(c['first_ms']-c['arrival_ms'] for c in clocks.values())
    assert math.isclose(ttft,r['metrics']['mean_ttft_ms'],abs_tol=1e-6)
    rows.append(dict(group=g['group'],**g['cell'],gpu=g['gpu'],arm=a,repeat=rep,in_basic=rep<2,**r['metrics']))
  for reference in ARMS[1:]:
   avg=lambda a,k,n:st.fmean(by[a][i]['metrics'][k] for i in range(n))
   p=dict(group=g['group'],**g['cell'],gpu=g['gpu'],reference=reference,
    gain_pct=100*(1-avg('arrival','mean_ttft_ms',2)/avg(reference,'mean_ttft_ms',2)),
    gain_four_repeats_pct=100*(1-avg('arrival','mean_ttft_ms',4)/avg(reference,'mean_ttft_ms',4)))
   for i in range(2):p[f'gain_r{i}']=100*(1-by['arrival'][i]['metrics']['mean_ttft_ms']/by[reference][i]['metrics']['mean_ttft_ms'])
   for a,prefix in [('arrival','arrival'),(reference,'reference')]:
    for k in ['mean_ttft_ms','p95_ttft_ms','preemptions','recomputed_tokens','interventions','throughput_tokens_s']:
     p[f'{prefix}_{k}']=avg(a,k,2)
   pairs.append(p)
 caprows=[];overall=[]
 for family in ['measured_lengths','synthetic']:
  for ref in ARMS[1:]:
   selected=[p for p in pairs if p['family']==family and p['reference']==ref]
   if not selected:continue
   def summarize(ps):
    r=dict(cells=len(ps),gain_pct=st.fmean(p['gain_pct'] for p in ps),gain_four_repeats_pct=st.fmean(p['gain_four_repeats_pct'] for p in ps),
     worse_over_2=sum(p['gain_pct']<-2 for p in ps),better_over_2=sum(p['gain_pct']>2 for p in ps),worst_pct=min(p['gain_pct'] for p in ps),
     positive_both_repeats=sum(p['gain_r0']>0 and p['gain_r1']>0 for p in ps),zero_intervention_cells=sum(p['arrival_interventions']==0 and p['reference_interventions']==0 for p in ps))
    for k in ['preemptions','recomputed_tokens']:
     a=sum(p[f'arrival_{k}'] for p in ps);b=sum(p[f'reference_{k}'] for p in ps)
     r[k+'_reduction_pct']=100*(1-a/b) if b else None
    return r
   overall.append(dict(family=family,reference=ref,**summarize(selected)))
   for cap in sorted({p['capacity'] for p in selected}):
    ps=[p for p in selected if p['capacity']==cap]
    caprows.append(dict(family=family,capacity=cap,reference=ref,**summarize(ps)))
 summary=dict(created=time.time(),complete_groups=len(groups),measured_groups=sum(g['cell']['family']=='measured_lengths' for g in groups),
  synthetic_groups=sum(g['cell']['family']=='synthetic' for g in groups),basic_methods=ARMS,basic_repeats=2,basic_runs=len(groups)*12,
  original_completed_runs=len(groups)*32,distinct_source_windows=sorted({g['cell']['source']+'/'+g['cell']['window'] for g in groups if g['cell']['family']=='measured_lengths'}),
  overall=overall,capacities=caprows,metric_clock_checks=True,round_audit='See original analysis/status.json and summary.json; automatically running after stop.',
  stop=read(OUT/'user_stop.json'))
 audit_path=OUT/'analysis/summary.json'
 if audit_path.exists():
  final_audit=read(audit_path)
  if final_audit['audit']['passed'] and final_audit['accepted_groups']==len(groups):
   summary['round_audit']=dict(passed=True,rounds=final_audit['audit']['rounds'],source_sha256=digest(audit_path))
 write(RESULT/'summary.json',summary);csvout('runs.csv',rows);csvout('comparisons.csv',pairs);csvout('capacity_summary.csv',caprows)
 lines=['# 旧式＋A：缩小范围后的基本结果','',
 f"已按用户要求停止大矩阵。利用已完成的{len(groups)}个完整组，每组仅取6方法×前2重复＝12次作为基本比较，共{len(groups)*12}条记录；没有追加GPU运行。",'',
 f"其中实测长度{summary['measured_groups']}组，合成负载{summary['synthetic_groups']}组。各容量完成数量不均衡，因此这是一版初步结果，不是均衡容量扫描的最终结论。",'',
 '主指标mean TTFT；正值表示A更快。先平均组内两次，再对场景相对改善等权平均。按用户要求固定取前两次，不挑选表现更好的重复；已有四重复结果同时列出作敏感性核对，不增加运行。','',
 '| 实测长度对照 | 场景数 | 前2次平均改善 | 已有4次平均改善 | 改善>2% | 退化>2% | 最差场景 |',
 '|---|---:|---:|---:|---:|---:|---:|']
 for r in overall:
  if r['family']=='measured_lengths':lines.append(f"| {NAMES[r['reference']]} | {r['cells']} | {r['gain_pct']:+.2f}% | {r['gain_four_repeats_pct']:+.2f}% | {r['better_over_2']} | {r['worse_over_2']} | {r['worst_pct']:+.2f}% |")
 lines+=['','## 各容量初步比较（只含实测长度）','','不同容量的场景组成不齐；不得把此表解释为单因素容量曲线。','',
 '| KV块 | 场景数 | A对旧式 | A对SRP | A对原生调度 | A对Sarathi规则 | A对Past-Future规则 |',
 '|---|---:|---:|---:|---:|---:|---:|']
 for cap in sorted({r['capacity'] for r in caprows if r['family']=='measured_lengths'}):
  rs={r['reference']:r for r in caprows if r['family']=='measured_lengths' and r['capacity']==cap}
  lines.append(f"| {cap} | {rs['old']['cells']} | "+' | '.join(f"{rs[a]['gain_pct']:+.2f}%" for a in ARMS[1:])+' |')
 lines+=['','## 抢占与重算（实测长度总量）','','| 对照 | 抢占减少 | 重算token减少 |','|---|---:|---:|']
 for r in overall:
  if r['family']!='measured_lengths':continue
  fmt=lambda v:'—' if v is None else f'{v:+.2f}%'
  lines.append(f"| {NAMES[r['reference']]} | {fmt(r['preemptions_reduction_pct'])} | {fmt(r['recomputed_tokens_reduction_pct'])} |")
 lines+=['','## 边界与原始记录','',
 '所有方法共用Qwen2.5-7B、相同长度回放、预算1024/chunk256/并发32及组内同GPU。Sarathi和Past-Future是论文规则适配版，非原系统；原生调度有统计插桩。',
 '模拟器4992条主轨迹与48条敏感性轨迹已完成，用于分配器/因果/机制验证，不当作GPU性能。',
 '中止时不完整的组保留原始文件，未加入完整组配对均值。外来任务污染和校准失败的attempt同样不混入结果。既有完成组保留全部8方法×4重复，但主表只取上述6方法的前2重复。',
 '窗口跨形状/容量复用，不是每一组都有独立请求。约60组的初步平均不能证明普遍适用，尤其每容量只有少数组时，应结合退化单元和零干预情况审阅。',
 '本报告已重新核对逐请求时钟与mean TTFT；完整逐轮审计由原后台自动继续，其最终状态见原实验analysis/status.json。',
 '', '[JSON汇总](../../artifacts/v1_arrival_kv_quick_20260925/summary.json) · [逐场景比较](../../artifacts/v1_arrival_kv_quick_20260925/comparisons.csv) · [原始实验进度](../v1_arrival_kv_papers_20260925/PROGRESS_CN.md)']
 main={r['reference']:r for r in overall if r['family']=='measured_lengths'}
 lines[2:2]=[f"**基本判断：** A相对旧式的平均TTFT改善为{main['old']['gain_pct']:+.2f}%，已有四重复为{main['old']['gain_four_repeats_pct']:+.2f}%；这批数据没有显示加入A的额外平均收益。完整方案相对SRP与两种论文规则适配版平均更快，但这些收益不能全部归因于A项。",'', '相对Past-Future，A虽然mean TTFT更低，但抢占与重算更多，不能称为全指标更优。','']
 if isinstance(summary['round_audit'],dict):
  lines=[f"完整逐轮审计已通过：{summary['round_audit']['rounds']:,}轮（全部59个完整组的1,888次原始正式运行及使用过的校准）。GPU已停止；后台finished_with_gaps表示按用户要求提前结束、未执行完原矩阵，不是审计失败。" if line.startswith('本报告已重新核对') else line for line in lines]
 (DEST/'BASIC_RESULTS_CN.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({k:summary[k] for k in ['complete_groups','measured_groups','synthetic_groups','basic_runs','overall']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
