"""Describe complete paired groups only; preserve branches and exclusions."""
import csv
import statistics as st
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_fixed_order_20260927.common import *


def csv_write(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    d=read(OUT/'design.json')
    pairs=[];flat=[];branches=[];excluded=[];complete=0
    metrics=['mean_ttft_ms','p95_ttft_ms','mean_tbt_ms','p95_tbt_ms','mean_e2e_ms','p95_e2e_ms',
             'throughput_tokens_s','preemptions','recomputed_tokens','interventions','dynamic_nonstandard_rounds','decision_ms_mean','schedule_ms_mean','no_prefill_progress_rounds','max_no_prefill_progress_streak','tiny_prefill_rounds','weak_prefill_rounds','different_from_reference_rounds']
    for group in d['groups']:
        folder=OUT/'groups'/group['key']
        if not (folder/'status.json').exists():continue
        status=read(folder/'status.json')
        if (folder/'exclusion.json').exists() or status['state']=='failed':
            excluded.append(dict(group=group['key'],state=status['state'],reason=read(folder/'exclusion.json') if (folder/'exclusion.json').exists() else status.get('error')))
            continue
        if status['state']!='complete':continue
        assert read(folder/'calibration.json')['passed']
        runs=[read(p) for p in sorted((folder/'runs').glob('*.json'))]
        formal=[r for r in runs if r['phase']=='formal']
        assert len(formal)==24 and all(r['audit']['passed'] for r in formal)
        complete+=1
        for shape in ('steady','burst_stop'):
            averaged={}
            for arm in ARMS:
                rr=sorted([r for r in formal if r['shape']==shape and r['arm']==arm],key=lambda r:r['repeat'])
                assert len(rr)==2 and [r['repeat'] for r in rr]==[0,1]
                averaged[arm]={m:st.fmean(r['metrics'][m] for r in rr) for m in metrics}
                for r in rr:
                    flat.append(dict(group=group['key'],source=group['source'],capacity=group['capacity'],shape=shape,
                                     arm=arm,repeat=r['repeat'],gpu=r['gpu'],**{m:r['metrics'][m] for m in metrics}))
            for arm in ARMS:
                a=averaged[arm];b=averaged['current']
                gain=100*(1-a['mean_ttft_ms']/b['mean_ttft_ms'])
                row=dict(group=group['key'],source=group['source'],capacity=group['capacity'],shape=shape,arm=arm,
                         gain_current_pct=gain,**a)
                for rep in (0,1):
                    ours=next(r for r in formal if r['shape']==shape and r['arm']==arm and r['repeat']==rep)
                    base=next(r for r in formal if r['shape']==shape and r['arm']=='current' and r['repeat']==rep)
                    row[f'gain_r{rep}_pct']=100*(1-ours['metrics']['mean_ttft_ms']/base['metrics']['mean_ttft_ms'])
                pairs.append(row)
        probes=read(folder/'probes.json')
        for target in sorted({r['target'] for r in probes['results']}):
            selected=[r for r in probes['results'] if r['target']==target]
            current=[r for r in selected if 'current' in r['aliases']]
            assert len(current)==2
            refarea=st.fmean(r['area_ms_request'] for r in current)
            for arm in sorted({r['arm'] for r in selected}):
                rr=[r for r in selected if r['arm']==arm]
                area=st.fmean(r['area_ms_request'] for r in rr)
                branches.append(dict(group=group['key'],source=group['source'],capacity=group['capacity'],target=target,
                    arm=arm,aliases='/'.join(rr[0]['aliases']),budget=rr[0]['budget'],chunk=rr[0]['chunk'],saving=rr[0]['saving'],
                    mean_area_ms_request=area,saved_vs_current_ms_request=refarea-area,equivalent_ttft_saved_ms=(refarea-area)/64,
                    gain_r0_area_ms=next(r['area_ms_request'] for r in current if r['repeat']==0)-next(r['area_ms_request'] for r in rr if r['repeat']==0),
                    gain_r1_area_ms=next(r['area_ms_request'] for r in current if r['repeat']==1)-next(r['area_ms_request'] for r in rr if r['repeat']==1)))
    csv_write(OUT/'analysis/runs.csv',flat);csv_write(OUT/'analysis/comparisons.csv',pairs);csv_write(OUT/'analysis/probes.csv',branches)
    summary=dict(complete_groups=complete,planned_groups=6,formal_runs=len(flat),formal_expected=144,excluded=excluded,
                 note='Exploratory reused windows; positive TTFT improvement means faster; simulator is not GPU timing.')
    write(OUT/'analysis/summary.json',summary)
    lines=['# 固定候选顺序与动态空间约束实验：GPU结果','',f'接受完整组 {complete}/6；正式闭环运行 {len(flat)}/144。仅统计完整、校准通过、无污染的组。','',
           '正数表示相对当前旧式＋A降低mean TTFT。每场景先平均两次，再等权平均场景。候选代价是启发式，不将局部评分当作真实TTFT。','',
           '| 方法 | 场景数 | 平均改善 | 最差改善 | >2%改善 | >2%退化 | 动态非原四动作轮数 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        rr=[r for r in pairs if r['arm']==arm]
        if rr:lines.append(f"| {arm} | {len(rr)} | {st.fmean(r['gain_current_pct'] for r in rr):+.3f}% | {min(r['gain_current_pct'] for r in rr):+.3f}% | {sum(r['gain_current_pct']>2 for r in rr)} | {sum(r['gain_current_pct']< -2 for r in rr)} | {sum(r['dynamic_nonstandard_rounds'] for r in rr):.1f} |")
    lines+=['','逐场景结果见 analysis/comparisons.csv；辅助延迟、吞吐、重算和真实决策开销见 analysis/runs.csv。',
            '同状态一次动作的结果见 analysis/probes.csv；面积差为current动作减替代动作，随后共同继续current，不等于全策略收益。',
            '空选点和运行中的分支记录保存在各groups目录；没有为了取得收益而补选状态。', '',f'排除组：{excluded}', '']
    (OUT/'RESULTS_CN.md').write_text('\n'.join(lines))
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
