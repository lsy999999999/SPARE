import csv
import statistics as st
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_efficient_points_20260927.common import *


def csv_write(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    design=read(OUT/'design.json');flat=[];pairs=[];excluded=[];completed=0
    metrics=['mean_ttft_ms','p95_ttft_ms','mean_tbt_ms','p95_tbt_ms','mean_e2e_ms','p95_e2e_ms',
        'throughput_tokens_s','preemptions','recomputed_tokens','interventions','schedule_ms_mean',
        'decision_ms_mean','no_prefill_progress_rounds','tiny_prefill_rounds','weak_prefill_rounds','different_from_reference_rounds']
    for group in design['groups']:
        folder=OUT/'groups'/group['key']
        if not (folder/'status.json').exists():continue
        status=read(folder/'status.json')
        if status['state']=='failed' or (folder/'exclusion.json').exists():
            excluded.append(dict(group=group['key'],reason=status.get('error'),contamination=(folder/'exclusion.json').exists()));continue
        if status['state']!='complete':continue
        assert read(folder/'calibration.json')['passed']
        runs=[read(p) for p in sorted((folder/'runs').glob('*.json'))]
        runs=[r for r in runs if r['phase']=='formal']
        assert len(runs)==20 and all(r['audit']['passed'] for r in runs)
        assert all(r['calibration_sha256']==digest(OUT/'calibration.json') for r in runs)
        completed+=1
        for r in runs:
            flat.append(dict(group=group['key'],source=group['source'],capacity=group['capacity'],shape=r['shape'],
                arm=r['arm'],repeat=r['repeat'],gpu=r['gpu'],**{m:r['metrics'][m] for m in metrics}))
        for shape in ('steady','burst_stop'):
            averaged={arm:{m:st.fmean(r['metrics'][m] for r in runs if r['shape']==shape and r['arm']==arm) for m in metrics} for arm in ARMS}
            for arm in ARMS:
                row=dict(group=group['key'],source=group['source'],capacity=group['capacity'],shape=shape,arm=arm,
                    gain_current_pct=100*(1-averaged[arm]['mean_ttft_ms']/averaged['current']['mean_ttft_ms']),
                    gain_fixed_pct=100*(1-averaged[arm]['mean_ttft_ms']/averaged['points_fixed']['mean_ttft_ms']),**averaged[arm])
                for rep in (0,1):
                    a=next(r for r in runs if r['shape']==shape and r['arm']==arm and r['repeat']==rep)
                    b=next(r for r in runs if r['shape']==shape and r['arm']=='current' and r['repeat']==rep)
                    row[f'gain_r{rep}_pct']=100*(1-a['metrics']['mean_ttft_ms']/b['metrics']['mean_ttft_ms'])
                pairs.append(row)
    csv_write(OUT/'analysis/runs.csv',flat);csv_write(OUT/'analysis/comparisons.csv',pairs)
    summary=dict(complete_groups=completed,planned_groups=6,formal_runs=len(flat),formal_expected=120,excluded=excluded)
    write(OUT/'analysis/summary.json',summary)
    lines=['# 实测高效预算点与动态选择：结果','',f'完整合格组 {completed}/6；正式运行 {len(flat)}/120。','',
        '正数表示 mean TTFT 降低；先平均同场景两次重复，再等权汇总场景。仅包含完整、A/A通过且无外部GPU进程污染的组。','',
        '| 方案 | 场景数 | 相对旧式＋A | 相对实测点固定序 | 最差场景（相对旧式＋A） |','|---|---:|---:|---:|---:|']
    for arm in ARMS:
        rr=[r for r in pairs if r['arm']==arm]
        if rr:lines.append(f"| {arm} | {len(rr)} | {st.fmean(r['gain_current_pct'] for r in rr):+.3f}% | {st.fmean(r['gain_fixed_pct'] for r in rr):+.3f}% | {min(r['gain_current_pct'] for r in rr):+.3f}% |")
    if (OUT/'calibration.json').exists():lines+=['',f"冻结候选顺序：{read(OUT/'calibration.json')['pairs']}"]
    lines+=['','辅助延迟、重算、抢占和开销见 analysis/runs.csv。逐场景比较见 analysis/comparisons.csv。',
        '这是小规模方向实验；耗时来自独立合成状态，w0工作负载在历史实验中使用过，不宣称未见测试集。',f'排除组：{excluded}','']
    (OUT/'RESULTS_CN.md').write_text('\n'.join(lines));print(summary,flush=True)


if __name__=='__main__':main()
