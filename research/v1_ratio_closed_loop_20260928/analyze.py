"""Full GPU audit and paired full-loop comparisons; no post-hoc window removal."""
import csv,gzip,statistics as st,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_ratio_closed_loop_20260928.common import *
from research.v1_ratio_closed_loop_20260928.audit import audit
from research.v1_ratio_closed_loop_20260928.policy import pairs,small_chunk

def summarize(rows):
    return dict(scenes=len(rows),mean_improvement_pct=st.fmean(r['improvement_pct'] for r in rows) if rows else None,
        worst_improvement_pct=min((r['improvement_pct'] for r in rows),default=None),
        best_improvement_pct=max((r['improvement_pct'] for r in rows),default=None),
        both_repeats_above_2pct=sum(min(r['repeat_improvement_pct'])>2 for r in rows),
        both_repeats_below_minus2pct=sum(max(r['repeat_improvement_pct'])< -2 for r in rows))

def main():
    d=verify();checks=[];comparisons=[];runs=[];excluded=[]
    for g in d['groups']:
        folder=OUT/'groups'/g['key']
        if not (folder/'status.json').exists() or read(folder/'status.json')['state']!='complete' or (folder/'exclusion.json').exists():
            excluded.append(g['key']);continue
        aa=read(folder/'aa.json');assert aa['passed'] and (max(aa['values'])-min(aa['values']))/st.fmean(aa['values'])<=.05
        rr={};paths=sorted((folder/'runs').glob('*.json'));assert len(paths)==14
        for path in paths:
            r=read(path);rp=OUT/r['rounds_path'];assert digest(rp)==r['rounds_sha256']
            assert r['requests']==read(OUT/f"inputs/{g['input']}_{r['shape']}.json")
            assert r['arm']==r['config']['threshold']['candidate_mode']
            with gzip.open(rp,'rt') as f:rows=json.load(f)
            check=audit(r,rows);assert check==r['audit'];checks.append(dict(group=g['key'],run=path.stem,**check))
            runs.append(dict(group=g['key'],phase=r['phase'],shape=r['shape'],arm=r['arm'],repeat=r['repeat'],metrics=r['metrics'],audit=check))
            if r['phase']=='formal':rr[r['shape'],r['arm'],r['repeat']]=r
        for shape in SHAPES:
            base=[rr[shape,'fixed64',rep] for rep in range(2)]
            bm=st.fmean(r['metrics']['mean_ttft_ms'] for r in base)
            for arm in ARMS[1:]:
                alt=[rr[shape,arm,rep] for rep in range(2)];am=st.fmean(r['metrics']['mean_ttft_ms'] for r in alt)
                gains=[100*(b['metrics']['mean_ttft_ms']-a['metrics']['mean_ttft_ms'])/b['metrics']['mean_ttft_ms'] for b,a in zip(base,alt)]
                same=pairs(g['budget'],g['chunk'],arm)==pairs(g['budget'],g['chunk'],'fixed64')
                row=dict(group=g['key'],source=g['source'],window=g['window'],shape=shape,budget=g['budget'],chunk=g['chunk'],
                    arm=arm,small_chunk=small_chunk(g['chunk'],arm),same_candidates=same,
                    baseline_ttft_ms=bm,arm_ttft_ms=am,improvement_pct=100*(bm-am)/bm,repeat_improvement_pct=gains,
                    changed_rounds=sum(r['audit']['different_from_fixed64'] for r in alt),
                    rounds=sum(r['audit']['rounds'] for r in alt),
                    small_candidate_rounds=sum(r['audit']['selected_candidates'][str(i)] for r in alt for i in (2,3)),
                    baseline_preemptions=sum(r['metrics']['preemptions'] for r in base),arm_preemptions=sum(r['metrics']['preemptions'] for r in alt),
                    baseline_recomputed=sum(r['metrics']['recomputed_tokens'] for r in base),arm_recomputed=sum(r['metrics']['recomputed_tokens'] for r in alt),
                    baseline_schedule_ms=st.fmean(r['metrics']['schedule_ms_mean'] for r in base),
                    arm_schedule_ms=st.fmean(r['metrics']['schedule_ms_mean'] for r in alt))
                if same:assert row['changed_rounds']==0
                comparisons.append(row)
    summary=dict(accepted_groups=len(d['groups'])-len(excluded),excluded=excluded,comparisons=comparisons,
        arms={a:dict(all=summarize([r for r in comparisons if r['arm']==a]),
                     changed_candidate_configs=summarize([r for r in comparisons if r['arm']==a and not r['same_candidates']]),
                     equivalent_controls=summarize([r for r in comparisons if r['arm']==a and r['same_candidates']])) for a in ARMS[1:]})
    write(OUT/'analysis/audit.json',dict(passed=True,episodes=len(checks),rounds=sum(c['rounds'] for c in checks),excluded=excluded,checks=checks))
    write(OUT/'analysis/summary.json',summary);write(OUT/'analysis/runs.json',runs)
    if comparisons:
        with (OUT/'analysis/comparisons.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
    lines=['# 比例化候选 GPU 闭环结果','',f'有效组 {summary["accepted_groups"]}/16；完整运行 {len(checks)}；审计轮数 {sum(c["rounds"] for c in checks)}；排除 {excluded}。',
        '', '正值代表mean TTFT改善；同场景先平均两次，再对场景等权平均。不将两次重复当显著性检验。',
        '', '|规则|全部场景平均改善|改变候选的配置平均改善|等价对照平均改善|', '|---|---:|---:|---:|']
    for arm,v in summary['arms'].items():
        lines.append(f'|{arm}|{v["all"]["mean_improvement_pct"]}|{v["changed_candidate_configs"]["mean_improvement_pct"]}|{v["equivalent_controls"]["mean_improvement_pct"]}|')
    lines+=['','|来源/窗口|形状|T/K|规则/小K|改善%|两次改善%|与fixed64同状态不同计划轮数|','|---|---|---|---|---:|---|---:|']
    for r in comparisons:
        values='/'.join(f'{v:+.3f}' for v in r['repeat_improvement_pct'])
        lines.append(f'|{r["source"]}/{r["window"]}|{r["shape"]}|{r["budget"]}/{r["chunk"]}|{r["arm"]}/{r["small_chunk"]}|{r["improvement_pct"]:+.3f}|{values}|{r["changed_rounds"]}|')
    lines+=['','比例的分母是K的参考值：ratio16严格对应用户64/1024，ratio4用于保留旧K256→64参考点。',
            '基线与变体用相同实现；n、保护、固定顺序均不变。Simulator另列，不能当作GPU实测。']
    (OUT/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n')
    print({k:v for k,v in summary.items() if k!='comparisons'},flush=True)

if __name__=='__main__':main()
