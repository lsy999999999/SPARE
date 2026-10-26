"""Recompute saved metrics and audit new-policy choices before reporting."""
import gzip
import json
import math
from pathlib import Path
import statistics as st
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_state_aware_20260923.common import *
from research.v1_state_aware_20260923.run import audit, jobs
from research.v1_budget_20260922.run import quantile


def recompute(row):
    audit(row)
    with gzip.open(OUT/row['rounds_path'],'rt') as f: rounds=json.load(f)
    for r in rounds:
        if r['verified']:
            assert r['actual_growth_blocks']==r['free_before']-r['free_after']>=0
        if row['job']['method'].startswith('pressure_') and r['intervened']:
            g=r['baseline_growth_blocks']; saving=g-r['actual_growth_blocks']
            assert 4*g>=max(r['free_before'],1)
            assert saving>= (1 if row['job']['method']=='pressure_one' else (g+1)//2)
            assert r['candidate_diagnostics'][-1]['rejection'] is None
            for x in r['reductions']:
                assert not (x['outputs']==0 and x['baseline_tokens']==x['remaining'])
                assert not (x['outputs']>0 and x['remaining']==1)
        for d in r['allocations']:
            if r['verified']:
                assert d['blocks_after']-d['blocks_before']==max(0,(d['before']+d['tokens']+15)//16-d['blocks_before'])
    if row['status']!='complete': return None
    clocks=row['clocks']; tt=[c['first_ms']-c['arrival_ms'] for c in clocks.values()]
    es=[c['finish_ms']-c['arrival_ms'] for c in clocks.values()]
    gaps=[b-a for c in clocks.values() for a,b in zip(c['token_times_ms'],c['token_times_ms'][1:])]
    long_ids=[r['id'] for r in sorted(row['requests'],key=lambda r:(-r['prompt'],r['id']))[:max(1,len(clocks)//4)]]
    result=dict(mean_ttft_ms=st.fmean(tt),p95_ttft_ms=quantile(tt,.95),mean_e2e_ms=st.fmean(es),
        p95_e2e_ms=quantile(es,.95),throughput_tokens_s=sum(c['tokens'] for c in clocks.values())/(row['elapsed_ms']/1000),
        p95_tbt_ms=quantile(gaps,.95),long_prompt_ttft_ms=st.fmean(clocks[r]['first_ms']-clocks[r]['arrival_ms'] for r in long_ids),
        max_ttft_ms=max(tt),preemptions=sum(r['preemptions'] for r in rounds),
        recomputed_tokens=sum(r['recomputed_tokens'] for r in rounds),interventions=sum(r['intervened'] for r in rounds),
        decision_ms_total=sum(r['decision_ms'] for r in rounds),max_consecutive_reductions=max(r['max_consecutive_reductions'] for r in rounds))
    for key in ('mean_ttft_ms','p95_ttft_ms','mean_e2e_ms','p95_e2e_ms','throughput_tokens_s','p95_tbt_ms','preemptions','recomputed_tokens','interventions'):
        assert math.isclose(result[key],row['metrics'][key],rel_tol=1e-9,abs_tol=1e-6),(key,result[key],row['metrics'][key])
    return result


def main():
    m=check_manifest(); results=[]; hashes={}
    cells=[dict(key=f'validation_g{g}',gpu=g,capacity=768,block=None) for g in m['gpus']]+m['cells']
    for cell in cells:
        for j in jobs(m,cell):
            p=OUT/'runs'/(j['key']+'.json'); row=read(p)
            assert row['job']==j and row['requests']==__import__('research.v1_state_aware_20260923.run',fromlist=['requests']).requests(j)
            met=recompute(row); hashes[j['key']]=digest(p)
            if j['phase']=='compare': results.append(dict(job=j,status=row['status'],metrics=met))
    assert len(results)==72
    cells_summary=[]
    for cell in m['cells']:
        grouped={}
        for method in METHODS:
            rs=[r for r in results if r['job']['cell']==cell and r['job']['method']==method]
            assert len(rs)==2
            grouped[method]=None if any(r['metrics'] is None for r in rs) else {
                k:st.fmean(r['metrics'][k] for r in rs) for k in rs[0]['metrics']}
        gains={method:{base:100*(1-met['mean_ttft_ms']/grouped[base]['mean_ttft_ms'])
                       for base in ('a_frozen','a_any_positive_saving','fixed_small') if grouped[base] is not None}
               for method,met in grouped.items() if met is not None}
        cells_summary.append(dict(cell=cell,metrics=grouped,gain_pct=gains))
    payload=dict(complete=True,all_runs_successful=all(r['metrics'] is not None for r in results),
                 claim='development only; repeats averaged within each of two windows',
                 runs=results,cells=cells_summary,run_hashes=hashes)
    write(OUT/'summary.json',payload)
    lines=['# 当前状态驱动预算：开发结果','',
           '72 次比较已审计；每格两次技术重复平均，只有两个开发窗口。下列收益均相对同点对照，正值更好。', '',
           '| 窗口 / 容量 | 方法 | TTFT ms | 对固定512改善 | 对1-block改善 | 吞吐 tok/s | 抢占 |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for cell in cells_summary:
        for method,met in cell['metrics'].items():
            if met is None:
                lines.append(f"| {cell['cell']['key']} | {method} | 未完整完成 | — | — | — | — |")
                continue
            gain=cell['gain_pct'][method]
            fmt=lambda base:f"{gain[base]:+.1f}%" if base in gain else '—'
            lines.append(f"| {cell['cell']['key']} | {method} | {met['mean_ttft_ms']:.1f} | {fmt('fixed_small')} | {fmt('a_any_positive_saving')} | {met['throughput_tokens_s']:.1f} | {met['preemptions']:.1f} |")
    lines += ['', '完整尾延迟、长输入组、E2E、输出间隔、连续减配和在线决策成本见 summary.json。',
              'CPU 模拟使用真实 V1 分配/恢复，但没有校准毫秒时钟，因此不提供模拟 TTFT 排名。',
              '新规则参数预先固定，结果无论正负均保留；该开发结果不能代替新窗口的独立确认。']
    (OUT/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'audit.json',dict(passed=True,compared_runs=72,validated_runs=len(hashes),source_hashes=m['source_hashes']))
    print(OUT/'RESULTS_CN.md',flush=True)


if __name__=='__main__': main()
