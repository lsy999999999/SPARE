"""Incremental paired GPU summaries and action/arrival mechanism diagnostics."""
import csv
import gzip
import math
from pathlib import Path
import statistics as st
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_replica_extension_20260925.common import *


def mechanism(result):
    target=OUT/'analysis/mechanism'/f"{result['task']['key']}_{result['job']['key']}_replica{result['rank']}.json"
    if target.exists():
        existing=read(target)
        assert existing['rounds_sha256']==result['rounds_sha256']
        return existing
    path=OUT/result['rounds_path'];assert digest(path)==result['rounds_sha256']
    with gzip.open(path,'rt') as stream:rounds=json.load(stream)
    events=[e for row in rounds for e in row.get('arrival_events',[])]
    counts=dict(rounds=len(rounds),eligible_rounds=0,threshold_changed=0,action_changed=0,
                interventions=0,deferred_tokens=0,deferred_requests=0,physical_blocks_saved=0,
                stop_tail_action_changes=0)
    details=[];forecast_errors=[];estimator_errors=[];last_arrival=max(e['time_s'] for e in events)
    previous_end=0.;total_area=0.
    for row in rounds:
        feat=row['features'];rate=feat['arrival_rate'];t=feat['arrival_time_s']
        expected=sum(e['prompt']*math.exp(-(t-e['time_s'])) for e in events if e['time_s']<=t)
        assert math.isclose(rate,expected,abs_tol=1e-7,rel_tol=1e-8)
        future=sum(e['prompt'] for e in events if t<e['time_s']<=t+1.)
        forecast_errors.append(rate-future)
        f=row['free_before'];w=feat['remaining_prefill']
        n0=max(1,min(100000,math.ceil(8+128*f/(1+w/16))))
        na=max(1,min(100000,math.ceil(8+128*f/(1+(w+rate)/16))))
        assert row['threshold']==na
        candidates=[c for c in row['candidate_diagnostics'] if c['guard'] and c['tokens'] and c['saving'] is not None]
        eligible=row['baseline'] is not None and row['baseline_growth_blocks']>0 and len(row['state'])>=2
        action=lambda n:next((c['tokens'] for c in candidates if c['saving']>=n),row['baseline']) if eligible else row['baseline']
        if eligible:
            assert action(na)==row['scheduled']
            changed=action(n0)!=action(na)
            counts['eligible_rounds']+=1;counts['threshold_changed']+=n0!=na;counts['action_changed']+=changed
            counts['stop_tail_action_changes']+=changed and t>last_arrival
        else:changed=False
        area=sum(max(0,min(c['first_ms'],row['step_end_ms'])-max(c['arrival_ms'],previous_end)) for c in result['clocks'].values())
        previous_end=row['step_end_ms'];total_area+=area
        if row['intervened']:
            saving=row['baseline_growth_blocks']-row['actual_growth_blocks']
            assert saving>=na
            counts['interventions']+=1;counts['physical_blocks_saved']+=saving
            counts['deferred_tokens']+=sum(r['baseline_tokens']-r['actual_tokens'] for r in row['reductions'])
            counts['deferred_requests']+=len(row['reductions'])
            details.append(dict(round=row['round'],time_s=t,free=f,work=w,arrival_rate=rate,
                forecast_next_second_tokens=rate,observed_next_second_tokens=future,
                threshold_basic=n0,threshold_arrival=na,action_changed_vs_basic=changed,
                physical_saving=saving,deferred_requests=len(row['reductions']),
                deferred_tokens=sum(r['baseline_tokens']-r['actual_tokens'] for r in row['reductions']),
                step_ms=row['step_ms'],interval_waiting_area_ms=area,preemptions=row['preemptions']))
    expected_area=sum(c['first_ms']-c['arrival_ms'] for c in result['clocks'].values())
    assert math.isclose(total_area,expected_area,abs_tol=1e-5,rel_tol=1e-9)
    answer=dict(task=result['task']['key'],job=result['job'],rank=result['rank'],rounds_sha256=result['rounds_sha256'],
        counts=counts,forecast_bias_tokens=st.fmean(forecast_errors),forecast_mae_tokens=st.fmean(abs(x) for x in forecast_errors),
        waiting_area_verified=True,intervention_details=details,
        caveat='Decision-sampled forecast errors in a declared finite replay. Same-state action diagnostics are not closed-loop causal outcomes.')
    write(target,answer);return answer


def csv_write(path,rows):
    if not rows:return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    design=read(OUT/'design.json');manifest=read(OUT/'manifest.json')
    rows=[];pairs=[];diagnostics=[];accepted=[];failures=[];partial=[]
    for task in design['tasks']:
        folder=OUT/'groups'/task['key'];status=read(folder/'status.json') if (folder/'status.json').exists() else {}
        if status.get('state')!='complete' or (folder/'contamination.json').exists() or (folder/'interrupted.json').exists():
            (failures if status.get('state')=='failed' or (folder/'controller_failure.json').exists() else partial).append(task['key'])
            continue
        assert read(folder/'calibration.json')['passed']
        accepted.append(task['key']);records=[read(p) for p in (folder/'runs').glob('*.json')]
        records=[r for r in records if r['job']['phase']=='compare']
        assert len(records)==len(task['cases'])*len(task['loads'])*len(ARMS)*2
        for r in records:
            j=r['job']
            recomputed=st.fmean(c['first_ms']-c['arrival_ms'] for c in r['clocks'].values())
            assert math.isclose(recomputed,r['metrics']['mean_ttft_ms'],abs_tol=1e-6)
            rows.append(dict(task=task['key'],model=task['model'],capacity=task['capacity'],replicas=task['replicas'],
                case=j['case'],load=j['load'],arm=j['arm'],repeat=j['repeat'],**r['metrics']))
            if j['arm']=='arrival':
                for path in r['replica_files']:diagnostics.append(mechanism(read(OUT/path)))
        for case in task['cases']:
            for load in task['loads']:
                cell=[r for r in rows if r['task']==task['key'] and r['case']==case and r['load']==load]
                means={arm:st.fmean(r['mean_ttft_ms'] for r in cell if r['arm']==arm) for arm in ARMS}
                for reference in ARMS:
                    if reference=='arrival':continue
                    a=[r for r in cell if r['arm']=='arrival'];b=[r for r in cell if r['arm']==reference]
                    a.sort(key=lambda r:r['repeat']);b.sort(key=lambda r:r['repeat'])
                    pairs.append(dict(task=task['key'],model=task['model'],capacity=task['capacity'],replicas=task['replicas'],case=case,load=load,
                        reference=reference,gain_pct=100*(1-means['arrival']/means[reference]),
                        gain_r0=100*(1-a[0]['mean_ttft_ms']/b[0]['mean_ttft_ms']),gain_r1=100*(1-a[1]['mean_ttft_ms']/b[1]['mean_ttft_ms']),
                        arrival_ttft_ms=means['arrival'],reference_ttft_ms=means[reference],
                        arrival_interventions=st.fmean(r['interventions'] for r in a),reference_interventions=st.fmean(r['interventions'] for r in b),
                        arrival_preemptions=st.fmean(r['preemptions'] for r in a),reference_preemptions=st.fmean(r['preemptions'] for r in b),
                        arrival_recomputed_tokens=st.fmean(r['recomputed_tokens'] for r in a),reference_recomputed_tokens=st.fmean(r['recomputed_tokens'] for r in b)))
    (OUT/'analysis').mkdir(exist_ok=True)
    csv_write(OUT/'analysis/runs.csv',rows);csv_write(OUT/'analysis/comparisons.csv',pairs)
    scaling=[]
    for task in design['tasks']:
        if task['replicas'] not in (2,4) or task['key'] not in accepted:continue
        for case in task['cases']:
            for load in task['loads']:
                for arm in ARMS:
                    single=[r for r in rows if r['model']==task['model'] and r['capacity']==task['capacity']
                            and r['case']==case and r['replicas']==1 and r['load']=='fixed_total' and r['arm']==arm]
                    multi=[r for r in rows if r['task']==task['key'] and r['case']==case and r['load']==load and r['arm']==arm]
                    if len(single)!=2 or len(multi)!=2:continue
                    one=st.fmean(r['mean_ttft_ms'] for r in single);two=st.fmean(r['mean_ttft_ms'] for r in multi)
                    throughput_one=st.fmean(r['throughput_tokens_s'] for r in single)
                    throughput_two=st.fmean(r['throughput_tokens_s'] for r in multi)
                    scaling.append(dict(model=task['model'],capacity_per_replica=task['capacity'],case=case,load=load,arm=arm,
                        replicas=task['replicas'],single_mean_ttft_ms=one,multi_mean_ttft_ms=two,ttft_improvement_pct=100*(1-two/one),
                        throughput_ratio=throughput_two/throughput_one,
                        resource_note=f"{task['replicas']} GPUs versus one; proportional total KV. Hardware scaling, not isolated scheduler benefit."))
    csv_write(OUT/'analysis/scaling.csv',scaling)
    summary=dict(updated=time.time(),accepted_groups=accepted,failed_groups=failures,pending_or_partial=partial,
        formal_global_episodes=len(rows),paired_cells=len(pairs)//5,planned_global_episodes=design['planned_global_episodes'],
        primary='Mean of scenario relative mean-TTFT gains after averaging two repeats; models and replica loads remain separated.',
        mechanics=[{k:v for k,v in d.items() if k!='intervention_details'} for d in diagnostics])
    write(OUT/'analysis/summary.json',summary)
    lines=['# 机制、多模型与多副本：GPU阶段性结果','',
        f"完整通过组：{len(accepted)}/{len(design['tasks'])}；正式全局episode：{len(rows)}/{design['planned_global_episodes']}；失败组：{len(failures)}。只有完整、校准通过且无污染的组进入下表。",'',
        '一个复用的实测长度窗口、两种到达形状、每方法两重复；属于起步探索，不作跨负载普遍性或显著性结论。正值表示basic+A更快。','',
        '| 模型 | KV块/副本 | 副本 | 输入负载 | 到达形状 | A对basic | A对SRP | A对原生 | A对Sarathi规则 | A对Past-Future规则 |',
        '|---|---:|---:|---|---|---:|---:|---:|---:|---:|']
    keys=sorted({(p['model'],p['capacity'],p['replicas'],p['load'],p['case']) for p in pairs})
    for model,cap,replicas,load,case in keys:
        by={p['reference']:p for p in pairs if (p['model'],p['capacity'],p['replicas'],p['load'],p['case'])==(model,cap,replicas,load,case)}
        lines.append(f'| {model} | {cap} | {replicas} | {load} | {case.removeprefix("conversation_w0_")} | '+
                     ' | '.join(f"{by[ref]['gain_pct']:+.2f}%" for ref in ('old','srp','native','sarathi_common','past_future_common'))+' |')
    lines+=['','## 机制观察（执行动作与等待成本）','',
        '离线重建同状态basic/A决策与到达估计，并核对逐时间区间等待面积。阈值变化不等于动作变化；单条轨迹中的延后与抢占统计不作为因果分解。','',
        '| 模型/容量/副本 | 场景/负载 | A干预（两次平均） | basic干预 | A抢占 | basic抢占 | TTFT改善 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for p in pairs:
        if p['reference']!='old':continue
        lines.append(f"| {p['model']}/{p['capacity']}/{p['replicas']} | {p['case']}/{p['load']} | {p['arrival_interventions']:.1f} | {p['reference_interventions']:.1f} | {p['arrival_preemptions']:.1f} | {p['reference_preemptions']:.1f} | {p['gain_pct']:+.2f}% |")
    lines+=['','## 多副本解释边界','',
        '共同入口真实派发，按到达顺序轮询，不使用长度预测。fixed_total保持全局64请求，per_replica使各副本均收到64请求。TTFT从全局到达计起，含入口和副本队列等待。',
        '每副本配额固定，增加副本也增加GPU与总KV；表中算法比较固定副本数，不能将1→2/4副本的全部改善归给算法。论文方法为规则适配版，不是原完整系统。','',
        '失败组：'+(', '.join(failures) if failures else '无'),
        '待完成/部分组：'+(', '.join(partial) if partial else '无'),'',
        '机器可读结果：analysis/comparisons.csv、analysis/runs.csv、analysis/summary.json；每个副本的逐轮机制记录在analysis/mechanism。']
    if scaling:
        lines+=['','## 同方法从一副本到多副本','',
            '以下包括增加GPU资源的效果；算法是否改善仍看前面的同副本数对照。fixed_total为相同全局请求，per_replica为全局请求按副本数倍增。','',
            '| 模型 | 副本 | 方法 | KV块/副本 | 形状 | 负载 | 1副本TTFT ms | 多副本TTFT ms | TTFT改善 | 吞吐倍数 |',
            '|---|---:|---|---:|---|---|---:|---:|---:|---:|']
        for r in scaling:
            lines.append(f"| {r['model']} | {r['replicas']} | {r['arm']} | {r['capacity_per_replica']} | {r['case'].removeprefix('conversation_w0_')} | {r['load']} | {r['single_mean_ttft_ms']:.2f} | {r['multi_mean_ttft_ms']:.2f} | {r['ttft_improvement_pct']:+.2f}% | {r['throughput_ratio']:.2f} |")
    probe_rows=[]
    for folder in sorted((OUT/'probes').glob('*_c*')):
        if not (folder/'summary.json').exists() or not (folder/'status.json').exists():continue
        if read(folder/'status.json')['state']!='complete' or (folder/'contamination.json').exists() or (folder/'interrupted.json').exists():continue
        result=read(folder/'summary.json')
        for state in result['states']:probe_rows.append(dict(model=result['model'],capacity=result['capacity'],**state))
    csv_write(OUT/'analysis/probes.csv',probe_rows)
    if probe_rows:
        lines+=['','## 同状态一次性动作分支','',
            '所有请求预先到达；完整匹配物理/策略状态和前缀估计器时钟，仅本轮A与不节约不同，随后同用A。正值表示本轮节约减少了剩余首token等待；不是全策略平均收益。','',
            '| 模型 | KV块 | 分支轮 | 省块 | 等价平均TTFT减少ms | 第一次等待面积减少ms | 第二次等待面积减少ms |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for r in probe_rows:lines.append(f"| {r['model']} | {r['capacity']} | {r['round']} | {r['saving_blocks']} | {r['equivalent_mean_ttft_saved_ms']:+.3f} | {r['gain_r0_area_ms']:+.2f} | {r['gain_r1_area_ms']:+.2f} |")
    (OUT/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n')
    print(dict(accepted=len(accepted),failed=len(failures),formal=len(rows),paired_cells=len(pairs)//5),flush=True)


if __name__=='__main__':main()
