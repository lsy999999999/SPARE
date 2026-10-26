"""Regenerate selection, independently audit all rounds, report all fixed contrasts."""
import csv, gzip, json, statistics as st, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.v1_candidate_transfer_20260928.common import *
from research.v1_candidate_transfer_20260928.run import select_targets
from research.v1_candidate_audit_20260928.audit import audit

def load_rounds(run):
    p=OUT/run['rounds_path'];assert digest(p)==run['rounds_sha256']
    with gzip.open(p,'rt') as f:return json.load(f)

def main():
    d=verify();checks=[];contrasts=[];coverage=[];excluded=[]
    for g in d['groups']:
        folder=OUT/'groups'/g['key']
        if not (folder/'status.json').exists() or read(folder/'status.json')['state']!='complete' or (folder/'exclusion.json').exists():
            excluded.append(g['key']);continue
        assert read(folder/'aa.json')['passed']
        anchor_run=read(folder/'runs/anchor.json');anchor=load_rounds(anchor_run)
        selected=read(folder/'selected.json');assert selected==select_targets(anchor,g['budget'],g['chunk'])
        coverage.append(dict(group=g['key'],**{k:v for k,v in selected.items() if k!='states'}))
        runs={}
        for p in sorted((folder/'runs').glob('*.json')):
            r=read(p);rows=load_rounds(r);check=audit(r,rows,anchor)
            checks.append(dict(group=g['key'],run=p.stem,**check))
            if 'action' in r:runs[r['target'],r['action']['id'],r['repeat']]=r
        expected=3+REPEATS*sum(len(t['actions']) for t in selected['states'])
        assert len(list((folder/'runs').glob('*.json')))==expected
        for target in selected['states']:
            index=target['round']
            def runset(aid):return [runs[index,aid,r] for r in range(REPEATS)]
            for contrast in target['contrasts']:
                left,right=runset(contrast['left']),runset(contrast['right'])
                lv,rv=[[r['label']['area_per_request_ms'] for r in rs] for rs in (left,right)]
                gains=[x-y for x,y in zip(lv,rv)];noise=max(1.,max(lv)-min(lv),max(rv)-min(rv))
                positive=all(g>noise for g in gains);negative=all(g< -noise for g in gains)
                la,ra=left[0]['action'],right[0]['action']
                def mean_delta(field):return st.fmean(l['label'][field]-r['label'][field] for l,r in zip(left,right))
                row=dict(group=g['key'],source=g['source'],window=g['window'],base_T=g['budget'],base_K=g['chunk'],
                    round=index,**contrast,actual_current=target['actual_current'],n=target['threshold'],
                    base_growth=target['base_growth'],free=target['free'],left_parameters=[la['budget'],la['chunk']],
                    right_parameters=[ra['budget'],ra['chunk']],left_q=la['q'],right_q=ra['q'],
                    left_growth=la['growth'],right_growth=ra['growth'],left_partial=la['partial'],right_partial=ra['partial'],
                    gains_ms=gains,mean_gain_ms=st.fmean(gains),noise_scale_ms=noise,positive_confirmed=positive,
                    negative_confirmed=negative,first_gain_ms=mean_delta('first_round_ms'),later_gain_ms=mean_delta('later_ms'),
                    preemptions_saved=mean_delta('preemptions'),recomputed_saved=mean_delta('recomputed_tokens'))
                if contrast['kind']=='k_matched':
                    assert la['q']==ra['q'] and la['growth']==ra['growth'] and la['tokens']!=ra['tokens']
                contrasts.append(row)
    holes=[c for c in contrasts if c['kind']=='hole'];ks=[c for c in contrasts if c['kind']=='k_matched']
    covered={(c['source'],c['window']) for c in holes}
    gate=bool(len(holes)>=6 and len(covered)==4 and sum(c['positive_confirmed'] for c in holes)>=.75*len(holes)
        and not any(c['negative_confirmed'] for c in holes)
        and all(st.fmean(c['mean_gain_ms'] for c in holes if c['source']==s)>0 for s in ('code','conversation')))
    summary=dict(accepted_groups=len(coverage),excluded=excluded,contrasts=contrasts,coverage=coverage,
        hole=dict(count=len(holes),positive=sum(c['positive_confirmed'] for c in holes),negative=sum(c['negative_confirmed'] for c in holes)),
        k_matched=dict(count=len(ks),small_better=sum(c['positive_confirmed'] for c in ks),large_better=sum(c['negative_confirmed'] for c in ks)),
        closed_loop_gate_passed=gate)
    write(OUT/'analysis/summary.json',summary)
    write(OUT/'analysis/audit.json',dict(passed=True,episodes=len(checks),rounds=sum(c['rounds'] for c in checks),checks=checks,excluded=excluded))
    if contrasts:
        fields=list(dict.fromkeys(k for c in contrasts for k in c))
        with (OUT/'analysis/contrasts.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(contrasts)
    lines=['# 新窗口二维候选机制复核','',f'有效组 {len(coverage)}/{len(d["groups"])}；审计 {len(checks)} 次完整运行、{sum(c["rounds"] for c in checks)} 轮。',
        '',f'门槛缺口 {len(holes)} 个：正向复现 {summary["hole"]["positive"]}，反向复现 {summary["hole"]["negative"]}。',
        f'匹配 q/growth 的 K 对照 {len(ks)} 个：小 K 较好 {summary["k_matched"]["small_better"]}，大 K 较好 {summary["k_matched"]["large_better"]}。',
        f'预注册闭环门控：{gate}。',
        '', '收益单位为分支起点剩余首 token 等待面积/64，ms；hole 正值表示 inverse 改善，K 正值表示小 K 改善。',
        '参数是该计划的执行别名；真正 current 参数另见 contrasts.csv 的 actual_current。跨基础配置 anchor 不同。',
        '', '|组|类型/轮|左→右参数|q|growth|四次收益 ms|当轮/后续均值|噪声尺度|方向复现|',
        '|---|---|---|---|---|---|---|---:|---|']
    for c in contrasts:
        direction='正' if c['positive_confirmed'] else '负' if c['negative_confirmed'] else '未过'
        lines.append(f"|{c['group']}|{c['kind']}/{c['round']}|{c['left_parameters']}→{c['right_parameters']}|{c['left_q']}→{c['right_q']}|{c['left_growth']}→{c['right_growth']}|{' / '.join(f'{v:+.3f}' for v in c['gains_ms'])}|{c['first_gain_ms']:+.3f}/{c['later_gain_ms']:+.3f}|{c['noise_scale_ms']:.3f}|{direction}|")
    lines+=['','复现是预先固定的保守方向与噪声筛选，不是显著性检验。没有从本轮标签选状态、选动作或选赢家。',
        '四个新请求窗口仍来自同一周轨迹，只有 burst_stop、一个模型；单步收益不能当作闭环策略 TTFT 收益。',
        f'排除：{excluded}。缺失机会保留在 coverage 中，不用其他状态补齐。']
    (OUT/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n')
    print({k:v for k,v in summary.items() if k not in ('contrasts','coverage')},flush=True)

if __name__=='__main__':main()
