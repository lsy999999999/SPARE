"""Compact tables from saved observations; no policy selection or hidden filtering."""
from collections import defaultdict
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[2]
CPU = ROOT/'artifacts/v1_direction_probe_20260923_r1'
GPU = ROOT/'artifacts/v1_direction_probe_20260923_gpu_final'
SHADOW = ROOT/'artifacts/v1_direction_probe_20260923_shadow_r1'
LABELS = {'a_any_positive_saving':'1-block', 'fixed_small':'固定512',
          'pressure_one':'pressure_one', 'pressure_early':'单轮1/8',
          'pressure_min_delay':'单轮+最小延后', 'horizon4_first':'4轮+原顺序',
          'horizon8_first':'8轮+原顺序', 'horizon4_min_delay':'4轮+最小延后',
          'horizon4_risk':'4轮+风险排序', 'horizon8_risk':'8轮+风险排序'}


def read(path):
    return json.loads(path.read_text())


def main():
    cpu = read(CPU/'summary.json')
    lines = ['# 快速 CPU 方向实验', '',
             '40/40 完成；旧方法12点动作哈希对齐上一轮 CPU；逐轮原生分配断言和账目核对通过。', '',
             '所有请求同时到达。首 token 指标是单位调度轮数，不能作为 GPU TTFT 排名；没有毫秒模型。', '',
             '| 窗口 | KV | 方法 | 首token平均轮数 | 抢占 | 重算tokens | 决策均值μs |',
             '|---|---:|---|---:|---:|---:|---:|']
    for r in cpu['rows']:
        s = r['stats']
        lines.append(f"| {r['block']} | {r['capacity']} | {LABELS[r['method']]} | {s['mean_first_token_round']:.2f} | {s['preemptions']} | {s['recomputed_tokens']} | {s['decision_us_mean']:.1f} |")
    lines += ['', '## 同状态信号对比', '',
              '在 pressure_one 轨迹上只计算其他信号、不改变动作；动作哈希与原轨迹一致。下表限基础计划正常且本轮 G₀>0 的状态。', '',
              '| 窗口 | KV | 4轮与1/4门槛一致率 | 8轮与1/8门槛一致率 |', '|---|---:|---:|---:|']
    for r in read(SHADOW/'summary.json')['rows']:
        lines.append(f"| {r['block']} | {r['capacity']} | {r['h4']['actionable_same_trigger_fraction']:.2%} | {r['h8']['actionable_same_trigger_fraction']:.2%} |")
    (CPU/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n')
    paths = sorted((GPU/'runs').glob('*.json'))
    if not paths:
        return
    rows = [read(p) for p in paths]
    groups = defaultdict(list)
    for r in rows:
        if r['job']['phase']=='compare':
            groups[r['job']['cell']['capacity'],r['job']['method']].append(r)
    stats = {}
    for key, group in groups.items():
        if any(r['status']!='complete' for r in group):
            continue
        stats[key] = {name:st.fmean(r['metrics'][name] for r in group) for name in
            ('mean_ttft_ms','mean_e2e_ms','p95_tbt_ms','preemptions','recomputed_tokens',
             'decision_ms_mean','decision_ms_total','interventions','deferred_prefill_tokens')}
        xs = [r['metrics']['mean_ttft_ms'] for r in group]
        stats[key].update(repeats=len(group), spread=(max(xs)-min(xs))/st.fmean(xs),
                          decision_fraction=st.fmean(r['metrics']['decision_ms_total']/r['elapsed_ms'] for r in group))
    lines = ['# 快速 GPU 方向实验', '',
        f"共 {len(rows)} 次有效运行；正式比较 {sum(len(g) for g in groups.values())}/28，另6次A/A。只用 code_test0 两个容量。不是独立确认。", '',
        '2048两遍同卡、顺序反转；768初轮受外来进程干扰后整批排除，补跑在GPU6/7各完成全部方法，各卡内配对，两个顺序相反。768的两次极差同时包含跨卡差异。', '',
        '| KV | 方法 | 重复 | mean TTFT ms | 对1-block改善 | 对固定512改善 | TTFT重复极差/均值 | 抢占 | 重算tokens |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for cap in (768,2048):
        for method in read(GPU/'manifest.json')['methods']:
            if (cap,method) not in stats:
                continue
            s = stats[cap,method]
            gains = [f"{1-s['mean_ttft_ms']/stats[cap,b]['mean_ttft_ms']:+.1%}" if (cap,b) in stats else '—'
                     for b in ('a_any_positive_saving','fixed_small')]
            lines.append(f"| {cap} | {LABELS[method]} | {s['repeats']} | {s['mean_ttft_ms']:.1f} | {gains[0]} | {gains[1]} | {s['spread']:.2%} | {s['preemptions']:.1f} | {s['recomputed_tokens']:.0f} |")
    lines += ['', '## 代价与行为（不作为本轮优化约束）', '',
              '| KV | 方法 | E2E ms | p95 TBT ms | 决策均值ms | 决策时间/总时长 | 干预次数 | 延后prefill tokens |',
              '|---:|---|---:|---:|---:|---:|---:|---:|']
    for (cap,method),s in sorted(stats.items()):
        lines.append(f"| {cap} | {LABELS[method]} | {s['mean_e2e_ms']:.1f} | {s['p95_tbt_ms']:.1f} | {s['decision_ms_mean']:.3f} | {s['decision_fraction']:.2%} | {s['interventions']:.1f} | {s['deferred_prefill_tokens']:.0f} |")
    lines += ['', 'p95 TBT 是两次运行各自 p95 的平均；不是合并所有 token 后的 p95。小于运行波动的差异不作排名依据。',
              '所有方法的调度与诊断开销都进入端到端时钟。这里不消除新规则的计算成本。']
    lines += ['', '## 各技术重复内的配对收益', '',
              '| KV | GPU | 重复 | 方法 | 对同组1-block改善 | 对同组固定512改善 |', '|---:|---:|---:|---|---:|---:|']
    paired = {}
    for r in rows:
        if r['job']['phase']!='compare': continue
        j=r['job'];rep=int(j['key'].rsplit('_r',1)[1])
        paired[j['cell']['capacity'],j['cell']['gpu'],rep,j['method']]=r['metrics']['mean_ttft_ms']
    for (cap,gpu,rep,method),ttft in sorted(paired.items()):
        gains=[1-ttft/paired[cap,gpu,rep,b] for b in ('a_any_positive_saving','fixed_small')]
        lines.append(f'| {cap} | {gpu} | {rep} | {LABELS[method]} | {gains[0]:+.1%} | {gains[1]:+.1%} |')
    (GPU/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n')
    (GPU/'aggregated.json').write_text(json.dumps([dict(capacity=k[0],method=k[1],**v) for k,v in stats.items()],indent=2)+'\n')


if __name__ == '__main__': main()
