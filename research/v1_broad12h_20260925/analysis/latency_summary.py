"""Post-hoc latency summaries from completed, accepted groups; no GPU reruns."""
import csv
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/v1_broad12h_20260925_r1'
ARMS = ['old', 'arrival', 'combined', 'pressure_decode', 'pressure_constant']
NAMES = dict(zip(ARMS, ['旧式', '① 旧式＋A', '② 原D＋A', '③ 压力衰减D＋A', '④ 压力常数＋A']))
METRICS = ['mean_ttft_ms', 'p95_ttft_ms', 'mean_tbt_ms', 'p95_tbt_ms',
           'mean_e2e_ms', 'p95_e2e_ms', 'throughput_tokens_s']


def quantile(values, q):
    xs = sorted(values)
    p = (len(xs) - 1) * q
    i = int(p)
    return xs[i] + (xs[min(i + 1, len(xs) - 1)] - xs[i]) * (p - i)


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    cells = []
    for path in sorted((OUT / 'groups').glob('*.json')):
        group = json.loads(path.read_text())
        assert group['state'] == 'complete'
        folder = OUT / group['folder']
        assert not (folder / 'contamination.json').exists()
        arms = {arm: [] for arm in ARMS}
        for run_path in sorted((folder / 'runs').glob('*.json')):
            run = json.loads(run_path.read_text())
            assert run['status'] == 'complete'
            metrics = dict(run['metrics'])
            clocks = list(run['clocks'].values())
            assert len(clocks) == run['request_count'] == run['completed']
            ttft = [c['first_ms'] - c['arrival_ms'] for c in clocks]
            tbt = [b - a for c in clocks for a, b in
                   zip(c['token_times_ms'], c['token_times_ms'][1:])]
            e2e = [c['finish_ms'] - c['arrival_ms'] for c in clocks]
            for label, values in [('ttft', ttft), ('tbt', tbt), ('e2e', e2e)]:
                assert values
                assert math.isclose(metrics[f'mean_{label}_ms'], st.fmean(values), rel_tol=1e-10)
                p95 = quantile(values, .95)
                if f'p95_{label}_ms' in metrics:
                    assert math.isclose(metrics[f'p95_{label}_ms'], p95, rel_tol=1e-10)
                metrics[f'p95_{label}_ms'] = p95
            arms[run['job']['arm']].append((run['job']['repeat'], metrics))
        assert all(len(rs) == 2 and {r[0] for r in rs} == {0, 1} for rs in arms.values())
        means = {arm: {key: st.fmean(m[key] for _, m in runs) for key in METRICS}
                 for arm, runs in arms.items()}
        for arm in ARMS:
            row = dict(group=group['group'], **group['cell'], arm=arm, **means[arm])
            for key in METRICS:
                ratio = means[arm][key] / means['old'][key]
                row[key + '_gain_pct'] = 100 * (ratio - 1 if key == 'throughput_tokens_s' else 1 - ratio)
            cells.append(row)
    assert len(cells) == 176 * 5
    summary = []
    for family, count in [('measured_lengths', 160), ('synthetic', 16)]:
        for arm in ARMS:
            rows = [r for r in cells if r['family'] == family and r['arm'] == arm]
            assert len(rows) == count
            summary.append(dict(family=family, arm=arm, cells=len(rows),
                                **{key: st.fmean(r[key + '_gain_pct'] for r in rows) for key in METRICS}))
    write_csv(OUT / 'analysis/latency_cells.csv', cells)
    write_csv(OUT / 'analysis/latency_aggregates.csv', summary)
    lines = ['# 完整延迟指标补充分析', '',
             '本次仅重算已完成的176组正式GPU记录，没有新增实验或改变主指标。', '',
             '正值表示改善：延迟降低、吞吐提高。每场景先平均两次重复，再计算相对旧式变化，最后等权平均场景；不是合并所有请求后计算全局p95。', '',
             'TTFT和端到端延迟分别为首token/完成时间减计划到达时间。TBT按单次运行所有请求的相邻输出token间隔汇总，长输出请求贡献更多间隔；不包含首token等待。', '',
             'p95采用与原实验相同的线性插值分位数。p95端到端从原始clocks补算；其余延迟指标均由clocks重新计算并与原记录核对通过（1760次正式运行）。', '',
             '以下为辅助描述统计，不据小幅差异声称统计显著或更改mean TTFT主目标。各场景每次仅64个实测来源请求或48个合成请求，尾部分位数的稳定性有限。', '']
    for family, title in [('measured_lengths', '160个实测来源场景'), ('synthetic', '16个合成场景')]:
        lines += ['## ' + title, '',
                  '| 方法 | mean TTFT | p95 TTFT | mean TBT | p95 TBT | mean E2E | p95 E2E | 吞吐 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in summary:
            if row['family'] == family and row['arm'] != 'old':
                lines.append('| ' + NAMES[row['arm']] + ' | ' + ' | '.join(f'{row[k]:+.2f}%' for k in METRICS) + ' |')
        lines.append('')
    lines += ['原始数值及逐场景变化见 artifacts/v1_broad12h_20260925_r1/analysis/latency_cells.csv；汇总见同目录latency_aggregates.csv。', '']
    report = ROOT / 'research/v1_broad12h_20260925/LATENCY_METRICS_CN.md'
    report.write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
