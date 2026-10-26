"""Analyze all paired runs, validation-selected weights, and exploratory fronts."""
from collections import defaultdict
import csv
import os
import numpy as np
from .run import ROOT, OUT, ARMS, BUDGETS, CAPACITIES, read, write, digest

LABELS = {'arrival': 'SPARE', 'srp': 'SRP (fixed budget)', 'native': 'Native V1',
          'sarathi': 'Sarathi rule', 'past_future': 'Past-Future rule'}
COLORS = dict(zip(ARMS, ['#0072B2', '#D55E00', '#777777', '#009E73', '#CC79A7']))


def save_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def dominates(a, b):
    # Both metrics ordered as lower is better.
    a, b = np.asarray(a), np.asarray(b)
    tol = np.maximum(np.abs(b), 1) * 1e-10
    return bool(np.all(a <= b + tol) and np.any(a < b - tol))


def ci(values):
    return np.quantile(values, [.025, .975]).tolist()


def analyze():
    design = read(OUT / 'design.json')
    for rel, expected in design['hashes'].items():
        assert digest(ROOT / rel) == expected, rel
    raw = []
    for sample in design['samples']:
        for cap in CAPACITIES:
            part = read(OUT / 'runs' / f"{sample['name']}_c{cap}.json")
            assert len(part) == len(ARMS) * len(BUDGETS)
            assert {(r['arm'], r['budget']) for r in part} == {(a, b) for a in ARMS for b in BUDGETS}
            raw.extend(part)
    assert len(raw) == design['expected_runs'] == 5600
    assert all(r['complete'] and r['metrics']['completed'] == 64 for r in raw)
    flat = [dict(source=r['source'], day=r['day'], name=r['name'], split=r['split'],
                 capacity=r['capacity'], arm=r['arm'], budget=r['budget'],
                 ttft_s=r['metrics']['ttft_ms_mean'] / 1000,
                 output_tps=r['metrics']['output_throughput_tps'],
                 request_rps=r['metrics']['request_throughput_rps'],
                 interventions=r['metrics']['interventions'], preemptions=r['metrics']['preemptions'],
                 recomputed_tokens=r['metrics']['recomputed_tokens'],
                 trajectory_sha256=r['trajectory_sha256']) for r in raw]
    save_csv('runs.csv', flat)
    lookup = {(r['source'], r['capacity'], r['split'], r['arm'], r['budget'], r['name']): r for r in flat}
    assert len(lookup) == len(flat)
    points, pairs, selected, verdicts, weights, arrays = [], [], [], [], [], {}
    rng = np.random.default_rng(20261014)
    for source in ('code', 'conversation'):
        for cap in CAPACITIES:
            ids = {split: sorted(s['name'] for s in design['samples']
                                 if s['source'] == source and s['split'] == split)
                   for split in ('validation', 'test')}
            assert len(ids['validation']) == 20 and len(ids['test']) == 50
            day_indices = defaultdict(list)
            for i, name in enumerate(ids['test']):
                day_indices[lookup[source, cap, 'test', 'arrival', 1024, name]['day']].append(i)
            assert sorted(map(len, day_indices.values())) == [25, 25]
            boot = np.concatenate([rng.choice(indices, (5000, len(indices)))
                                   for indices in day_indices.values()], axis=1)
            data = {}
            for split in ids:
                for arm in ARMS:
                    for budget in BUDGETS:
                        rows = [lookup[source, cap, split, arm, budget, name] for name in ids[split]]
                        data[split, arm, budget] = np.array([[r['ttft_s'], r['output_tps']] for r in rows])
            means = {key: value.mean(axis=0) for key, value in data.items() if key[0] == 'test'}
            for arm in ARMS:
                for budget in BUDGETS:
                    arr = data['test', arm, budget]
                    mean = means['test', arm, budget]
                    intervals = np.quantile(arr[boot].mean(axis=1), [.025, .975], axis=0)
                    rivals = [f'{a}:{b}' for (_, a, b), v in means.items()
                              if dominates([v[0], -v[1]], [mean[0], -mean[1]])]
                    points.append(dict(source=source, capacity=cap, arm=arm, budget=budget, n=50,
                                       ttft_s=mean[0], output_tps=mean[1],
                                       ttft_ci_low=intervals[0, 0], ttft_ci_high=intervals[1, 0],
                                       output_ci_low=intervals[0, 1], output_ci_high=intervals[1, 1],
                                       nondominated=not rivals, dominated_by=';'.join(rivals)))
            for spare_budget in BUDGETS:
                ours = data['test', 'arrival', spare_budget]
                for arm in ARMS[1:]:
                    for budget in BUDGETS:
                        other = data['test', arm, budget]
                        # Positive means the comparator improves on SPARE.
                        dl = ours[:, 0] - other[:, 0]
                        dq = other[:, 1] - ours[:, 1]
                        lci, qci = ci(dl[boot].mean(axis=1)), ci(dq[boot].mean(axis=1))
                        pairs.append(dict(source=source, capacity=cap, spare_budget=spare_budget,
                                          baseline=arm, baseline_budget=budget,
                                          baseline_ttft_advantage_s=float(dl.mean()),
                                          ttft_ci_low=lci[0], ttft_ci_high=lci[1],
                                          baseline_output_advantage_tps=float(dq.mean()),
                                          output_ci_low=qci[0], output_ci_high=qci[1],
                                          baseline_dominates_mean=dominates(
                                              [other[:, 0].mean(), -other[:, 1].mean()],
                                              [ours[:, 0].mean(), -ours[:, 1].mean()]),
                                          baseline_dominates_window_fraction=float(np.mean(
                                              (dl >= 0) & (dq >= 0) & ((dl > 0) | (dq > 0))))))
            for w in np.linspace(0, 1, 11):
                scores = {}
                for arm in ARMS:
                    train_ref = data['validation', 'native', 1024]
                    def score(arr, ref):
                        return w * arr[:, 0] / ref[:, 0] + (1-w) * ref[:, 1] / arr[:, 1]
                    budget = min(BUDGETS, key=lambda b: (float(score(data['validation', arm, b], train_ref).mean()), b))
                    test = score(data['test', arm, budget], data['test', 'native', 1024])
                    mean = data['test', arm, budget].mean(axis=0)
                    interval = ci(test[boot].mean(axis=1))
                    row = dict(source=source, capacity=cap, w=round(float(w), 1), arm=arm,
                               selected_budget=budget,
                               validation_score=float(score(data['validation', arm, budget], train_ref).mean()),
                               test_score=float(test.mean()), score_ci_low=interval[0], score_ci_high=interval[1],
                               test_ttft_s=float(mean[0]), test_output_tps=float(mean[1]))
                    selected.append(row)
                    scores[arm] = row
                best = min(r['test_score'] for r in scores.values())
                winners = [a for a, r in scores.items() if abs(r['test_score']-best) < 1e-10]
                weights.append(dict(source=source, capacity=cap, w=round(float(w), 1),
                                    test_winners=';'.join(winners), spare_selected_budget=scores['arrival']['selected_budget'],
                                    spare_is_best='arrival' in winners,
                                    spare_relative_score_gap_pct=100*(scores['arrival']['test_score']/best-1)))
            ps = [p for p in points if p['source'] == source and p['capacity'] == cap]
            ours = [p for p in ps if p['arm'] == 'arrival']
            bases = [p for p in ps if p['arm'] != 'arrival']
            covered = [p for p in bases if any(dominates([o['ttft_s'], -o['output_tps']],
                                                       [p['ttft_s'], -p['output_tps']]) for o in ours)]
            default = next(p for p in ours if p['budget'] == 1024)
            verdicts.append(dict(source=source, capacity=cap,
                                 spare_nondominated_budgets=[p['budget'] for p in ours if p['nondominated']],
                                 default_dominated_by=default['dominated_by'],
                                 baseline_points_covered=len(covered), baseline_points_total=len(bases),
                                 weighted_test_wins=sum(r['spare_is_best'] for r in weights
                                                        if r['source'] == source and r['capacity'] == cap)))
            arrays[source, cap] = ps
    save_csv('points.csv', points)
    save_csv('paired_comparisons.csv', pairs)
    save_csv('selected_weights.csv', selected)
    save_csv('weight_winners.csv', weights)
    write(OUT / 'summary.json', dict(completed=len(flat), expected=5600,
                                    validation_runs=1600, test_runs=4000, verdicts=verdicts,
                                    uncertainty='Paired window bootstrap stratified by two fixed test dates; '
                                                'exploratory pointwise 95% intervals, no multiplicity adjustment.',
                                    performance_validated=False))
    plot(points, selected)
    report(points, pairs, verdicts, weights)
    print(__import__('json').dumps(verdicts, ensure_ascii=False, indent=2))


def plot(points, selected):
    os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig2, axes2 = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row, source in enumerate(('code', 'conversation')):
        for col, cap in enumerate(CAPACITIES):
            ax, ax2 = axes[row, col], axes2[row, col]
            for arm in ARMS:
                ps = [p for p in points if (p['source'], p['capacity'], p['arm']) == (source, cap, arm)]
                for p in ps:
                    ax.errorbar(p['ttft_s'], p['output_tps'],
                                xerr=[[p['ttft_s']-p['ttft_ci_low']], [p['ttft_ci_high']-p['ttft_s']]],
                                yerr=[[p['output_tps']-p['output_ci_low']], [p['output_ci_high']-p['output_tps']]],
                                color=COLORS[arm], alpha=.2, linewidth=.6, zorder=1)
                ax.scatter([p['ttft_s'] for p in ps], [p['output_tps'] for p in ps],
                           color=COLORS[arm], marker='*' if arm == 'arrival' else 'o',
                           s=105 if arm == 'arrival' else 27, label=LABELS[arm], zorder=3)
                front = [p for p in ps if not any(dominates([q['ttft_s'], -q['output_tps']],
                                                           [p['ttft_s'], -p['output_tps']]) for q in ps)]
                front.sort(key=lambda p: p['ttft_s'])
                ax.plot([p['ttft_s'] for p in front], [p['output_tps'] for p in front],
                        color=COLORS[arm], linewidth=1, alpha=.7)
                if arm == 'arrival':
                    for p in ps:
                        offset = (-24, -13) if source == 'code' and cap == 4096 and p['budget'] == 512 else (4, 5)
                        ax.annotate(str(p['budget']), (p['ttft_s'], p['output_tps']),
                                    xytext=offset, textcoords='offset points', fontsize=8, color=COLORS[arm])
                ss = [p for p in selected if (p['source'], p['capacity'], p['arm']) == (source, cap, arm)]
                ax2.plot([p['w'] for p in ss], [p['test_score'] for p in ss],
                         color=COLORS[arm], marker='.' if arm != 'arrival' else '*', label=LABELS[arm])
            for axis in (ax, ax2):
                axis.set_title(f'{source.title()} | KV {cap} blocks')
                axis.grid(alpha=.13)
            ax.set_xlabel('Mean TTFT (s) — lower is better')
            ax.set_ylabel('Output tokens/s — higher is better')
            ax2.set_xlabel('Latency weight w')
            ax2.set_ylabel('Test objective J(w) — lower is better')
    axes[0, 0].legend(fontsize=8, loc='best')
    axes2[0, 0].legend(fontsize=8, loc='best')
    fig.suptitle('CPU simulation: 50 paired test windows per point\n64 requests/window; 5.25 arrivals/s; historical fitted clock', fontsize=12)
    fig2.suptitle('Budgets selected on validation windows, evaluated on separate test dates', fontsize=12)
    for f, name in ((fig, 'pareto'), (fig2, 'weighted_objective')):
        f.savefig(OUT / f'{name}.png', dpi=180)
        f.savefig(OUT / f'{name}.pdf')
        plt.close(f)


def report(points, pairs, verdicts, weights):
    lines = ['# 第 1 点模拟验证结果', '',
             '完成 **5600/5600** 条 CPU simulator 轨迹：1600 条选参、4000 条测试，无失败。'
             '每个测试配置50个不同请求窗口；未加载模型、未运行GPU推理。', '',
             '**判断：可以保留双指标实验，但当前证据支持“调参后处于经验前沿”，不支持默认版本全面支配对照。** '
             '四个场景均有SPARE配置在测试均值前沿；仅Code/KV768覆盖全部16个对照配置。'
             '默认T1024在三个场景（Code768、Conversation768/4096）被固定预算SRP同时改善两项指标。', '',
             '44个场景×权重点中，验证集选参后的SPARE在测试均值上30次单独最优、10次与SRP并列、4次落后Past-Future。'
             'Code/KV4096的SPARE T512与SRP T512完全相同，Conversation/KV4096的同预算差异也很小；'
             '这些点不能作为动态门槛有明显增量收益的证据。', '',
             '本轮大多数权重选择同一个预算，尚未形成丰富的可调权衡曲线。建议先保留预算256/512的强对照，'
             '下一步检验预算/门槛是否需要随KV压力调整，再决定是否引入在线双目标决策。', '',
             '## Pareto 判断', '',
             '以下按50窗口的测试均值判断，比较集合为五个方法族各四档预算。'
             '“前沿预算”指在全部20个配置中不被支配的SPARE配置；覆盖数指被至少一个SPARE配置支配的对照配置数。', '',
             '|负载|KV块|SPARE前沿预算|默认T1024被哪些配置支配|覆盖对照点|加权目标最优次数（含并列）|',
             '|---|---:|---|---|---:|---:|']
    for v in verdicts:
        lines.append(f"|{v['source']}|{v['capacity']}|{v['spare_nondominated_budgets'] or '无'}|"
                     f"{v['default_dominated_by'] or '无'}|{v['baseline_points_covered']}/16|{v['weighted_test_wins']}/11|")
    lines += ['', '## 默认SPARE与各对照：测试均值', '',
              '每格为mean TTFT（s）/输出token每秒。所有方法保持K=256；完整四预算见points.csv。', '',
              '|负载|KV块|方法|T256|T512|T1024|T2048|', '|---|---:|---|---:|---:|---:|---:|']
    for source in ('code', 'conversation'):
        for cap in CAPACITIES:
            for arm in ARMS:
                ps = sorted([p for p in points if (p['source'], p['capacity'], p['arm']) == (source, cap, arm)], key=lambda p: p['budget'])
                lines.append(f'|{source}|{cap}|{LABELS[arm]}|' + '|'.join(f"{p['ttft_s']:.3f} / {p['output_tps']:.2f}" for p in ps) + '|')
    lines += ['', '## 默认SPARE被支配时的配对差异', '',
              '正数表示对照更好；95%区间为日期内窗口配对bootstrap，逐比较报告，未作多重校正。完整比较见paired_comparisons.csv。', '',
              '|负载|KV块|对照|对照TTFT优势（s）及95%区间|对照吞吐优势（token/s）及95%区间|',
              '|---|---:|---|---|---|']
    for p in pairs:
        if p['spare_budget'] == 1024 and p['baseline_dominates_mean']:
            lines.append(f"|{p['source']}|{p['capacity']}|{p['baseline']} T{p['baseline_budget']}|"
                         f"{p['baseline_ttft_advantage_s']:.3f} [{p['ttft_ci_low']:.3f}, {p['ttft_ci_high']:.3f}]|"
                         f"{p['baseline_output_advantage_tps']:.2f} [{p['output_ci_low']:.2f}, {p['output_ci_high']:.2f}]|")
    lines += ['', '## 评价口径', '',
              '- 每个来源/容量各20个选参窗口和50个测试窗口，选参日与测试日分开；每窗口64请求，5.25 req/s，容量768/4096块。',
              '- 五方法均扫描T=256/512/1024/2048，K=256；SRP即同排序固定预算对照。SPARE默认T1024。',
              '- w=0,.1,...,1；J=w L/L0+(1-w)Q0/Q。参考为相同窗口native T1024。逐窗口归一化，在选参集选预算，再报告测试分数。',
              '- Pareto图公开全部配置；权重图的配置完全由选参集确定。测试均值前沿属于探索性结果，不等于证明全参数空间最优。',
              '- 时间模型为既有GPU日志的回顾性拟合；本次只做CPU模拟。历史留出评估中逐请求TTFT MAPE约14.55%，不能把小幅模拟优势直接当作实机收益。',
              '- 95%区间固定两个测试日期的权重，在日期内部配对重采样窗口5000次；不代表跨日期不确定性。到达模板固定，未进行到达率扫描。',
              '- Q包含排空阶段，衡量有限工作负载完成效率；本实验未测最大可持续请求率。输出≤512、总长≤8000为预设输入条件。',
              '- 运行均检查全部完成、输出账目、KV归还与执行/重算守恒；源码和输入哈希前后核对通过。', '',
              '## 产物', '',
              '- [Pareto图](pareto.png) / [PDF](pareto.pdf)',
              '- [权重与测试目标](weighted_objective.png) / [PDF](weighted_objective.pdf)',
              '- runs.csv：全部5600条指标；points.csv：80个测试配置均值及区间。',
              '- selected_weights.csv：每个w的验证集选择和测试结果；weight_winners.csv：测试最优方法。',
              '- paired_comparisons.csv：全部SPARE配置与对照的配对差异。',
              '- design.json、inputs/、source/、runs/：冻结设计、输入、源码快照、原始逐运行结果。', '']
    (OUT / 'REPORT_CN.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    analyze()
