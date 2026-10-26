"""Audit source data and plot the fixed-seed TTFT-limit/capacity comparison."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/www_rl7047_frontier_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

SKILL_SCRIPTS = Path.home() / '.codex/skills/nature-figure/scripts'
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/www_rl7047_frontier_20261020_r1'
ANALYSIS = OUT / 'analysis'
PANELS = (('conversation', 2048), ('code', 2048),
          ('conversation', 4096), ('code', 4096))
COLORS = {'spare': '#b64342', 'rl_standalone': '#176c82'}
LABELS = {'spare': 'SPARE', 'rl_standalone': 'RL (seed 7047)'}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def passed(s, rate, limit):
    return (s['failed_samples'] == 0 and s['mean_ttft_ms'] <= limit
            and s['p99_tbt_ms'] <= 200 and s['backlog_drift'] <= max(2, rate))


def load_and_audit():
    design = read(OUT / 'design.json')
    assert read(OUT / 'status.json')['complete']
    groups, points, audit = {}, [], []
    for source, kv in PANELS:
        for arm in LABELS:
            path = OUT / 'groups' / f'{source}_c{kv}_{arm}_seed7047.json'
            group = read(path)
            assert group['stage'] == 'complete'
            assert group['input_sha256'] == design['input_sha256'][source]
            assert group['checkpoint_sha256'] == (design['selected']['sha256'] if arm != 'spare' else None)
            summaries = {}
            window_names = None
            for record in group['records']:
                file = ROOT / record['path']
                assert digest(file) == record['sha256']
                raw = read(file)
                assert len(raw['rows']) == 50
                assert raw['summary'] == record['summary']
                if 'reused_from' in raw:
                    assert digest(ROOT / raw['reused_from']) == raw['reused_sha256']
                names = {r['name'] for r in raw['rows']}
                assert len(names) == 50
                if window_names is None:
                    window_names = names
                assert names == window_names
                valid = [r for r in raw['rows'] if r['valid']]
                assert raw['summary']['failed_samples'] == 50-len(valid)
                for key in ('mean_ttft_ms', 'p99_tbt_ms', 'backlog_drift'):
                    expected = fmean(r[key] for r in valid)
                    assert math.isclose(expected, raw['summary'][key], abs_tol=1e-7, rel_tol=1e-10)
                summaries[record['rate_qps']] = raw['summary']
            previous = -1
            assert [p['mean_ttft_limit_ms'] for p in group['thresholds']] == design['limits_ms']
            for point in group['thresholds']:
                limit = point['mean_ttft_limit_ms']
                low = None
                for rate in sorted(summaries):
                    if not passed(summaries[rate], rate, limit):
                        high = rate
                        break
                    low = rate
                assert low == point['capacity_qps'] and high == point['next_failed_qps']
                if low is not None:
                    assert 0 < high-low <= .020001
                    assert low >= previous
                    previous = low
                else:
                    assert high == .25 and point['below_minimum_measured_rate']
                points.append(dict(source=source, kv_blocks=kv, method=LABELS[arm],
                    arm=arm, seed=7047 if arm != 'spare' else '',
                    mean_ttft_limit_s=limit/1000, capacity_qps=low,
                    next_failed_qps=high, resolution_qps=point['resolution_qps'],
                    attained_mean_ttft_s=(point['attained_mean_ttft_ms']/1000
                                          if point['attained_mean_ttft_ms'] is not None else None),
                    below_minimum_measured_rate=point['below_minimum_measured_rate'],
                    first_failure_reasons=';'.join(point['first_failure_reasons'])))
            original = read(ROOT / 'artifacts/www_rl_standalone_v2_20261019_r1/groups' / path.name)
            at2 = next(p for p in group['thresholds'] if p['mean_ttft_limit_ms']==2000)
            audit.append(dict(source=source, kv_blocks=kv, arm=arm,
                rate_records=len(summaries), raw_window_records=len(summaries)*50,
                original_capacity_2s=original['capacity_qps'], new_capacity_2s=at2['capacity_qps'],
                difference_2s=round(at2['capacity_qps']-original['capacity_qps'], 4),
                all_boundaries_verified=True))
            groups[source, kv, arm] = group
    for source, kv in PANELS:
        assert groups[source,kv,'spare']['input_sha256'] == groups[source,kv,'rl_standalone']['input_sha256']
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS/'data_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    with (ANALYSIS/'capacity_points.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    return groups, points, audit


def plot(points):
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans'],
        'font.size':8, 'axes.titlesize':8.5, 'axes.labelsize':8,
        'xtick.labelsize':7, 'ytick.labelsize':7, 'legend.fontsize':8,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.7,
        'pdf.fonttype':42, 'svg.fonttype':'none', 'savefig.dpi':300})
    fig, axes = plt.subplots(2,2,figsize=(183/25.4,132/25.4),sharex=True,sharey=True)
    fig.subplots_adjust(left=.09,right=.98,bottom=.16,top=.90,hspace=.38,wspace=.17)
    for letter, ax, (source,kv) in zip('abcd',axes.flat,PANELS):
        for arm in LABELS:
            selected = [p for p in points if (p['source'],p['kv_blocks'],p['arm'])==(source,kv,arm)]
            xs = [p['mean_ttft_limit_s'] for p in selected]
            ys = [float('nan') if p['capacity_qps'] is None else p['capacity_qps'] for p in selected]
            assert all(x>0 for x in xs)
            assert xs == sorted(xs)
            ax.plot(xs,ys,color=COLORS[arm],marker='o' if arm=='spare' else 's',
                linestyle='-' if arm=='spare' else '--',linewidth=1.4,
                markersize=3.3,markeredgewidth=.75,markerfacecolor='white',label=LABELS[arm])
        ax.set_xscale('log')
        ax.set_xlim(.23,8.7)
        ax.set_xticks((.25,.5,1,2,4,8), labels=('0.25','0.5','1','2','4','8'))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(0,6)
        ax.set_yticks((0,1,2,3,4,5,6))
        ax.grid(axis='y',color='#e4e7e9',linewidth=.55)
        ax.set_axisbelow(True)
        ax.set_title(f'{source.capitalize()} | KV = {kv}',pad=9)
        ax.text(-.13,1.08,letter,transform=ax.transAxes,ha='left',va='bottom',
                fontsize=9,fontweight='bold')
    for ax in axes[1]:
        ax.set_xlabel('Mean TTFT limit (s)')
    for ax in axes[:,0]:
        ax.set_ylabel('Serving capacity (QPS)')
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.00),ncol=2,
               frameon=False,handlelength=2.4,columnspacing=2.5)
    fig.text(.5,.025,'50 paired windows per condition; fixed RL checkpoint; simulator estimates',
             ha='center',va='bottom',fontsize=7,color='#4d5961')
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=list(axes.flat), panel_ids=list('abcd'),
        json_out=ANALYSIS/'rl7047_spare_capacity.alignment.json')
    for suffix in ('pdf','svg','png'):
        fig.savefig(ANALYSIS/f'rl7047_spare_capacity.{suffix}',dpi=300)
    plt.close(fig)


def report(groups, points, audit):
    status=read(OUT/'status.json')
    lines=['# RL 7047 与 SPARE：固定 KV 的 mean TTFT 上限–容量曲线','',
        '固定独立 Double DQN v2 的 seed 7047、第 256 episode checkpoint；所有上限、负载与 KV 使用同一权重，无重新训练。'
        '该 seed 是原 Conversation/KV=2048 测试容量最高的运行，checkpoint 在验证集上选定。','',
        '## 测量与复用','',
        '每种负载使用相同 50 个测试窗口。横轴是 **mean TTFT 上限**，不是容量点实际测得的 mean TTFT；'
        '实际值另存 CSV 的 attained_mean_ttft_s。纵轴为从低到高连续通过至首次失败之前的最高实测输入 QPS。'
        '另要求窗口 P99 TBT 的均值 ≤200 ms、40–60 s 平均 backlog drift ≤max(2,QPS)，且所有窗口有效。'
        '每个可行容量点的通过/失败间隔 ≤0.02 QPS；这不是置信区间。','',
        '横轴使用对数刻度，展示全部 17 个预先确定的上限：0.25、0.3、0.35、0.4、0.5、0.6、0.75、1、1.25、1.5、1.75、2、2.5、3、4、6、8 秒。'
        '若最低实测 0.25 QPS 已失败，容量记为缺测（CSV 空值），图中不绘制该点；不将其伪装成零容量。'
        '曲线连线仅用于引导阅读，不对未测容量内插。','',
        f'共完成 8 组、136 个上限/方法/配置组合；复用 {status["reused_rates"]} 个逐 QPS 文件，'
        f'新增 {status["new_rates"]} 个逐 QPS 文件，每个文件 50 窗口。'
        '复用前核对输入、模型及时间模型，SPARE 归档重叠点逐窗口核验；8 个当前代码回放校验全部通过。'
        '制图前重新计算所有原始窗口均值，并在最终全部 QPS 记录上核验首次失败和容量边界。','',
        '## 主要观察','',
        'Conversation、KV=2048 的匹配配置中，RL 在已测 0.35–2.5 秒上限下容量高于 SPARE，'
        '相对差为 +2.88% 至 +5.96%；3 秒及以上两者进入积压约束平台，分别为 4.34 与 4.36 QPS。'
        '0.25/0.3 秒上限下 RL 未在最低测试速率取得可行点。','',
        'Conversation、KV=4096 以及 Code、KV=4096 中，双方均有可行容量的全部已测上限下 SPARE 更高。'
        'Code、KV=2048 中 SPARE 在多数上限下更高；0.6 秒处 RL 为 1.50、SPARE 为 1.48 QPS，'
        '这一 0.02 QPS 的反向点估计与扫描分辨率同量级，不作为统计优势。','',
        '更宽松的 TTFT 上限无法消除积压约束造成的平台：KV=4096 下，RL 在 Conversation/Code 的平台为 '
        '4.73/3.42 QPS，SPARE 为 5.54/4.38 QPS。以上为固定模型的条件性表现，不将其概括为所有 RL 的性质。','',
        '## 容量结果','',
        '| 负载 | KV | TTFT 上限 (s) | SPARE QPS | RL 7047 QPS | RL 相对 SPARE |',
        '|---|---:|---:|---:|---:|---:|']
    for source,kv in PANELS:
        for limit in (.5,1,1.5,2,3,8):
            match={p['arm']:p['capacity_qps'] for p in points if
                   (p['source'],p['kv_blocks'],p['mean_ttft_limit_s'])==(source,kv,limit)}
            spare,rl=match['spare'],match['rl_standalone']
            gain=f'{100*(rl/spare-1):+.2f}%' if spare and rl else '—'
            lines.append(f'| {source.capitalize()} | {kv} | {limit:g} | {spare if spare is not None else "—"} | {rl if rl is not None else "—"} | {gain} |')
    lines+=['','## 资源平台与原 2 秒结果核验','',
        '| 负载 | KV | 方法 | 资源最高通过 QPS | 下一失败 QPS | 资源失败原因 |',
        '|---|---:|---|---:|---:|---|']
    for (source,kv,arm),g in groups.items():
        fail=next(r for r in g['records'] if r['rate_qps']==g['resource_first_fail_qps'])
        s=fail['summary']; why=[]
        if s['backlog_drift']>max(2,fail['rate_qps']):why.append('backlog drift')
        if s['p99_tbt_ms']>200:why.append('P99 TBT')
        if s['failed_samples']:why.append('invalid window')
        lines.append(f'| {source.capitalize()} | {kv} | {LABELS[arm]} | {g["resource_last_pass_qps"]} | {g["resource_first_fail_qps"]} | {", ".join(why)} |')
    differences=[a for a in audit if a['difference_2s']]
    lines+=['', '原 2 秒容量'+('全部复现。' if not differences else '有变化，见 data_audit.json；新曲线使用全部观测点的首次失败边界。'),'',
        '有限窗口的判据可能局部非单调。Conversation/KV=4096 的 SPARE 原有较稀扫描在 5.64 QPS 通过，'
        '本轮新增 5.56 QPS 已因积压失败；按预定首次失败规则，最终容量平台采用 5.54 QPS，'
        '不跨过这一失败点去选择更高的孤立通过点。全部观测均保留在 rates 与 group records 中。','',
        '所有结果为 CPU 模拟器点估计；策略决策 CPU 时间未加入模拟时钟，无真实 GPU serving 测量。'
        '仅展示用户指定的固定种子，不以误差线暗示训练种子变异或统计置信区间。','',
        '## 图注','',
        '**Figure | Serving capacity under varying mean TTFT limits.** '
        'Panels a–d show Conversation/KV=2048, Code/KV=2048, Conversation/KV=4096, and Code/KV=4096, respectively. '
        'SPARE and a frozen independent Double DQN (seed 7047, episode 256) use the same 50 test windows per workload. '
        'The x-axis is the allowed mean TTFT on a logarithmic scale. Capacity is the highest measured input QPS before '
        'the first failure of the TTFT, TBT, backlog, or validity criteria. TBT and backlog constraints are held fixed '
        'as the TTFT limit changes. Markers are simulator point estimates, with pass/fail brackets at most 0.02 QPS wide; '
        'connecting lines are visual guides. Infeasible limits at the minimum measured rate of 0.25 QPS are omitted. '
        'The RL run was selected for its highest test capacity on Conversation at KV=2048 and then held fixed across panels. '
        'No retraining or per-panel seed selection is performed.','',
        '[PDF](rl7047_spare_capacity.pdf) · [PNG](rl7047_spare_capacity.png) · [SVG](rl7047_spare_capacity.svg) · '
        '[逐点数据](capacity_points.csv) · [数据核验](data_audit.json)','']
    (ANALYSIS/'REPORT_CN.md').write_text('\n'.join(lines))


if __name__=='__main__':
    groups,points,audit=load_and_audit()
    plot(points)
    report(groups,points,audit)
    print(f'Validated {len(groups)} groups and {len(points)} threshold points; exported to {ANALYSIS}')
