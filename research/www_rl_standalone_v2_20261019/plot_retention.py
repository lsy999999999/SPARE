"""Two-panel matched-capacity and workload-transfer view of measured data."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/www_rl7047_retention_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

sys.path.insert(0, str(Path.home()/'.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT/'artifacts/www_rl7047_frontier_20261020_r1/analysis'
OUT = ANALYSIS/'retention'
STYLES = {'spare': ('SPARE', '#2878b8', '-', 'o'),
          'rl_standalone': ('RL (seed 7047)', '#d9892a', '--', 's')}


def main():
    OUT.mkdir(exist_ok=True)
    with (ANALYSIS/'capacity_points.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    retention = []
    for arm in STYLES:
        pair = {}
        for source in ('conversation','code'):
            match = [r for r in rows if r['arm']==arm and r['source']==source
                     and r['kv_blocks']=='2048' and float(r['mean_ttft_limit_s'])==2.]
            assert len(match)==1
            pair[source]=float(match[0]['capacity_qps'])
        assert pair['conversation']>0
        for source in pair:
            retention.append(dict(arm=arm, source=source, kv_blocks=2048,
                mean_ttft_limit_s=2., capacity_qps=pair[source],
                reference_capacity_qps=pair['conversation'],
                retention_pct=100*pair[source]/pair['conversation']))
    with (OUT/'retention_points.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(retention[0]))
        writer.writeheader();writer.writerows(retention)

    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,'xtick.labelsize':7,
        'ytick.labelsize':7,'legend.fontsize':8,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.8,'pdf.fonttype':42,
        'svg.fonttype':'none','savefig.dpi':300})
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,85/25.4))
    fig.subplots_adjust(left=.085,right=.98,bottom=.28,top=.82,wspace=.36)
    left,right=axes
    for arm,(label,color,style,marker) in STYLES.items():
        chosen=[r for r in rows if r['arm']==arm and r['source']=='conversation' and r['kv_blocks']=='2048']
        x=[float(r['mean_ttft_limit_s']) for r in chosen]
        y=[float(r['capacity_qps']) if r['capacity_qps'] else float('nan') for r in chosen]
        assert all(v>0 for v in x) and x==sorted(x)
        left.plot(x,y,color=color,linestyle=style,marker=marker,markersize=3.3,
                  linewidth=1.5,markerfacecolor='white',label=label)
        values=[r['retention_pct'] for r in retention if r['arm']==arm]
        # Only two measured workload endpoints; the segment is a visual guide.
        right.plot([0,1],values,color=color,linestyle=style,marker=marker,
                   markersize=4,linewidth=1.5,markerfacecolor='white')
        right.text(1.06,values[1],f'{values[1]:.1f}%',color=color,ha='left',va='center',fontsize=8)
    left.set_xscale('log')
    left.set_xticks([.25,.5,1,2,4,8],labels=['0.25','0.5','1','2','4','8'])
    left.xaxis.set_minor_locator(NullLocator())
    left.set_xlim(.23,8.7);left.set_ylim(0,5)
    left.set_xlabel('Mean TTFT limit (s)')
    left.set_ylabel('Serving capacity (QPS)')
    left.set_title('Matched configuration',pad=10)
    right.set_xticks([0,1],labels=['Conversation\n(reference)','Code\n(unseen)'])
    right.set_xlim(-.12,1.43);right.set_ylim(60,106)
    right.set_yticks([60,70,80,90,100])
    right.set_xlabel('Workload')
    right.set_ylabel('Capacity retention (%)')
    right.set_title('Workload transfer | TTFT limit = 2 s',pad=10)
    for letter,ax in zip('ab',axes):
        ax.grid(axis='y',color='#e2e5e8',linewidth=.6)
        ax.set_axisbelow(True)
        ax.text(-.14,1.09,letter,transform=ax.transAxes,fontweight='bold',fontsize=9,
                ha='left',va='bottom')
    handles,labels=left.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.99),ncol=2,frameon=False)
    fig.text(.5,.035,'KV = 2048 in both panels; 50 windows per workload; frozen RL checkpoint',
             ha='center',va='bottom',fontsize=7,color='#535d67')
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=list(axes),panel_ids=['a','b'],
        json_out=OUT/'matched_capacity_retention.alignment.json')
    fig.savefig(OUT/'matched_capacity_retention.pdf')
    fig.savefig(OUT/'matched_capacity_retention.svg')
    fig.savefig(OUT/'matched_capacity_retention.png',dpi=300)
    plt.close(fig)

    note='''# 匹配性能与跨负载容量保持率

本图采用用户给出的两面板结构，全部曲线点来自已完成的独立 RL seed 7047 实验，没有新增模拟或训练。

- 左图：Conversation、KV=2048 的全部 17 档 TTFT 上限。保留高上限平台及低上限不可行点，未将 RL 画成全范围领先。
- 右图：固定 KV=2048、mean TTFT 上限=2 秒及原 TBT/积压条件，比较 Conversation 与 Code。
- 保持率定义为 `100 × C_method(Code, KV=2048, 2s) / C_method(Conversation, KV=2048, 2s)`。每个方法使用自己的 Conversation 容量作为分母，参考点均为 100%。SPARE 为 3.32/3.84=86.46%，RL 为 3.12/4.02=77.61%。差为 8.85 个百分点。
- 右图只含两个实测工作负载端点。连接线是视觉引导，不表示中间分布被测量，也不表示连续的分布偏移强度。Code 同时改变 prompt/output 分布，不能把变化单独归因于某一种长度。
- 这一比例同时反映工作负载自身难度与策略表现，不能称为“纯泛化损失”。两方法的配对比较说明当前场景下 SPARE 保留了更大比例的自身参考容量。
- KV=4096 结果保留在原四面板图中；本图固定 KV 是为了单独呈现跨工作负载变化。KV 增大后的容量比例可超过 100%，不宜混入单调“偏移强度”轴。
- RL 始终为 seed 7047 / episode 256。该 seed 按原匹配配置的测试容量选出，checkpoint 按验证集选出。图中为 CPU 模拟器点估计，没有置信区间；50 个窗口与扫描分辨率不等于统计置信度。

若要绘制多级偏移曲线，需要预先定义偏移轴（例如固定 KV 下的 Code 请求占比），补测中间混合比例，并固定平均到达率扫描协议和模型。请求流混合后重新测量容量，不能对两端容量做线性插值。实际曲线是否单调由结果决定。

## Figure contract

Question: Can matched-configuration capacity gains coexist with weaker retention across a workload change? Panel a establishes the matched reference; panel b describes capacity retention under the fixed workload-transfer protocol. Quantitative two-panel grid, 183 x 85 mm, Python/matplotlib, editable PDF/SVG and a 300 dpi PNG preview. All matched-setting thresholds retained; the transfer panel fixes KV and TTFT to isolate one experimental axis. No uncertainty interval estimated. Colors and line styles follow the supplied layout; data geometry comes only from archived values.

## Caption

**Matched-configuration capacity and workload-transfer retention.** (a) Serving capacity versus the allowed mean TTFT for Conversation at KV=2048. (b) At a fixed mean TTFT limit of 2 s and KV=2048, capacity is normalized to each method's Conversation capacity. The reference and unseen workload are the only measured endpoints in this panel; connecting segments are visual guides. Both panels use the same frozen RL checkpoint and 50 test windows per workload. TBT and backlog constraints remain fixed. Estimates are from the simulator; no uncertainty interval is shown. Seed 7047 was chosen for its highest matched-configuration test capacity among the evaluated runs. The normalized change includes both workload difficulty and policy response.
'''
    (OUT/'README_CN.md').write_text(note)
    print(json.dumps(retention,indent=2))


if __name__=='__main__':
    main()
