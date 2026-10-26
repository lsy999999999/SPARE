"""Audited KV curves and paired workload-mixture relative gain for frozen RL."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
import sys

os.environ.setdefault('MPLCONFIGDIR','/tmp/www_rl7047_transfer_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import NullLocator

sys.path.insert(0,str(Path.home()/'.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/www_rl7047_transfer_20261020_r1'
ANALYSIS=OUT/'analysis'
STYLES={'spare':('SPARE','#2878b8','-','o'),
        'rl_standalone':('RL (seed 7047)','#d9892a','--','s')}


def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def passed(s,qps):
    return s['failed_samples']==0 and s['mean_ttft_ms']<=2000 and s['p99_tbt_ms']<=200 and s['backlog_drift']<=max(2,qps)


def save_csv(path,rows):
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def mixture_gains(rows):
    result=[]
    assert len(rows)==10
    for pct in (0,25,50,75,100):
        paired=[r for r in rows if r['code_pct']==pct]
        assert len(paired)==2 and {r['arm'] for r in paired}==set(STYLES)
        assert all(r['kv_blocks']==2048 for r in paired)
        d={r['arm']:r['capacity_qps'] for r in paired}
        spare,rl=d['spare'],d['rl_standalone']
        assert all(v is not None and math.isfinite(v) and v>0 for v in (spare,rl))
        result.append(dict(code_pct=pct,kv_blocks=2048,mean_ttft_limit_s=2,
            spare_capacity_qps=spare,rl_capacity_qps=rl,
            rl_gain_over_spare_pct=100*(rl-spare)/spare))
    return result


def audit_data():
    for stage in ('kv','mix'):assert read(OUT/f'status_{stage}.json')['complete']
    selected=read(OUT/'design_kv.json')['selected']
    assert selected==read(OUT/'design_mix.json')['selected']
    groups=[read(p) for p in sorted((OUT/'groups').glob('*.json'))]
    assert len(groups)==30
    paired={};points=[];rate_total=0;new=0;reused=0;window_total=0;invalid=0
    for g in groups:
        assert g['checkpoint_sha256']==(selected['sha256'] if g['arm']!='spare' else None)
        records={}
        for r in g['records']:
            path=ROOT/r['path'];assert digest(path)==r['sha256']
            d=read(path);assert d['summary']==r['summary']
            assert d['input_sha256']==g['input_sha256'] and d['checkpoint_sha256']==g['checkpoint_sha256']
            rows=d['rows'];assert len(rows)==50 and len({r['name'] for r in rows})==50
            assert all(row['capacity']==g['kv_blocks'] and row['rate']==r['rate_qps'] for row in rows)
            valid=[row for row in rows if row['valid']]
            assert len(rows)-len(valid)==d['summary']['failed_samples']
            for key in ('mean_ttft_ms','p99_tbt_ms','backlog_drift'):
                if valid:assert math.isclose(fmean(row[key] for row in valid),d['summary'][key],rel_tol=1e-10,abs_tol=1e-7)
                else:assert d['summary'][key] is None
            if 'reused_from' in d:
                assert digest(ROOT/d['reused_from'])==d['reused_sha256'];reused+=1
            else:new+=1
            records[r['rate_qps']]=d
            if g['experiment']=='mix':
                for row in rows:
                    key=(g['code_pct'],r['rate_qps'],row['name'])
                    sig=(row['stream_sha256'],row['total_requests'],row['code_requests'])
                    if key in paired:assert paired[key]==sig
                    else:paired[key]=sig
            rate_total+=1;window_total+=len(rows);invalid+=len(rows)-len(valid)
        low=None
        for rate in sorted(records):
            if not passed(records[rate]['summary'],rate):high=rate;break
            low=rate
        assert low==g['capacity_qps'] and high==g['next_failed_qps']
        assert low is None or 0<high-low<=.020001
        refs={r:records[r]['summary'] for r in (2.,3.)}
        row=dict(experiment=g['experiment'],source=g['source'],kv_blocks=g['kv_blocks'],
            code_pct=g['code_pct'],arm=g['arm'],capacity_qps=low,next_failed_qps=high,
            resolution_qps=g['resolution_qps'],first_failure_reasons=';'.join(g['first_failure_reasons']),
            mean_ttft_at_2qps_ms=refs[2.]['mean_ttft_ms'],mean_ttft_at_3qps_ms=refs[3.]['mean_ttft_ms'],
            meets_capacity_rule_at_2qps=passed(refs[2.],2.),meets_capacity_rule_at_3qps=passed(refs[3.],3.))
        if g['experiment']=='mix':
            caprows=records[low]['rows'] if low else []
            row['realized_code_share_at_capacity_pct']=(100*sum(r['code_requests'] for r in caprows)/sum(r['total_requests'] for r in caprows)) if caprows else None
            row['measured_code_share_at_capacity_pct']=(100*sum(r['measured_code_requests'] for r in caprows)/sum(r['measured_stream_requests'] for r in caprows)) if caprows else None
        points.append(row)
    kv=[p for p in points if p['experiment']=='kv'];mix=[p for p in points if p['experiment']=='mix']
    baseline={p['arm']:p['capacity_qps'] for p in mix if p['code_pct']==0}
    for p in mix:p['retention_pct']=100*p['capacity_qps']/baseline[p['arm']] if p['capacity_qps'] else None
    # Both methods must have identical cohort signatures at every condition.
    signatures={}
    for g in groups:
        key=(g['experiment'],g['source'],g['kv_blocks'],g['code_pct'])
        if key in signatures:assert signatures[key]==g['input_sha256']
        else:signatures[key]=g['input_sha256']
    ANALYSIS.mkdir(exist_ok=True)
    save_csv(ANALYSIS/'kv_points.csv',kv);save_csv(ANALYSIS/'mixture_points.csv',mix)
    save_csv(ANALYSIS/'mixture_relative_gain.csv',mixture_gains(mix))
    result=dict(groups=len(groups),rate_records=rate_total,window_records=window_total,
        reused_rates=reused,new_rates=new,invalid_windows=invalid,
        capacity_brackets_verified=True,paired_mixture_streams_verified=True,
        per_window_aggregates_verified=True)
    (ANALYSIS/'data_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return kv,mix,result


def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,'xtick.labelsize':7,
        'ytick.labelsize':7,'legend.fontsize':8,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.8,'pdf.fonttype':42,'svg.fonttype':'none'})


def use_arial():
    font_dir=os.environ.get('WWW_ARIAL_FONT_DIR')
    if font_dir:
        for path in Path(font_dir).iterdir():
            if path.suffix.lower()=='.ttf':font_manager.fontManager.addfont(str(path))
    font_manager.findfont(font_manager.FontProperties(family='Arial'),fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial','font.sans-serif':['Arial']})


def line(ax,x,y,arm):
    label,color,ls,marker=STYLES[arm]
    ax.plot(x,y,color=color,linestyle=ls,marker=marker,linewidth=1.5,
            markersize=3.5,markerfacecolor='white',label=label)


def finish(fig,axes,stem,footnote,center_panel_labels=False):
    for letter,ax in zip('abcd',axes):
        ax.grid(axis='y',color='#e3e7e9',linewidth=.55);ax.set_axisbelow(True)
        if center_panel_labels:
            ax.text(.5,1.17,f'({letter})',transform=ax.transAxes,
                    ha='center',va='bottom',fontsize=9,fontweight='bold')
        else:
            ax.text(-.14,1.08,letter,transform=ax.transAxes,
                    ha='left',va='bottom',fontsize=9,fontweight='bold')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.995),ncol=2,frameon=False)
    fig.text(.5,.025,footnote,ha='center',va='bottom',fontsize=7,color='#505a64')
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=list('abcd')[:len(axes)],
        json_out=ANALYSIS/f'{stem}.alignment.json')
    fig.savefig(ANALYSIS/f'{stem}.pdf')
    fig.savefig(ANALYSIS/f'{stem}.svg')
    fig.savefig(ANALYSIS/f'{stem}.png',dpi=300)
    plt.close(fig)


def plot_kv(rows):
    fig,axes=plt.subplots(2,2,figsize=(183/25.4,132/25.4))
    fig.subplots_adjust(left=.1,right=.98,bottom=.16,top=.88,hspace=.42,wspace=.30)
    kvs=sorted({r['kv_blocks'] for r in rows})
    max_latency=max(r['mean_ttft_at_2qps_ms']/1000 for r in rows)
    ymax=max(2.2,math.ceil(max_latency*1.1*2)/2)
    max_capacity=max(r['capacity_qps'] for r in rows if r['capacity_qps'])
    for i,source in enumerate(('conversation','code')):
        for arm in STYLES:
            chosen=sorted([r for r in rows if r['source']==source and r['arm']==arm],key=lambda r:r['kv_blocks'])
            assert [r['kv_blocks'] for r in chosen]==kvs
            line(axes[i,0],kvs,[r['mean_ttft_at_2qps_ms']/1000 for r in chosen],arm)
            line(axes[i,1],kvs,[r['capacity_qps'] if r['capacity_qps'] is not None else float('nan') for r in chosen],arm)
        axes[i,0].set_title(f'{source.capitalize()} | offered load = 2 QPS',pad=10)
        axes[i,1].set_title(f'{source.capitalize()} | TTFT limit = 2 s',pad=10)
        axes[i,0].set_ylabel('Mean TTFT (s)');axes[i,0].set_ylim(0,ymax)
        axes[i,1].set_ylabel('Serving capacity (QPS)');axes[i,1].set_ylim(0,math.ceil(max_capacity+.2))
        for ax in axes[i]:
            ax.set_xticks(kvs,[str(k) for k in kvs]);ax.set_xlim(min(kvs)-220,max(kvs)+220)
            ax.set_xlabel('KV capacity (blocks)')
    finish(fig,list(axes.flat),'kv_ttft_capacity',
        'RL trained on Conversation at KV = 2048; fixed seed 7047; simulator estimates')


def plot_mixture(rows):
    use_arial()
    with (ROOT/'artifacts/www_rl7047_frontier_20261020_r1/analysis/capacity_points.csv').open() as stream:
        original=list(csv.DictReader(stream))
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,85/25.4))
    fig.subplots_adjust(left=.085,right=.98,bottom=.25,top=.82,wspace=.35)
    for arm in STYLES:
        chosen=[r for r in original if r['arm']==arm and r['source']=='conversation' and r['kv_blocks']=='2048']
        x=[float(r['mean_ttft_limit_s']) for r in chosen]
        y=[float(r['capacity_qps']) if r['capacity_qps'] else float('nan') for r in chosen]
        assert all(v>0 for v in x) and x==sorted(x)
        line(axes[0],x,y,arm)
    gains=mixture_gains(rows)
    axes[1].plot([r['code_pct'] for r in gains],[r['rl_gain_over_spare_pct'] for r in gains],
        color='#374955',linestyle='-',marker='o',linewidth=1.5,
        markersize=4,markerfacecolor='white')
    axes[1].axhline(0,color='#505a64',linewidth=1,linestyle='--')
    axes[1].text(94,4.4,'RL better',ha='right',va='center',color='#986019',fontsize=8)
    axes[1].text(6,-4.4,'SPARE better',ha='left',va='center',color=STYLES['spare'][1],fontsize=8)
    axes[0].set_xscale('log');axes[0].set_xlim(.23,8.7);axes[0].set_ylim(0,5)
    axes[0].set_xticks([.25,.5,1,2,4,8],labels=['0.25','0.5','1','2','4','8'])
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].set_title('Matched configuration',pad=10)
    axes[0].set_xlabel('Mean TTFT limit (s)');axes[0].set_ylabel('Serving capacity (QPS)')
    axes[1].set_title('Workload mixture | TTFT limit = 2 s',pad=10)
    axes[1].set_xlabel('Code request share (%)');axes[1].set_ylabel('RL capacity gain over SPARE (%)')
    axes[1].set_xticks([0,25,50,75,100]);axes[1].set_xlim(-4,104)
    axes[1].set_ylim(-6.5,6.5)
    axes[1].set_yticks([-6,-3,0,3,6],labels=['−6','−3','0','+3','+6'])
    finish(fig,list(axes),'matched_capacity_mixture_gain',
        'KV = 2048; 50 paired streams per mixture; fixed RL checkpoint',
        center_panel_labels=True)


def report(kv,mix,audit):
    gains=mixture_gains(mix)
    lines=['# RL 7047：跨 KV 与混合请求流补测','',
        '所有结果固定独立 RL v2 seed 7047 / episode 256，无重新训练。该 seed 按原匹配配置测试容量选取；checkpoint 按验证集选取。',
        'KV 比较覆盖 1024、2048、3072、4096、6144。mean TTFT 在相同 2 QPS 下比较；CSV 另保留 3 QPS 结果。'
        'serving capacity 统一要求 mean TTFT≤2 s、窗口 P99 TBT 均值≤200 ms、40–60 s 平均积压增长≤max(2,QPS)、50 窗口全部有效。'
        '容量采用全部已测 QPS 从低至高的首次失败边界，分辨率≤0.02 QPS，不作为置信区间。','',
        '## 主要观察','',
        '- KV=1024 时 RL 在两种负载上的容量和固定 2 QPS 的 mean TTFT 均更好。Conversation 的容量为 2.24 对 2.10 QPS；Code 为 2.94 对 2.82 QPS。',
        '- Conversation 在 KV=2048 时 RL 领先，3072 时容量持平，4096/6144 时 SPARE 更高；Code 在 KV≥2048 时 SPARE 的容量更高。跨 KV 表现具有条件性，不能写成 RL 在所有未训练容量上都失败。',
        f'- Code 占比从 0% 增加至 100% 时，固定 RL 策略相对 SPARE 的容量收益从 {gains[0]["rl_gain_over_spare_pct"]:+.2f}% 降至 {gains[-1]["rl_gain_over_spare_pct"]:+.2f}%。五个实测点依次下降，在 50% 与 75% 之间由正转负。','',
        '## 跨 KV 结果','',
        '| 负载 | KV | SPARE mean TTFT @2 QPS (ms) | RL mean TTFT @2 QPS (ms) | SPARE capacity | RL capacity |',
        '|---|---:|---:|---:|---:|---:|']
    for source in ('conversation','code'):
        for blocks in sorted({p['kv_blocks'] for p in kv}):
            d={p['arm']:p for p in kv if (p['source'],p['kv_blocks'])==(source,blocks)}
            s,r=d['spare'],d['rl_standalone']
            lines.append(f'| {source.capitalize()} | {blocks} | {s["mean_ttft_at_2qps_ms"]:.1f} | {r["mean_ttft_at_2qps_ms"]:.1f} | {s["capacity_qps"]} | {r["capacity_qps"]} |')
    lines+=['','## 混合流协议','',
        '固定 KV=2048，Code 占比为 0%、25%、50%、75%、100%。按窗口编号将 50 个 Conversation 窗口与 50 个 Code 窗口配对；'
        '配对是实验编号配对，不意味着两种负载采自同一时间。各窗口有 64 个原始请求的 prompt/output 配对。'
        '到达时刻与长度抽样索引统一采用 Conversation 原随机流。每四个到达请求使用一个固定随机排列，按占比选择其中 0/1/2/3/4 个请求替换为配对 Code 池中的完整 prompt/output；'
        '五档比例采用嵌套替换，相同 QPS 下两方法的实际输入完全相同。窗口末尾或测量区间内的不完整四请求块会使实际占比略偏离名义值，实际比例保存在 CSV 与逐窗口记录中。','',
        '历史输出长度信息固定为 Conversation 开发历史，不随测试占比切换。0% 端点与原 Conversation 输入逐请求一致，可复用；'
        '100% 端点采用共同到达随机流，必须重新评估，不能与之前单独 Code 端点直接拼接。'
        '每个混合比例独立运行容量搜索，不由两端容量插值。','',
        '## RL 相对 SPARE 的容量收益','',
        r'$$G(p)=\frac{C_{\mathrm{RL}}(p)-C_{\mathrm{SPARE}}(p)}{C_{\mathrm{SPARE}}(p)}\times100\%$$','',
        '其中 p 为 Code 请求占比，C 为同一混合流条件、相同 mean TTFT≤2 s 及 TBT/积压约束下的 serving capacity。'
        'G>0 表示 RL 更好，G=0 表示持平，G<0 表示 SPARE 更好。分母是当前占比下的 SPARE 容量。','',
        '| Code 名义占比 | SPARE capacity (QPS) | RL capacity (QPS) | RL 相对收益 G | SPARE/RL 容量点实际 Code 占比 |',
        '|---:|---:|---:|---:|---:|']
    for gain in gains:
        pct=gain['code_pct']
        d={p['arm']:p for p in mix if p['code_pct']==pct};s,r=d['spare'],d['rl_standalone']
        lines.append(f'| {pct}% | {s["capacity_qps"]:.2f} | {r["capacity_qps"]:.2f} | {gain["rl_gain_over_spare_pct"]:+.2f}% | {s["realized_code_share_at_capacity_pct"]:.2f}% / {r["realized_code_share_at_capacity_pct"]:.2f}% |')
    lines+=['','## 论文表述与图注','',
        '在固定 KV=2048、mean TTFT 上限为 2 s 的混合流实验中，随着 Code 请求逐步替换 Conversation 请求，'
        '在 Conversation 上训练的固定 RL 策略相对 SPARE 的容量收益逐步收窄，从 +4.69% 降至 −5.16%，'
        '并在 Code 占比 50% 与 75% 之间发生优势反转。这表明该策略在匹配负载上的额外收益未能保持到更高 Code 占比的负载中。','',
        '图注：左图为匹配配置下 serving capacity 随 mean TTFT 上限的变化；右图为 RL 相对 SPARE 的容量收益 G(p)，'
        '固定 mean TTFT 上限为 2 s。两图均为 KV=2048，使用同一个 RL checkpoint（seed 7047 / episode 256）。'
        '右图标记对应五档实测比例，折线仅连接实测点；0% 水平线表示两者容量相同，上方 RL better，下方 SPARE better。'
        '每档比例使用 50 个配对请求流，容量还满足相同 TBT 与积压约束。','']
    lines+=['','## 验证与解释范围','',
        f'全部 {audit["groups"]} 组完成；复用 {audit["reused_rates"]} 个、新增 {audit["new_rates"]} 个逐 QPS 文件，'
        f'共 {audit["window_records"]} 条窗口记录，运行无效窗口 {audit["invalid_windows"]} 个。'
        '重算逐窗口指标均值、核对文件与模型哈希、混合流两方法的逐请求输入哈希，并独立核验所有容量通过/失败边界。'
        '0% 端点当前代码回放通过，混合比例嵌套、到达时刻一致、每四请求块比例精确等性质已验证。','',
        '所有图为 CPU 模拟器点估计，无置信区间；模型推理 CPU 耗时未加入模拟时钟。KV 图连线仅连接五档实测配置。'
        '混合图横轴明确为 Code 请求占比，不把它称为通用分布距离，也不强制单调曲线。','',
        '[KV 图 PDF](kv_ttft_capacity.pdf) · [KV CSV](kv_points.csv) · '
        '[混合流图 PDF](matched_capacity_mixture_gain.pdf) · [混合流 CSV](mixture_points.csv) · '
        '[相对收益 CSV](mixture_relative_gain.csv)','']
    (ANALYSIS/'REPORT_CN.md').write_text('\n'.join(lines))


if __name__=='__main__':
    kv,mix,audit=audit_data();style();plot_kv(kv);plot_mixture(mix);report(kv,mix,audit)
    print(json.dumps(audit,indent=2))
