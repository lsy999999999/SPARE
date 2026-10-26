"""Source data and exploratory figures for the two non-comparable KV cohorts."""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/www_kv_extreme_stress_20261016_r1'
FIG = ROOT / 'writing/www/figures/newidea'
os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'mpl_cache'))
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path.home() / '.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

from research.www_kv_extreme_stress_20261016.run import CAPACITIES, METHODS, RATES, base

STYLE = (
    ('arrival', 'SPARE', '#195c9b', '-', 'o'),
    ('native', 'vLLM V1', '#494949', '--', 's'),
    ('sarathi', 'Sarathi rule', '#439bc2', '-.', '^'),
    ('sola_rule', 'SOLA rule', '#d25a35', ':', 'v'),
    ('uniboost_static', 'UniBoost rule', '#9562a0', '--', 'D'),
    ('past_future', 'Past-Future rule', '#238a83', '-.', 'P'),
    ('s3_proxy', 'S3 proxy', '#b64342', ':', 'X'),
    ('apt_serve_kv_rule', 'Apt-Serve rule', '#a67b1e', '--', '*'),
)


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load(cohort):
    status = base.read(OUT / cohort / 'status.json')
    assert status['stage'] == 'complete' and not status['errors'], status
    design = base.read(OUT / cohort / 'design.json')
    assert tuple(design['methods']) == METHODS
    assert tuple(design['capacities']) == CAPACITIES[cohort]
    rows = []
    for source in ('code', 'conversation'):
        for blocks in CAPACITIES[cohort]:
            for arm in METHODS:
                result = base.read(OUT / cohort / 'groups' / f'{source}_c{blocks}_{arm}.json')
                assert result['cohort'] == cohort
                path = OUT / cohort / 'rates' / f'{source}_c{blocks}_{arm}_r0.5.json'
                record = base.read(path)
                assert len(record['rows']) == 10
                assert result['reference_feasible'] == (result['capacity_qps'] >= .5)
                if result['first_failed_qps'] is not None:
                    failure = base.read(OUT / cohort / 'rates' /
                        f"{source}_c{blocks}_{arm}_r{result['first_failed_qps']:g}.json")['summary']
                    reasons = [name for name, key, limit in (
                        ('p95_ttft', 'p95_ttft_ms', 12000.),
                        ('p99_tbt', 'p99_tbt_ms', 200.),
                        ('backlog', 'backlog_drift', failure['drift_limit']))
                        if failure[key] is not None and failure[key] > limit]
                    reasons += failure.get('failure_reasons', [])
                else:
                    reasons = []
                rows.append(dict(cohort=cohort, source=source, kv_blocks=blocks,
                    arm=arm, method=next(label for a,label,*_ in STYLE if a==arm),
                    capacity_qps=result['capacity_qps'],
                    next_failed_qps=result['first_failed_qps'],
                    capacity_lower_bound=result['capacity_is_lower_bound'],
                    mean_ttft_ms_at_0_5_qps=record['summary']['mean_ttft_ms'],
                    reference_feasible=result['reference_feasible'],
                    first_failure_reasons=';'.join(reasons),
                    measured_windows=10, source_days=2,
                    rates_measured=result['rates_done']))
    assert len(rows)==2*len(CAPACITIES[cohort])*len(METHODS)
    analysis=OUT/'analysis';analysis.mkdir(exist_ok=True)
    write_csv(analysis/f'{cohort}_by_source.csv', rows)
    return rows


def plot(cohort, rows):
    FIG.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({'font.family':'sans-serif',
        'font.sans-serif':['DejaVu Sans','Arial','Helvetica'],
        'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
    fig, axes = plt.subplots(2,2,figsize=(10.5,6.1))
    fig.subplots_adjust(left=.08,right=.98,bottom=.20,top=.91,wspace=.25,hspace=.42)
    tick_candidates=(512,768,1024,2048,4096,8192) if cohort=='full' else (
        128,160,192,256,384,512,768,1024,2048,4096)
    ticks=[c for c in tick_candidates if c in CAPACITIES[cohort]]
    for i,source in enumerate(('code','conversation')):
        for j,(field,ylabel) in enumerate((('capacity_qps','SLO serving capacity (QPS)'),
                                            ('mean_ttft_ms_at_0_5_qps','Mean TTFT at 0.5 QPS (ms)'))):
            ax=axes[i,j]
            for arm,label,color,ls,mark in STYLE:
                points=sorted((r for r in rows if r['source']==source and r['arm']==arm),
                              key=lambda r:r['kv_blocks'])
                x=[r['kv_blocks'] for r in points]
                y=[float(r[field]) if r['reference_feasible'] or j==0 else float('nan') for r in points]
                ax.plot(x,y,color=color,linestyle=ls,marker=mark,markersize=3,
                        linewidth=1.8 if arm=='arrival' else 1.,
                        alpha=1 if arm=='arrival' else .83,zorder=5 if arm=='arrival' else 2,label=label)
            ax.set_xscale('log',base=2)
            ax.set_xticks(ticks,[str(t) for t in ticks],minor=False)
            ax.tick_params(axis='x',labelsize=6.5)
            ax.set_xlim(CAPACITIES[cohort][0]*.93,CAPACITIES[cohort][-1]*1.08)
            ax.set_ylim(bottom=0)
            ax.set_xlabel('KV capacity (blocks)')
            ax.set_ylabel(ylabel)
            ax.set_title(source.capitalize())
            ax.grid(axis='y',color='#e1e4e6',linewidth=.5)
            ax.annotate(chr(97+i*2+j),xy=(0,1),xycoords='axes fraction',
                        xytext=(-27,12),textcoords='offset points',fontsize=9,fontweight='bold')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.52,.035))
    fig.canvas.draw()
    stem=FIG/f'kv_extreme_{cohort}_capacity'
    require_matplotlib_panel_alignment(fig,json_out=str(stem)+'.alignment.json',strict=True)
    fig.savefig(str(stem)+'.pdf')
    fig.savefig(str(stem)+'.svg')
    fig.savefig(str(stem)+'.png',dpi=300)
    plt.close(fig)


if __name__=='__main__':
    for cohort in ('full','short'):
        plot(cohort,load(cohort))
