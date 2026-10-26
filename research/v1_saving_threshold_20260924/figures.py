"""Standalone capacity and factor figures, generated from saved measurements."""
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from research.v1_saving_threshold_20260924.common import *


def main():
    data=read(OUT/'analysis/summary.json')
    with (OUT/'analysis/runs.csv').open() as f:rows=list(csv.DictReader(f))
    target=OUT/'figures';target.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,8),sharex=True,sharey=True)
    image=None
    for ax,(source,rate) in zip(axes.flat,[('code',1),('code',2),('conversation',1),('conversation',2)]):
        cell=[r for r in rows if r['phase']=='map' and r['case']==f'{source}_rate{rate}' and r['status']=='complete']
        values=np.full((len(THRESHOLDS),len(CAPS)),np.nan)
        for ci,cap in enumerate(CAPS):
            lookup={json.loads(r['threshold']):float(r['ttft_ms']) for r in cell if int(r['capacity'])==cap}
            if 'off' not in lookup:continue
            for ni,n in enumerate(THRESHOLDS):
                if n in lookup:values[ni,ci]=100*(1-lookup[n]/lookup['off'])
        image=ax.imshow(values,aspect='auto',norm=TwoSlopeNorm(vmin=-50,vcenter=0,vmax=80),cmap='RdYlGn')
        for ni in range(len(THRESHOLDS)):
            for ci in range(len(CAPS)):
                if np.isfinite(values[ni,ci]):ax.text(ci,ni,str(round(float(values[ni,ci]))),ha='center',va='center',fontsize=8)
        ax.set_title(f'{source}, arrival rate x{rate}')
        ax.set_xticks(range(len(CAPS)),CAPS,rotation=40)
        ax.set_yticks(range(len(THRESHOLDS)),THRESHOLDS)
        ax.set_xlabel('Total physical KV blocks');ax.set_ylabel('Minimum blocks saved')
    fig.suptitle('Capacity / threshold response map (one run per cell)')
    fig.subplots_adjust(left=.07,right=.85,bottom=.12,top=.91,hspace=.35,wspace=.12)
    cax=fig.add_axes([.89,.17,.018,.66])
    fig.colorbar(image,cax=cax,label='TTFT improvement over no intervention (%)')
    fig.savefig(target/'capacity_threshold_map.png',dpi=180);fig.savefig(target/'capacity_threshold_map.pdf');plt.close(fig)

    factors=data['factor_interactions']
    if factors:
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,factor in zip(axes.flat,['density','burst','prompt','output']):
            for source,color in [('code','#3366aa'),('conversation','#ce6c22')]:
                ds=sorted([d for d in factors if d['factor']==factor and d['source']==source],key=lambda d:d['capacity'])
                for n,style in [(1,'-o'),(16,'--s')]:
                    ax.plot([d['capacity'] for d in ds],[d[f'interaction_n{n}_pp'] for d in ds],style,color=color,label=f'{source}, n={n}')
            ax.axhline(0,color='gray',lw=.7);ax.set_title(factor)
            ax.set_xlabel('KV blocks');ax.set_ylabel('High vs low change in intervention gain (pp)')
            ax.legend(fontsize=8)
        fig.suptitle('Controlled factor interactions');fig.tight_layout()
        fig.savefig(target/'factor_interactions.png',dpi=180);fig.savefig(target/'factor_interactions.pdf');plt.close(fig)

        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for ax,factor in zip(axes.flat,['density','burst','prompt','output']):
            ds=sorted([d for d in factors if d['factor']==factor],key=lambda d:(d['source'],d['capacity']))
            x=np.arange(len(ds))
            ax.bar(x-.18,[d['low_n1_advantage_over_n16_percent'] for d in ds],width=.36,label='Low factor level')
            ax.bar(x+.18,[d['high_n1_advantage_over_n16_percent'] for d in ds],width=.36,label='High factor level')
            ax.set_xticks(x,[f"{d['source'][:4]}\n{d['capacity']}" for d in ds])
            ax.axhline(0,color='gray',lw=.8);ax.set_title(factor)
            ax.set_ylabel('n=1 advantage over n=16 (%)');ax.legend(fontsize=8)
        fig.suptitle('Does the factor change threshold preference?');fig.tight_layout()
        fig.savefig(target/'factor_threshold_preference.png',dpi=180);fig.savefig(target/'factor_threshold_preference.pdf');plt.close(fig)

    evidence=OUT/'analysis/evidence.json'
    if evidence.exists():
        cells=read(evidence)['capacity_cells']
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
        for ax,source in zip(axes,['code','conversation']):
            ds=sorted([d for d in cells if d['case']==source+'_rate1'],key=lambda d:d['capacity'])
            for key,style,label in [('n1_gain_percent','-o','n=1'),('n16_gain_percent','--s','n=16'),('best_gain_percent',':^','Best tested threshold')]:
                ax.plot([d['capacity'] for d in ds],[d[key] for d in ds],style,label=label)
            ax.axhline(0,color='gray',lw=.8);ax.axhline(5,color='gray',lw=.6,ls=':')
            ax.set_title(source+', original arrival rate');ax.set_xlabel('KV blocks');ax.legend(fontsize=8)
        axes[0].set_ylabel('TTFT improvement over no intervention (%)')
        fig.suptitle('Capacity response with exploratory refinement points');fig.tight_layout()
        fig.savefig(target/'capacity_refinement.png',dpi=180);fig.savefig(target/'capacity_refinement.pdf');plt.close(fig)

    formula=[r for r in rows if r['phase']=='formula' and r['status']=='complete']
    if formula:
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for ax,source in zip(axes,['code','conversation']):
            for arm,style in [('formula','-o'),('ablated','--s'),('off',':^')]:
                ds=sorted([r for r in formula if r['case']==source+'_new' and r['arm']==arm],key=lambda r:int(r['capacity']))
                ax.plot([int(r['capacity']) for r in ds],[float(r['ttft_ms']) for r in ds],style,label=arm)
            ax.set_title(source+' new window');ax.set_xlabel('KV blocks');ax.set_ylabel('Mean TTFT (ms)');ax.legend()
        fig.tight_layout();fig.savefig(target/'formula_new_windows.png',dpi=180);fig.savefig(target/'formula_new_windows.pdf');plt.close(fig)
    print('Figures:',target)


if __name__=='__main__':main()
