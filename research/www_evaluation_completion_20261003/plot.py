import statistics
from collections import defaultdict
from .common import OUT, read

def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,profile in zip(axes,('resource','resource_staircase','gpu_fitted')):
        for arm in ('spare','basic','srp','fixed512'):
            values=defaultdict(list)
            for r in results:
                t=r['task']
                if t['family']=='sustained' and t['profile']==profile and t['arm']==arm and t['capacity']==2048:
                    values[t['arrival_rps']].append(r['metrics']['ttft_ms_mean']/1000)
            if values:
                rates=sorted(values);ax.plot(rates,[statistics.fmean(values[x]) for x in rates],marker='o',label=arm)
        ax.set_title(profile);ax.set_xlabel('Arrival rate (requests/s)');ax.grid(alpha=.25)
    axes[0].set_ylabel('Mean TTFT (s)');axes[-1].legend();fig.suptitle('512-request replay, 2048 configured KV blocks; six windows')
    fig.tight_layout();fig.savefig(OUT/'sustained_load.png',dpi=170);fig.savefig(OUT/'sustained_load.pdf');plt.close(fig)

if __name__=="__main__":
    results=[]
    for t in read(OUT/"design.json")["tasks"]:
        p=OUT/"runs"/t["name"]/"result.json"
        if p.exists() and read(p).get("task"):results.append(read(p))
    plot(results)
