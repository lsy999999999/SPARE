"""Plot only the 1008 audited G3 runs; standalone panels for LaTeX."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_evaluation_revision import save, OUT, COLORS

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT/'artifacts/www_legacy_baselines_20261006_r1/analysis'
def read(name):
    with (DATA/name).open() as f:
        return list(csv.DictReader(f))

data = {r['arm']:r for r in read('aggregate.csv')}
pairs = {r['reference']:r for r in read('relative_to_arrival.csv')}
labels = {'arrival':'SPARE', 'native':'vLLM V1 default', 'past_future':'Past-Future rule',
          'orca_max':'Orca reservation', 's3_proxy':'S³ packing proxy',
          'ltr_proxy':'LTR score proxy', 'ltr_oracle':'LTR order oracle'}
colors = dict(COLORS, orca_max='#c29342', s3_proxy='#8173b1',
              ltr_proxy='#c77051', ltr_oracle='#497c45')
fig, axes = plt.subplots(2,3,figsize=(13,7),layout='constrained')
specs = [('ttft_ms_mean',.001,'Mean TTFT','Seconds ↓'),
         ('ttft_ms_p99',.001,'P99 TTFT','Seconds ↓'),
         ('e2e_ms_mean',.001,'Mean E2E','Seconds ↓'),
         ('output_throughput_tps',1,'Output throughput','Tokens/s ↑'),
         ('slo_ttft5000_maxitl200_attainment',100,'Joint SLO attainment','Percent ↑')]
for ax,(metric,scale,title,unit) in zip(axes.flat,specs):
    for i,arm in enumerate(labels):
        value=float(data[arm][metric])*scale
        ax.barh(i,value,color=colors[arm],hatch='///' if arm=='ltr_oracle' else None)
        ax.text(value,i,f' {value:.1f}',va='center',fontsize=8)
    ax.set_yticks(range(len(labels)),labels.values(),fontsize=8)
    ax.invert_yaxis();ax.set_title(title);ax.set_xlabel(unit)
    ax.set_xlim(0,ax.get_xlim()[1]*1.13);ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
ax=axes.flat[-1]
for i,arm in enumerate(list(labels)[1:]):
    r=pairs[arm];value=float(r['arrival_mean_ttft_improvement_pct'])
    lo=float(r['bootstrap_window_95ci_low_pct']);hi=float(r['bootstrap_window_95ci_high_pct'])
    ax.errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt='o',capsize=3,color=colors[arm])
ax.set_yticks(range(6),list(labels.values())[1:],fontsize=8);ax.invert_yaxis()
ax.axvline(0,color='#687789',lw=.7);ax.grid(axis='x',alpha=.16)
ax.set_xlabel('Paired mean TTFT reduction (%) ↑');ax.set_title('SPARE vs. each reference · 95% CI')
fig.suptitle('G3 · 36 scenarios × four repeats · shared V1 backend\nSLO: TTFT ≤ 5 s and max ITL ≤ 200 ms; CIs clustered by six source windows',fontsize=11)
OUT.mkdir(parents=True,exist_ok=True)
save(fig,'E12_g3_baselines')
print('Wrote E12_g3_baselines and six standalone PDF/PNG panels')
