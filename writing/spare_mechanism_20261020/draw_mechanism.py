"""Reproduce the SPARE mechanism diagram from the author's method description.

Run with Python plus matplotlib. Optional skill audit location can be supplied
through FIGURE_AUDIT_DIR; by default the locally installed audit is used.
All geometry is in millimetres. Token glyphs are qualitative, not observations.
"""
from pathlib import Path
import os
import sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp/spare-mechanism-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.path import Path as MPath

OUT = Path(__file__).resolve().parent
AUDIT = Path(os.environ.get('FIGURE_AUDIT_DIR', str(Path.home() / '.codex/skills/nature-figure/scripts')))
sys.path.insert(0, str(AUDIT))
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 8,
    'mathtext.fontset': 'dejavusans',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'axes.spines.top': False,
    'axes.spines.right': False,
})
INK = '#203447'
MUTED = '#526575'
BLUE = '#276D9C'
PALE = '#EDF4F9'
TEAL = '#21786D'
TEAL_PALE = '#EDF6F3'
AMBER = '#A56B22'
AMBER_PALE = '#FBF3E7'
EDGE = '#C8D3DD'

fig = plt.figure(figsize=(7.0866141732, 6.1023622047), facecolor='white')  # 180 × 155 mm
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 180), ylim=(0, 155))
ax.set_axis_off()

def text(x, y, value, size=8, color=INK, weight='normal', ha='left', va='center', **kw):
    return ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                   ha=ha, va=va, linespacing=1.4, **kw)

def box(x, y, w, h, fill='white', edge=EDGE, radius=1.8, lw=.75):
    patch = FancyBboxPatch((x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={radius}',
        facecolor=fill, edgecolor=edge, linewidth=lw)
    ax.add_patch(patch)
    return patch

def arrow(points, color=MUTED, dashed=False, lw=1.0):
    path = MPath(points, [MPath.MOVETO] + [MPath.LINETO] * (len(points)-1))
    patch = FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=8,
        linewidth=lw, color=color, linestyle=(0,(3,2)) if dashed else '-')
    ax.add_patch(patch)

def heading(x, y, n, title, color=BLUE):
    ax.add_patch(Circle((x+2.4, y), 2.4, facecolor=color, edgecolor='none'))
    text(x+2.4, y, str(n), size=8, color='white', weight='bold', ha='center')
    text(x+6.5, y, title, size=9, weight='bold')

def tokens(x, y, styles):
    for j, style in enumerate(styles):
        fill, edge = (BLUE, BLUE) if style=='keep' else (AMBER_PALE, AMBER)
        ax.add_patch(Rectangle((x+j*4.8,y),3.6,3.0,facecolor=fill,
                              edgecolor=edge,linewidth=.6))
        if style != 'keep':
            ax.plot([x+j*4.8+.3,x+j*4.8+3.3],[y+.3,y+2.7],color=AMBER,lw=.6)

# Headline states the mechanism; measured outcomes remain in the manuscript.
text(4, 149.5, 'SPARE', size=14, weight='bold', color=BLUE)
text(26, 149.5, 'Selective prefill scheduling', size=11, weight='bold')
text(4, 142.5, 'Preserve output opportunities · Defer work only for sufficient KV block savings', size=8.5)

heading(4, 134.5, 1, 'Observe')
heading(57, 134.5, 2, 'Project & protect')
heading(117, 134.5, 3, 'Select budget')

# 1. Online state and pressure estimator.
box(4, 83, 46, 45)
text(7,123,'Online state',weight='bold',color=BLUE)
text(7,116,'Running + waiting requests')
text(7,110,'Input, progress, resident KV')
text(7,104,'Observed arrival history')
text(7,96,r'$\ell_{i,k}=P_i+o_{i,k}-c_{i,k}$',size=9)
text(7,88.5,r'$F_k=C-\sum_i b_{i,k}$',size=9)

arrow([(27,83),(27,78)],color=TEAL)
box(4,44,46,34,fill=TEAL_PALE,edge='#94BFB5')
text(7,73,'Input pressure',weight='bold',color=TEAL)
text(7,66.5,r'$W_k$: pending input',size=8.5)
text(7,60.5,r'$A_k$: recent input rate',size=8.5)
text(7,52.5,r'$Q_k=1+(W_k+\tau A_k)/B$',size=8.5)

# 2. Base projection fixes the reference allocation and priority.
arrow([(50,115),(57,115)])
box(57,104,53,24,fill=PALE)
text(60,123,'Default plan',weight='bold',color=BLUE)
text(60,116,r'$a_0=\mathrm{Project}(S_k,T,K)$',size=9)
text(60,109,'Same request order for all plans',size=8)
arrow([(83.5,104),(83.5,99)])
box(57,44,53,55,fill=PALE)
text(60,94,'Protect base allocations',weight='bold',color=BLUE)
text(60,87,'Normal decode',weight='bold')
tokens(60,80,['keep'])
text(66,81.5,'Next output token',size=8)
text(60,74,'Prefill finishing this round',weight='bold')
tokens(60,67,['keep','keep','keep'])
text(76,68.5,'First token',size=8)
text(60,60.5,r'$x_{i,k}(a)\geq x_{i,k}(a_0),\ i\in\mathcal{P}_k$',size=8.5)
tokens(60,48,['defer','defer'])
text(71,50,'Other work may wait',size=8,color=AMBER)

# 3. Ordered trial configurations and acceptance checks.
arrow([(110,115),(117,115)])
box(117,104,59,24)
text(120,123,'Project candidates in order',weight='bold',color=BLUE)
text(120,116.5,'1  (256, 256)    2  (512, 256)',size=8)
text(120,110.5,'3  (1024, 64)    4  (512, 64)',size=8)
arrow([(147,104),(147,99)])
box(117,44,59,55,fill=TEAL_PALE,edge='#94BFB5')
text(120,94,'Accept the first eligible plan',weight='bold',color=TEAL)
text(120,87.5,'Nonempty; feasible; no preemption')
text(120,81.5,'Protected allocations preserved')
text(120,75,r'$d_k(a)=G_k(a_0)-G_k(a)$',size=9)
text(120,68.5,r'$d_k(a)\geq n_k$',size=9,weight='bold',color=TEAL)
text(144,68.5,'physical blocks',size=8,color=TEAL)
text(120,61.5,'Base uses new blocks; ≥2 requests')
text(120,53.5,'No eligible plan / base preemption:',size=8)
text(120,47.5,r'use defaults $(T,K)$',size=8,weight='bold')
arrow([(110,81.5),(117,81.5)],color=BLUE)

# Pressure is a control input to the acceptance gate, not an execution stage.
arrow([(27,44),(27,36.5),(147,36.5),(147,44)],color=TEAL,dashed=True)
text(32,40.5,r'$n_k=\mathrm{clip}(\lceil n_0+\alpha F_k/Q_k\rceil,\,1,\,n_{\max})$',size=9,color=TEAL)
text(32,32,'Higher input pressure → smaller required block savings',size=8,color=TEAL)

# 4. Native execution and the loop to the next observed state.
box(57,8,119,20,fill=PALE,edge='#A8BBCD')
heading(60,22.5,4,'Execute with the native scheduler')
text(60,15.5,'Allocate tokens + KV blocks → run prefill / decode → release completed KV',size=8)
text(60,10.5,'Use selected budgets for this iteration; restore defaults afterward',size=8,color=MUTED)
arrow([(176,67),(179,67),(179,18),(176,18)],color=TEAL)
arrow([(57,18),(3,18),(3,137.5),(27,137.5),(27,128)],color=MUTED,dashed=True)
text(8,23,'Next iteration',size=8,weight='bold')
text(8,13,'Updated queues',size=8,color=MUTED)
text(8,8,'and KV state',size=8,color=MUTED)

fig.canvas.draw()
require_matplotlib_panel_alignment(fig, json_out=OUT/'spare_mechanism.alignment.json',
    tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
fig.savefig(OUT/'spare_mechanism.pdf',facecolor='white')
fig.savefig(OUT/'spare_mechanism.svg',facecolor='white')
fig.savefig(OUT/'spare_mechanism.png',dpi=600,facecolor='white')
plt.close(fig)
print(OUT/'spare_mechanism.pdf')
