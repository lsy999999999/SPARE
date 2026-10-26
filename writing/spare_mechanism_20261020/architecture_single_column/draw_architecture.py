"""SPARE scheduling architecture at its final 85 mm column width.

Mechanism schematic based on manuscript Sections 3.1–3.4 and Algorithm 1.
Block groups illustrate allocation categories, not numerical observations.
"""
from pathlib import Path
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/spare-architecture-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Arc, Circle
from matplotlib.path import Path as MPath

OUT = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get('FIGURE_AUDIT_DIR', str(Path.home() / '.codex/skills/nature-figure/scripts')))
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'font.size': 8, 'mathtext.fontset': 'dejavusans',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
})
INK = '#20252B'
MUTED = '#5E6872'
BLUE = '#315E9D'
WORK = '#5C83B1'
ORANGE = '#C35C1D'
SAVED = '#B67C21'
GRAY = '#C2C8CF'
YELLOW = '#FFDE7A'
PALE = '#FFF5D5'
fig = plt.figure(figsize=(3.3464566929, 5.1968503937), facecolor='white')  # 85 × 132 mm
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 85), ylim=(0, 132))
ax.set_axis_off()


def text(x, y, value, size=8, ha='left', color=INK, bold=False):
    return ax.text(x, y, value, fontsize=size, ha=ha, va='center',
                   color=color, weight='bold' if bold else 'normal')


def box(x, y, w, h, fill='white', edge=INK, lw=.65, radius=1.2):
    item = FancyBboxPatch((x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={radius}',
        facecolor=fill, edgecolor=edge, linewidth=lw)
    ax.add_patch(item)
    return item


def arrow(points, color=BLUE, dashed=False, head=6, lw=.85):
    path = MPath(points, [MPath.MOVETO] + [MPath.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-' if head == 0 else '-|>',
        mutation_scale=max(1, head), color=color, linewidth=lw,
        linestyle=(0, (3, 2.4)) if dashed else '-'))


def module(x, y, w, h, title):
    box(x, y, w, h, fill=PALE)
    box(x, y + h - 7, w, 7, fill=YELLOW)
    text(x + w / 2, y + h - 3.5, title, size=8.1, ha='center', bold=True)


def cell(x, y, kind='work', w=3, h=2.6):
    fill, edge = {
        'work': (WORK, BLUE), 'resident': (GRAY, MUTED),
        'eligible': ('white', SAVED), 'avoided': ('white', SAVED),
    }[kind]
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=.55))
    if kind == 'avoided':
        ax.plot([x + .3, x + w - .3], [y + .3, y + h - .3], color=SAVED, lw=.6)


def lock(x, y):
    ax.add_patch(Arc((x + .9, y + 1.5), 1.4, 1.6, theta1=0, theta2=180,
                     color=BLUE, lw=.75))
    box(x, y, 1.8, 1.5, fill=BLUE, edge=BLUE, lw=.4, radius=.2)
    ax.add_patch(Circle((x + .9, y + .75), .16, facecolor='white', edgecolor='none'))


# Runtime observations feed both controller branches.
box(5, 118, 75, 12, fill='#F3F4F5', edge=MUTED)
text(42.5, 126.6, 'Observed runtime state', size=8.5, ha='center', bold=True)
text(42.5, 122, 'Queues · free KV · recent arrivals', size=7.8, ha='center')
boundary = Rectangle((3, 14.5), 79, 98.5, facecolor='#FFFDF4',
                     edgecolor=MUTED, linewidth=.6, linestyle=(0, (2, 2)))
ax.add_patch(boundary)
text(6, 110, 'SPARE', size=9, bold=True)
text(43, 110, 'Per iteration', size=7.5, ha='center', color=MUTED)
arrow([(26, 118), (26, 105)], dashed=True)
arrow([(65.5, 118), (65.5, 105)], dashed=True)

# Base-plan allocations are protected; other work is eligible for deferral.
module(7, 75, 42, 30, '1  Protect output')
text(28, 95, r'$a_0=\mathrm{Project}(S_k,T,K)$', size=7.8, ha='center', color=MUTED)
for yy, label, count in [(89, 'Normal decode', 1), (83.5, 'Finishing prefill', 3),
                         (78, 'Other: may defer', 3)]:
    text(9, yy + 1.3, label, size=7.2)
    for i in range(count):
        cell(34 + 3.4 * i, yy, kind='eligible' if yy == 78 else 'work', w=2.7)
    if yy != 78:
        lock(45.5, yy + .3)

# Free space and observed input pressure determine one shared threshold.
module(53, 75, 25, 30, '2  Threshold')
text(65.5, 94.5, r'Free KV $F_k$', size=7.8, ha='center')
text(65.5, 89, 'Pending input', size=7.5, ha='center')
text(65.5, 85, '+ recent arrivals', size=7.2, ha='center')
text(65.5, 79.3, r'$F_k/Q_k\;\to\;n_k$', size=8.2, ha='center', color=BLUE)

arrow([(28, 75), (28, 68.5), (42.5, 68.5)], dashed=True, head=0)
arrow([(65.5, 75), (65.5, 68.5), (42.5, 68.5)], dashed=True, head=0)
arrow([(42.5, 68.5), (42.5, 66)], dashed=True)
text(12, 70.8, r'$a_0,\mathcal{P}_k$', size=8.3, color=BLUE)
text(69, 70.8, r'$n_k$', size=8.3, color=BLUE)

# A single selection module combines projection, protection, and block savings.
module(7, 23.2, 71, 42.8, '3  Project candidates and select')
text(42.5, 55.4, 'Ordered configurations (Table 1)', size=7.8, ha='center')
text(42.5, 51.1, r'Project$(S_k,b,c)$ · same state and order', size=7.7,
     ha='center', color=MUTED)
text(42.5, 46.8, 'Feasible, nonempty · preserve protected work', size=7.2, ha='center')
text(42.5, 42.5, r'First with $d_k\geq n_k$; else default $(T,K)$',
     size=8, ha='center', bold=True)

# Categories of physical KV blocks; ellipses explicitly avoid numerical counts.
text(33.5, 38.6, 'Resident', size=6.8, ha='center', color=MUTED)
text(44, 38.6, 'New', size=6.8, ha='center', color=BLUE)
text(59, 38.6, 'Avoided', size=6.8, ha='center', color=SAVED)
for yy, label in [(33.5, 'KV: base'), (28.5, 'Candidate')]:
    text(10, yy + 1.5, label, size=7.5)
    for j in range(7):
        kind = 'resident' if j < 2 else 'work'
        if label == 'Candidate' and j >= 4:
            kind = 'avoided'
        cell(29 + j * 5, yy, kind=kind, w=4, h=3)
    text(68.2, yy + 1.5, r'$\cdots$', size=10, ha='center', color=MUTED)
text(42.5, 25.1, r'KV groups: $d_k=G_k(a_0)-G_k(a)$', size=7.6, ha='center')

# Exceptional base states select the default configuration directly.
text(8, 20.1, 'Default if base preempts / no new blocks', size=7.2)
text(8, 16.5, 'or fewer than two unfinished requests.', size=7.2)
arrow([(72, 23.2), (72, 11.5)], color=ORANGE)
box(7, 1.5, 71, 10, fill='#F3F4F5', edge=MUTED)
text(42.5, 8.4, 'Native scheduler + GPU', size=8.3, ha='center', bold=True)
text(42.5, 4.4, r'Execute $(T_k,K_k)$; update state', size=7.8, ha='center')
arrow([(7, 6.5), (1.6, 6.5), (1.6, 124), (5, 124)], dashed=True, lw=.7)

fig.canvas.draw()
require_matplotlib_panel_alignment(fig, json_out=OUT / 'spare_architecture.alignment.json',
    tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
fig.savefig(OUT / 'spare_architecture.pdf', facecolor='white')
fig.savefig(OUT / 'spare_architecture.svg', facecolor='white')
fig.savefig(OUT / 'spare_architecture.png', dpi=600, facecolor='white')
fig.savefig(OUT / 'spare_architecture.preview.png', dpi=300, facecolor='white')
plt.close(fig)
print(OUT / 'spare_architecture.pdf')
