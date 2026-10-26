"""SPARE mechanism overview at the manuscript's single-column width.

Contract: selective deferral preserves protected output opportunities while
reducing new physical KV allocation when the pressure-dependent gate is met.
One schematic canvas: constraints -> budget decision -> native execution;
the block comparison is conceptual, not measured data or a numerical example.
Python/matplotlib backend. Final size: 85 x 105 mm; minimum text: 8.3 pt.
"""
from pathlib import Path
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/spare-simplified-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Arc, Circle, FancyArrowPatch
from matplotlib.path import Path as MPath

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.home() / '.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'font.size': 8.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False,
})
INK = '#20252B'
MUTED = '#5E6872'
BLUE = '#315E9D'
WORK = '#5C83B1'
ORANGE = '#C35C1D'
YELLOW = '#FFDE7A'
PALE = '#FFF5D5'

fig = plt.figure(figsize=(85 / 25.4, 105 / 25.4), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 85), ylim=(0, 105))
ax.set_axis_off()


def txt(x, y, text, size=8.5, bold=False, color=INK, ha='left'):
    return ax.text(x, y, text, ha=ha, va='center', fontsize=size,
                   weight='bold' if bold else 'normal', color=color)


def box(x, y, w, h, fill='white', edge=INK, lw=.7, radius=1.5, dash=False):
    patch = FancyBboxPatch((x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={radius}',
        facecolor=fill, edgecolor=edge, linewidth=lw,
        linestyle=(0, (3, 2.5)) if dash else '-')
    ax.add_patch(patch)
    return patch


def arrow(points, color=BLUE, dash=False, lw=1):
    path = MPath(points, [MPath.MOVETO] + [MPath.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>',
        mutation_scale=8, color=color, linewidth=lw,
        linestyle=(0, (3, 2.4)) if dash else '-'))


def lock(x, y):
    ax.add_patch(Arc((x + 1.1, y + 1.7), 1.6, 1.9,
                    theta1=0, theta2=180, color=BLUE, lw=.8))
    box(x, y, 2.2, 1.8, BLUE, BLUE, radius=.25, lw=.5)
    ax.add_patch(Circle((x + 1.1, y + .85), .17, facecolor='white', edgecolor='none'))


# SPARE's two inputs meet at one dominant decision block.
box(2, 47, 81, 56, '#FFFDF4', MUTED, lw=.65, dash=True)
txt(5, 99, 'SPARE', size=11, bold=True)

box(5, 73, 36, 21, '#F2F6FB', '#A7BAD3')
box(44, 73, 36, 21, '#F2F6FB', '#A7BAD3')
txt(23, 90.3, 'Protect outputs', size=8.8, bold=True, ha='center')
lock(8, 82.7)
txt(12, 83.6, 'Decode', size=8.3)
lock(8, 76.6)
txt(12, 77.5, 'Finishing prefill', size=8.3)

txt(62, 90.3, 'KV pressure', size=8.8, bold=True, ha='center')
txt(62, 84.7, 'Free KV', size=8.3, ha='center')
txt(62, 80.5, 'Pending input', size=8.3, ha='center')
txt(62, 76.3, 'Recent arrivals', size=8.3, ha='center')

arrow([(23, 73), (23, 66)], dash=True)
arrow([(62, 73), (62, 66)], dash=True)
box(7, 50, 71, 16, YELLOW, INK, lw=.9)
txt(42.5, 60.6, 'Selectively defer work', size=10.5, bold=True, ha='center')
txt(42.5, 54.3, 'Token budget + chunk size', size=8.7, ha='center')

# The native scheduler remains responsible for real execution.
arrow([(42.5, 50), (42.5, 41.8)], color=ORANGE, lw=1.2)
box(7, 31.5, 47, 10, '#F3F4F5')
txt(30.5, 36.5, 'Native scheduler', size=9, bold=True, ha='center')
arrow([(54, 36.5), (64, 36.5)], color=ORANGE, lw=1.2)
box(64, 31.5, 14, 10, '#F3F4F5', radius=.5)
txt(71, 36.5, 'GPU', size=9, bold=True, ha='center')

# A direct visual comparison communicates the purpose of deferral.
txt(42.5, 25.5, 'Fewer new KV blocks', size=10, bold=True, ha='center')
for yy, label in ((16.5, 'Default'), (7.5, 'SPARE')):
    txt(7, yy + 1.9, label, size=8.7, bold=(label == 'SPARE'))
    for j in range(7):
        x = 29 + j * 6.6
        avoided = label == 'SPARE' and j >= 4
        ax.add_patch(Rectangle((x, yy), 5.3, 3.8,
            facecolor='white' if avoided else WORK,
            edgecolor=ORANGE if avoided else BLUE,
            linewidth=.7, linestyle=(0, (2, 1.7)) if avoided else '-'))

fig.canvas.draw()
require_matplotlib_panel_alignment(fig, json_out=OUT / 'spare_architecture.alignment.json',
    tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
fig.savefig(OUT / 'spare_architecture.pdf', facecolor='white')
fig.savefig(OUT / 'spare_architecture.svg', facecolor='white')
fig.savefig(OUT / 'spare_architecture.png', dpi=600, facecolor='white')
fig.savefig(OUT / 'spare_architecture.preview.png', dpi=300, facecolor='white')
plt.close(fig)
print(OUT / 'spare_architecture.pdf')
