"""Concise SPARE architecture schematic for the methodology overview.

The canvas and macro layout match architecture_v2 (180 x 137 mm): serving
path at the top, three SPARE components in the middle, and a physical KV
comparison below. Detailed equations and guards remain in the manuscript.
"""
from pathlib import Path
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/spare-architecture-v3-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch, Arc
from matplotlib.path import Path as MPath

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.home() / '.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'font.size': 8, 'mathtext.fontset': 'dejavusans',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.spines.top': False, 'axes.spines.right': False,
})

INK = '#20252B'
MUTED = '#5E6872'
ORANGE = '#C35C1D'
BLUE = '#315E9D'
WORK = '#5C83B1'
YELLOW = '#FFDE7A'
PALE = '#FFF5D5'
REGION = '#FFFDF4'
GRAY = '#C2C8CF'
SAVED = '#B67C21'

fig = plt.figure(figsize=(180 / 25.4, 137 / 25.4), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 180), ylim=(0, 137))
ax.set_axis_off()


def txt(x, y, s, size=8, ha='left', color=INK, bold=False, va='center'):
    return ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=color,
                   weight='bold' if bold else 'normal', linespacing=1.15)


def rect(x, y, w, h, fill='white', edge=INK, lw=.65, ls='-'):
    patch = Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                      linewidth=lw, linestyle=ls)
    ax.add_patch(patch)
    return patch


def rounded(x, y, w, h, fill='white', edge=INK, lw=.7, r=1.4, ls='-'):
    patch = FancyBboxPatch((x, y), w, h,
        boxstyle=f'round,pad=0,rounding_size={r}', facecolor=fill,
        edgecolor=edge, linewidth=lw, linestyle=ls)
    ax.add_patch(patch)
    return patch


def arrow(points, color=ORANGE, dash=False, lw=1.05, head=8):
    path = MPath(points, [MPath.MOVETO] + [MPath.LINETO] * (len(points) - 1))
    patch = FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=head,
        color=color, linewidth=lw,
        linestyle=(0, (3, 2.4)) if dash else '-')
    ax.add_patch(patch)
    return patch


def line(points, color=INK, lw=.6, ls='-'):
    ax.plot([p[0] for p in points], [p[1] for p in points],
            color=color, lw=lw, ls=ls)


def module(x, y, w, h, title):
    rounded(x, y, w, h, PALE)
    rounded(x, y + h - 9, w, 9, YELLOW)
    txt(x + w / 2, y + h - 4.5, title, size=8.7, ha='center', bold=True)


def cell(x, y, w=3.4, h=3, kind='work'):
    fill, edge, ls = {
        'work': (WORK, BLUE, '-'), 'resident': (GRAY, MUTED, '-'),
        'free': ('white', MUTED, '-'), 'defer': ('white', SAVED, (0, (2, 1.5))),
    }[kind]
    rect(x, y, w, h, fill, edge, lw=.6, ls=ls)


def lock(x, y):
    ax.add_patch(Arc((x + 1.2, y + 2), 1.8, 2.1,
                     theta1=0, theta2=180, color=BLUE, lw=.85))
    rounded(x, y, 2.4, 1.8, BLUE, BLUE, lw=.5, r=.25)
    ax.add_patch(Circle((x + 1.2, y + .95), .2,
                        facecolor='white', edgecolor='none'))


def person(x, y, scale=1):
    ax.add_patch(Circle((x, y + 3.4 * scale), 1.45 * scale,
                        facecolor=INK, edgecolor='none'))
    rounded(x - 2.4 * scale, y - 1.1 * scale, 4.8 * scale, 3 * scale,
            INK, INK, lw=.5, r=1)


def gpu(x, y):
    rect(x, y, 11, 10, fill='#F0F2F4', lw=.8)
    rect(x + 2, y + 2, 7, 6, fill='white', lw=.7)
    txt(x + 5.5, y + 5, 'GPU', size=7.8, ha='center', bold=True)
    for i in range(4):
        d = 1.4 + 2.3 * i
        line([(x - 1.2, y + d), (x, y + d)], lw=.8)
        line([(x + 11, y + d), (x + 12.2, y + d)], lw=.8)
        line([(x + d, y - 1.2), (x + d, y)], lw=.8)
        line([(x + d, y + 10), (x + d, y + 11.2)], lw=.8)


# Serving path. Labels sit directly on the process arrows.
person(9, 124, 1.05)
person(14, 126, .8)
txt(11, 117, 'Users', size=8.5, ha='center')
arrow([(19, 125), (29, 125)])
txt(22.5, 129.8, 'Requests', size=7.6, ha='center', color=ORANGE)

txt(45, 134, 'Request queues', size=8.7, ha='center', bold=True)
for row, y in enumerate((126.7, 120.3)):
    for j, count in enumerate((3, 5, 4)):
        x = 30 + j * 10.5
        rounded(x, y, 9, 4.1, '#F2F4F6', MUTED, lw=.55, r=.6)
        for k in range(count):
            rect(x + .8 + k * 1.3, y + 1, 1, 2,
                 WORK if row == 0 else '#E4B75B', edge='none', lw=0)
arrow([(61, 125), (78, 125)])
txt(69.5, 129.8, 'Queued work', size=7.6, ha='center', color=ORANGE)

rounded(79, 118.5, 42, 13, '#F3F4F5')
txt(100, 127.8, 'Native scheduler', size=8.7, ha='center', bold=True)
for j in range(8):
    cell(84 + j * 4.1, 120.5, w=3.1, h=2.3,
         kind='work' if j < 5 else 'free')
arrow([(121, 125), (140, 125)])
txt(130.5, 129.8, 'Execution', size=7.6, ha='center', color=ORANGE)

txt(159, 135, 'Serving engine', size=8.7, ha='center', bold=True)
gpu(143, 120)
for row in range(3):
    for col in range(3):
        cell(162 + col * 4.1, 120 + row * 3.7, w=3.3, h=2.8,
             kind='resident' if row + col < 3 else 'free')
txt(167.7, 116.2, 'KV cache', size=7.8, ha='center')


# SPARE retains the original three-component architecture.
rect(4, 22, 172, 86, REGION, MUTED, lw=.65, ls=(0, (2, 2)))
txt(8, 103.6, 'SPARE', size=10.5, bold=True)

arrow([(44, 118.5), (44, 96)], color=BLUE, dash=True)
txt(46, 111.5, 'Request state', size=7.8, color=BLUE)
arrow([(166, 113), (166, 96)], color=BLUE, dash=True)
txt(164, 111, 'KV + input state', size=7.8, ha='right', color=BLUE)
arrow([(99, 96), (99, 118.5)], color=ORANGE)
txt(96, 111, 'Budget + chunk', size=7.8, ha='right', color=ORANGE)

module(9, 55, 47, 41, 'Output protection')
for y, label, count, protected in (
        (77, 'Decode', 2, True),
        (68, 'Finishing prefill', 3, True),
        (59, 'Deferrable work', 4, False)):
    txt(12, y + 1.5, label, size=8)
    for j in range(count):
        cell(35 + j * 4.2, y, kind='work' if protected else 'defer')
    if protected:
        lock(51.5, y + .5)

module(67, 49, 55, 47, 'Budget selection')
txt(74, 82.5, 'Candidates', size=7.8, bold=True)
txt(113.5, 82.5, 'Chunk', size=7.8, bold=True, ha='center')
for j, (y, fraction, chunks) in enumerate(((76, .28, 4), (69, .5, 4),
                                            (62, 1, 2), (55, .5, 2)), 1):
    txt(71.5, y + 1.5, str(j), size=7.7, color=MUTED)
    rect(78, y, 27, 3, 'white', '#C5B77E', lw=.5)
    rect(78, y, 27 * fraction, 3, YELLOW, INK, lw=.5)
    for k in range(chunks):
        cell(109 + k * 3.4, y, w=2.6, h=3, kind='work')
txt(94.5, 51.8, 'First valid candidate', size=8.1, ha='center', bold=True)

module(133, 55, 41, 41, 'Pressure control')
txt(143, 82.5, 'Free KV', size=7.8, ha='center')
txt(164, 82.5, 'Input load', size=7.8, ha='center')
for row in range(2):
    for col in range(4):
        cell(137 + col * 3.3, 72 + row * 3.4, w=2.6, h=2.7,
             kind='resident' if col < 2 else 'free')
for j in range(3):
    rounded(157 + j * 4.1, 72 + j * 1.2, 3.2, 5,
            'white', SAVED, lw=.55, r=.3)
    line([(157.6 + j * 4.1, 73.3 + j * 1.2),
          (159.7 + j * 4.1, 73.3 + j * 1.2)], color=SAVED, lw=.55)
    line([(157.6 + j * 4.1, 75 + j * 1.2),
          (159.7 + j * 4.1, 75 + j * 1.2)], color=SAVED, lw=.55)
arrow([(145, 67), (162, 67)], color=BLUE, dash=True, head=6)
txt(153.5, 62.5, 'Savings threshold', size=8, ha='center', bold=True)

# Cross-component arrows carry the method logic.
arrow([(56, 73), (67, 73)], color=BLUE, dash=True)
txt(61.5, 77.2, 'Protect', size=7.3, ha='center', color=BLUE)
arrow([(133, 60), (122, 60)], color=BLUE, dash=True)
txt(127.5, 64.2, 'KV gate', size=7.3, ha='center', color=BLUE)


# Physical block comparison: the visible outcome of an accepted candidate.
txt(12, 45, 'Physical KV allocation', size=8.7, bold=True)
for y, label in ((36.5, 'Default'), (28.5, 'SPARE')):
    txt(12, y + 2, label, size=8.2, bold=(label == 'SPARE'))
    for j in range(12):
        kind = 'resident' if j < 4 else ('defer' if label == 'SPARE' and j >= 8 else 'work')
        cell(32 + j * 6, y, w=5, h=4, kind=kind)

arrow([(108, 30.5), (131, 30.5)], color=ORANGE, lw=1, head=7)
txt(119.5, 34.2, 'Avoid new blocks', size=7.8, ha='center', color=ORANGE)
rounded(132, 25.5, 39, 10, '#F2F6FB', '#A7BAD3', lw=.7, r=1)
txt(151.5, 30.5, 'More free KV', size=8.5, ha='center', bold=True, color=BLUE)


# Compact flow legend mirrors the reference figure's visual grammar.
arrow([(22, 14), (36, 14)], color=ORANGE)
txt(39, 14, 'Scheduling flow', size=7.8)
arrow([(96, 14), (110, 14)], color=BLUE, dash=True)
txt(113, 14, 'State / constraint', size=7.8)

fig.canvas.draw()
require_matplotlib_panel_alignment(
    fig, json_out=OUT / 'spare_architecture.alignment.json',
    tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
fig.savefig(OUT / 'spare_architecture.pdf', facecolor='white')
fig.savefig(OUT / 'spare_architecture.svg', facecolor='white')
fig.savefig(OUT / 'spare_architecture.png', dpi=600, facecolor='white')
fig.savefig(OUT / 'spare_architecture.preview.png', dpi=300, facecolor='white')
plt.close(fig)
print(OUT / 'spare_architecture.pdf')
