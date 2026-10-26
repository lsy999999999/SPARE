"""Original SPARE architecture schematic. Python/matplotlib only.

The graphical vocabulary uses system icons, request strips, a budget menu,
physical-block groups, and the method's analytical pressure response.
Illustrations do not encode benchmark observations or a numerical replay.
"""
from pathlib import Path
import math
import os
import sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp/spare-architecture-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch, Arc
from matplotlib.path import Path as MPath

OUT = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get('FIGURE_AUDIT_DIR', str(Path.home()/'.codex/skills/nature-figure/scripts')))
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({
    'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans'],
    'font.size':8, 'mathtext.fontset':'dejavusans',
    'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
    'axes.spines.top':False, 'axes.spines.right':False,
})
INK='#20252B'
MUTED='#5E6872'
ORANGE='#C35C1D'
BLUE='#315E9D'
WORK='#5C83B1'
YELLOW='#FFDE7A'
PALE='#FFF5D5'
REGION='#FFFDF4'
GRAY='#C2C8CF'
SAVED='#B67C21'
fig=plt.figure(figsize=(7.0866141732,5.3937007874),facecolor='white')  # 180 × 137 mm
ax=fig.add_axes([0,0,1,1])
ax.set(xlim=(0,180),ylim=(0,137))
ax.set_axis_off()

def txt(x,y,s,size=8,ha='left',color=INK,bold=False,va='center'):
    return ax.text(x,y,s,fontsize=size,ha=ha,va=va,color=color,
                   weight='bold' if bold else 'normal',linespacing=1.18)

def rect(x,y,w,h,fill='white',edge=INK,lw=.65,ls='-'):
    p=Rectangle((x,y),w,h,facecolor=fill,edgecolor=edge,linewidth=lw,linestyle=ls)
    ax.add_patch(p)
    return p

def rounded(x,y,w,h,fill='white',edge=INK,lw=.7,r=1.4):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                    facecolor=fill,edgecolor=edge,linewidth=lw)
    ax.add_patch(p)
    return p

def arr(points,color=ORANGE,dash=False,lw=1.05,head=8):
    path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1))
    p=FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=head,
                     color=color,linewidth=lw,linestyle=(0,(3,2.4)) if dash else '-')
    ax.add_patch(p)

def line(points,color=INK,lw=.6,ls='-'):
    ax.plot([p[0] for p in points],[p[1] for p in points],color=color,lw=lw,ls=ls)

def module(x,y,w,h,title):
    rounded(x,y,w,h,PALE)
    rounded(x,y+h-9,w,9,YELLOW)
    txt(x+w/2,y+h-4.5,title,size=8.2,ha='center',bold=True)

def cell(x,y,w=3.2,h=2.8,kind='work'):
    fill,edge={'work':(WORK,BLUE),'resident':(GRAY,MUTED),
               'free':('white',MUTED),'eligible':('white',SAVED),
               'defer':('white',SAVED)}[kind]
    rect(x,y,w,h,fill,edge,lw=.55)
    if kind=='defer':
        line([(x+.35,y+.3),(x+w-.35,y+h-.3)],color=SAVED,lw=.6)

def lock(x,y):
    ax.add_patch(Arc((x+1.2,y+2.0),1.8,2.1,theta1=0,theta2=180,color=BLUE,lw=.85))
    rounded(x,y,2.4,1.8,BLUE,BLUE,lw=.5,r=.25)
    ax.add_patch(Circle((x+1.2,y+.95),.22,facecolor='white',edgecolor='none'))

def person(x,y,scale=1):
    ax.add_patch(Circle((x,y+3.4*scale),1.45*scale,facecolor=INK,edgecolor='none'))
    rounded(x-2.4*scale,y-1.1*scale,4.8*scale,3.0*scale,INK,INK,lw=.5,r=1.0)

def gpu(x,y):
    rect(x,y,11,10,fill='#F0F2F4',lw=.8)
    rect(x+2,y+2,7,6,fill='white',lw=.7)
    txt(x+5.5,y+5,'GPU',size=7.5,ha='center',bold=True)
    for i in range(4):
        d=1.4+2.3*i
        line([(x-1.2,y+d),(x,y+d)],lw=.8)
        line([(x+11,y+d),(x+12.2,y+d)],lw=.8)
        line([(x+d,y-1.2),(x+d,y)],lw=.8)
        line([(x+d,y+10),(x+d,y+11.2)],lw=.8)

# External serving path: real execution stays in the native serving engine.
person(9,124,1.05)
person(14,126,.8)
txt(11,117,'Users',size=8.5,ha='center')
arr([(19,125),(29,125)])

txt(45,134,'Request queues',size=8.5,ha='center',bold=True)
for row,y in enumerate((126.7,120.3)):
    for j,n in enumerate((3,5,4)):
        x=30+j*10.5
        rounded(x,y,9,4.1,'#F2F4F6',MUTED,lw=.55,r=.6)
        for t in range(n):
            rect(x+.8+t*1.3,y+1,1,2,WORK if row==0 else '#E4B75B',edge='none',lw=0)
arr([(61,125),(78,125)])

rounded(79,118.5,42,13,'#F3F4F5')
txt(100,127.8,'Native scheduler',size=8.5,ha='center',bold=True)
for j in range(8):
    cell(84+j*4.1,120.5,w=3.1,h=2.3,kind='work' if j<5 else 'free')
arr([(121,125),(140,125)])

txt(159,135,'Serving engine',size=8.5,ha='center',bold=True)
gpu(143,120)
for r in range(3):
    for c in range(3):
        cell(162+c*4.1,120+r*3.7,w=3.3,h=2.8,kind='resident' if r+c<3 else 'free')
txt(167.7,116.2,'KV cache',size=7.5,ha='center')

# The controller boundary and its three actual method components.
rect(4,22,172,86,REGION,MUTED,lw=.65,ls=(0,(2,2)))
txt(8,103.6,'SPARE',size=10,bold=True)

# Separate state observations (blue dashed) from actions (orange solid).
arr([(44,118.5),(44,96)],color=BLUE,dash=True)
txt(46,112.5,'Request state',size=7.8,color=BLUE)
arr([(166,113),(166,96)],color=BLUE,dash=True)
txt(164,111,'Free KV + arrivals',size=7.8,ha='right',color=BLUE)
arr([(99,96),(99,118.5)],color=ORANGE)
txt(102,111,r'$(T_k,K_k)$',size=9,color=ORANGE)

module(9,55,51,41,'Output protection (§3.2)')
txt(34.5,84.5,r'$a_0=\mathrm{Project}(S_k,T,K)$',size=7.7,ha='center',color=MUTED)
txt(34.5,80.6,'Decode first; then shortest remaining',size=7.0,ha='center',color=MUTED)
for yy,label,count in ((76,'Normal decode',1),(69,'Finishing prefill',3),(62,'Other work',4)):
    txt(12,yy+1.2,label,size=7.7)
    for j in range(count):
        cell(35+j*4.25,yy,kind='eligible' if label=='Other work' else 'work')
    if label!='Other work':
        lock(54,yy+.4)
txt(44,58.2,'may defer',size=6.9,ha='center',color=SAVED)
arr([(60,73),(69,73)],color=BLUE,dash=True)
txt(64.5,77.2,r'$a_0,\mathcal{P}_k$',size=8.5,ha='center',color=BLUE)

rounded(9,47.4,51,6.2,'white',SAVED,lw=.6,r=.7)
txt(34.5,51.8,'Default if base plan needs preemption',
    size=6.8,ha='center',color=INK)
txt(34.5,49.1,'or adds no KV blocks or <2 requests',
    size=6.8,ha='center',color=INK)
arr([(60,50.5),(69,50.5)],color=ORANGE,lw=.8,head=6)

module(69,48,51,48,'Budget selection (§3.4)')
txt(94.5,85.0,r'Project$(S_k,b,c)$; same state/order',
    size=7.4,ha='center',color=MUTED)
txt(91,81.5,'Total budget',size=7.3,ha='center',color=MUTED)
txt(113.5,81.5,'Chunk',size=7.3,ha='center',color=MUTED)
for j,(yy,label,fraction,chunk) in enumerate(((77,'T/4',.25,'K'),(70.7,'T/2',.5,'K'),
                                           (64.4,'T',1,'K_{64}'),(58.1,'T/2',.5,'K_{64}')),1):
    txt(72.2,yy+1.2,str(j),size=7.5,color=MUTED)
    txt(77,yy+1.2,label,size=8)
    rect(89,yy,18,2.7,'white','#C5B77E',lw=.45)
    rect(89,yy,18*fraction,2.7,YELLOW,INK,lw=.45)
    txt(113.7,yy+1.2,f'${chunk}$',size=8.5,ha='center')
txt(94.5,53.6,r'Feasible · protected · $d_k\geq n_k$',size=7.7,ha='center')
txt(94.5,49.7,'First eligible; else default',size=7.7,ha='center',bold=True)

module(130,55,44,41,'Pressure control (§3.3)')
txt(140.7,83.3,'Free KV',size=7.5,ha='center',color=MUTED)
txt(162.2,83.3,'Queue + arrivals',size=7.0,ha='center',color=MUTED)
for r in range(2):
    for c in range(4):
        cell(134+c*3.4,73+r*3.4,w=2.7,h=2.7,kind='resident' if c<2 else 'free')
for j in range(3):
    rounded(155+j*4.3,74+j*1.2,3.4,5.1,'white',SAVED,lw=.55,r=.3)
    for k in range(2):
        line([(155.6+j*4.3,75.2+j*1.2+k*1.4),(157.6+j*4.3,75.2+j*1.2+k*1.4)],color=SAVED,lw=.5)
txt(140.7,68.7,r'$F_k$',size=9,ha='center')
txt(162.2,68.7,r'$W_k+\tau A_k$',size=9,ha='center')
txt(152,59.8,r'$F_k/Q_k\;\longrightarrow\;n_k$',size=9,ha='center')
arr([(130,59.8),(120,59.8)],color=BLUE,dash=True)
txt(125,64,r'$n_k$',size=9,ha='center',color=BLUE)

# Physical allocation, rather than token counts, is the decision quantity.
# Ellipses make the block groups visibly non-numerical; this is not a trial.
txt(12,44.8,'Physical KV allocation',size=8.1,bold=True)
txt(81,44.8,'(schematic groups, not counts)',size=7.3,color=MUTED)
for yy,label in ((36.7,'Default'),(29.6,'Candidate')):
    txt(12,yy+1.5,label,size=8)
    for j in range(12):
        kind='resident' if j<4 else ('defer' if label=='Candidate' and j>=8 else 'work')
        cell(32+j*6,yy,w=5,h=3.8,kind=kind)
    txt(109,yy+1.9,r'$\cdots$',size=11,ha='center',color=MUTED)
line([(32,27.7),(32,26.6),(55,26.6),(55,27.7)],color=MUTED)
txt(43.5,24,'Same resident KV',size=7.3,ha='center',color=MUTED)
line([(80,27.7),(80,26.6),(115,26.6),(115,27.7)],color=SAVED)
txt(97.5,24,r'$d_k=G_k(a_0)-G_k(a)$',size=7.3,ha='center',color=SAVED)
arr([(116,31.5),(124,31.5),(124,52.3),(120,52.3)],color=BLUE,dash=True)
txt(126,36.5,r'$d_k$',size=9,color=BLUE)

# Theoretical response of Equation (11), at fixed free KV; not empirical data.
txt(152,46,'Pressure response',size=8.1,ha='center',bold=True)
arr([(137,29.5),(173,29.5)],color=INK,lw=.65,head=5)
arr([(137,29.5),(137,43)],color=INK,lw=.65,head=5)
txt(134.7,41.5,r'$n_k$',size=9,ha='right')
txt(172.5,26.4,r'$Q_k$',size=9,ha='center')
ratios=[1+11*i/150 for i in range(151)]
thresholds=[math.ceil(8+128/q) for q in ratios]
xx=[139+31*(q-1)/11 for q in ratios]
yy=[32+9*(n-min(thresholds))/(max(thresholds)-min(thresholds)) for n in thresholds]
ax.step(xx,yy,where='post',color=BLUE,lw=1.15)
txt(150.5,24,'Fixed free KV',size=7.3,ha='center',color=MUTED)

# Legend: line semantics and visual encodings, outside the controller boundary.
arr([(9,15),(23,15)],color=ORANGE)
txt(26,15,'Request / execution / budget flow',size=7.7)
arr([(105,15),(119,15)],color=BLUE,dash=True)
txt(122,15,'Observed / projected state',size=7.7)
cell(9,6.1,w=3.8,h=3,kind='work')
txt(15,7.6,'Scheduled work / new KV',size=7.7)
cell(71,6.1,w=3.8,h=3,kind='eligible')
txt(77,7.6,'May defer',size=7.7)
cell(103,6.1,w=3.8,h=3,kind='defer')
txt(109,7.6,'New KV avoided',size=7.7)
cell(149,6.1,w=3.8,h=3,kind='resident')
txt(155,7.6,'Resident KV',size=7.7)

fig.canvas.draw()
require_matplotlib_panel_alignment(fig,json_out=OUT/'spare_architecture.alignment.json',
    tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
fig.savefig(OUT/'spare_architecture.pdf',facecolor='white')
fig.savefig(OUT/'spare_architecture.svg',facecolor='white')
fig.savefig(OUT/'spare_architecture.png',dpi=600,facecolor='white')
fig.savefig(OUT/'spare_architecture.preview.png',dpi=300,facecolor='white')
plt.close(fig)
print(OUT/'spare_architecture.pdf')
