"""Typeset the audited paired results as a standalone PDF table."""
import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'writing/www/manuscript/sensitivity_capacity_20261024'
KVS=(1024,2048,4096)
ARMS=('spare','n0_4','n0_16','alpha64','alpha256','H05','H2','tau05','tau2')
LABELS=('Default','n₀ = 4','n₀ = 16','α = 64','α = 256','H = 0.5 s','H = 2 s','τ = 0.5 s','τ = 2 s')


def main():
    fonts=Path('/usr/share/fonts/truetype/dejavu')
    pdfmetrics.registerFont(TTFont('DejaVu',str(fonts/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('DejaVuBold',str(fonts/'DejaVuSans-Bold.ttf')))
    rows=list(csv.DictReader((OUT/'combined.csv').open()))
    index={(r['source'],int(r['kv_blocks']),r['arm'],float(r['theta_s'])):r for r in rows}
    for theta in (2.,4.):
        doc=SimpleDocTemplate(str(OUT/f'table_{theta:g}s.pdf'),pagesize=landscape(A4),
                              leftMargin=30,rightMargin=30,topMargin=24,bottomMargin=20)
        title=ParagraphStyle('title',fontName='DejaVuBold',fontSize=14,leading=19)
        note=ParagraphStyle('note',fontName='DejaVu',fontSize=8.3,leading=12)
        story=[Paragraph(f'Parameter sensitivity: capacity and matched-load latency ({theta:g}-s limit)',title),
               Spacer(1,7),Paragraph('Same KV budgets and 24 paired windows per workload. '
               'GPU-fitted CPU simulation; 60 s of arrivals with 20 s of warm-up.',note),Spacer(1,12)]
        cells=[['Configuration','KV = 1,024','','KV = 2,048','','KV = 4,096',''],
               ['', 'Capacity ↑ (QPS)','TTFT ↓ (ms)','Capacity ↑ (QPS)','TTFT ↓ (ms)',
                'Capacity ↑ (QPS)','TTFT ↓ (ms)']]
        style=[('FONTNAME',(0,0),(-1,-1),'DejaVu'),('FONTSIZE',(0,0),(-1,-1),8.5),
               ('FONTNAME',(0,0),(-1,1),'DejaVuBold'),('ALIGN',(1,0),(-1,1),'CENTER'),
               ('ALIGN',(1,2),(-1,-1),'RIGHT'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
               ('SPAN',(1,0),(2,0)),('SPAN',(3,0),(4,0)),('SPAN',(5,0),(6,0)),
               ('SPAN',(0,0),(0,1)),('LINEABOVE',(0,0),(-1,0),1,colors.black),
               ('LINEBELOW',(0,1),(-1,1),.6,colors.black),
               ('LINEBELOW',(0,-1),(-1,-1),1,colors.black),
               ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8)]
        for source in ('code','conversation'):
            qps=' / '.join(f"{float(index[source,k,'spare',theta]['matched_qps']):g}" for k in KVS)
            n=len(cells)
            cells.append([f'{source.title()} — TTFT at {qps} QPS for the three KV budgets','','','','','',''])
            style += [('SPAN',(0,n),(-1,n)),('BACKGROUND',(0,n),(-1,n),colors.HexColor('#eef2f5')),
                      ('FONTNAME',(0,n),(-1,n),'DejaVuBold'),('ALIGN',(0,n),(-1,n),'LEFT')]
            for arm,label in zip(ARMS,LABELS):
                row=[label]
                for kv in KVS:
                    r=index[source,kv,arm,theta]
                    row += [f"{float(r['capacity_qps']):.2f}",f"{float(r['mean_ttft_ms']):.1f}"]
                if arm=='spare':style.append(('FONTNAME',(0,len(cells)),(-1,len(cells)),'DejaVuBold'))
                cells.append(row)
        table=Table(cells,colWidths=[122]+[109]*6,rowHeights=[19]*2+[17.5]*20)
        table.setStyle(TableStyle(style));story.append(table);story.append(Spacer(1,10))
        story.append(Paragraph('Default: n₀ = 8, α = 128, H = τ = 1 s. Each row changes one parameter. '
            'Bold marks the default, not the best value. Capacity also requires mean window P99 TBT ≤ 200 ms '
            'and mean backlog growth ≤ max(2, QPS). Capacity boundaries are resolved to 0.01–0.02 QPS; '
            'this is search resolution, not a confidence interval.',note))
        doc.build(story)
        print(OUT/f'table_{theta:g}s.pdf')


if __name__=='__main__':main()
