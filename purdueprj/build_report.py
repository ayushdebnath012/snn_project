"""Build PAPER_REPORT.pdf from PAPER_REPORT.md using ReportLab."""
import re, os, sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image, Preformatted, HRFlowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, 'PAPER_REPORT.md'), encoding='utf-8') as f:
    md = f.read()

doc = SimpleDocTemplate(
    os.path.join(HERE, 'PAPER_REPORT.pdf'), pagesize=A4,
    leftMargin=2.5*cm, rightMargin=2.5*cm,
    topMargin=2.5*cm, bottomMargin=2.5*cm
)

styles = getSampleStyleSheet()
body   = ParagraphStyle('body', parent=styles['Normal'], fontSize=10.5,
           leading=15, spaceAfter=5, alignment=TA_JUSTIFY)
h1     = ParagraphStyle('h1', parent=styles['Heading1'], fontSize=16,
           spaceAfter=10, spaceBefore=18)
h2     = ParagraphStyle('h2', parent=styles['Heading2'], fontSize=13,
           spaceAfter=8, spaceBefore=14)
h3     = ParagraphStyle('h3', parent=styles['Heading3'], fontSize=11.5,
           spaceAfter=6, spaceBefore=10)
code_s = ParagraphStyle('code', fontName='Courier', fontSize=8.5,
           leading=12, backColor=colors.HexColor('#f4f4f4'),
           spaceAfter=6, leftIndent=12, borderPad=4)
cap_s  = ParagraphStyle('cap', parent=body, fontSize=9,
           textColor=colors.grey, alignment=TA_CENTER, spaceAfter=10)
bul_s  = ParagraphStyle('bul', parent=body, leftIndent=20, firstLineIndent=-12)


def inline(text):
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*',     r'<i>\1</i>', text)
    text = re.sub(r'`(.+?)`',       r'<font name="Courier">\1</font>', text)
    text = text.replace('×', '&times;').replace('→', '&#8594;')
    return text


story     = []
lines     = md.split('\n')
i         = 0
in_code   = False
code_buf  = []
table_buf = []

while i < len(lines):
    line = lines[i]

    # ── code block ────────────────────────────────────────────────────────────
    if line.startswith('```'):
        if not in_code:
            in_code = True
            code_buf = []
        else:
            in_code = False
            story.append(Preformatted('\n'.join(code_buf), code_s))
        i += 1
        continue
    if in_code:
        code_buf.append(line)
        i += 1
        continue

    # ── image ─────────────────────────────────────────────────────────────────
    img_m = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line.strip())
    if img_m:
        alt, path = img_m.group(1), img_m.group(2)
        if not os.path.isabs(path):
            path = os.path.join(HERE, path)
        try:
            im = Image(path, width=14*cm, height=8*cm, kind='proportional')
            story.append(im)
            story.append(Paragraph(f'<i>{inline(alt)}</i>', cap_s))
        except Exception as e:
            story.append(Paragraph(f'[Image not found: {path}]', body))
        i += 1
        continue

    # ── table row ─────────────────────────────────────────────────────────────
    if line.startswith('|'):
        cells = [c.strip() for c in line.strip('|').split('|')]
        if all(re.match(r'^[-: ]+$', c) for c in cells):
            i += 1
            continue
        table_buf.append([inline(c) for c in cells])
        next_is_table = (i + 1 < len(lines) and lines[i+1].startswith('|'))
        if not next_is_table and table_buf:
            ncols = len(table_buf[0])
            col_w = [14*cm / ncols] * ncols
            tw = Table([[Paragraph(c, ParagraphStyle('tc', fontName='Helvetica',
                          fontSize=9, leading=12)) for c in row]
                        for row in table_buf],
                       colWidths=col_w, repeatRows=1)
            tw.setStyle(TableStyle([
                ('BACKGROUND',    (0, 0), (-1, 0), colors.HexColor('#e0e0e0')),
                ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE',      (0, 0), (-1,-1), 9),
                ('GRID',          (0, 0), (-1,-1), 0.4, colors.grey),
                ('ROWBACKGROUNDS',(0, 1), (-1,-1),
                 [colors.white, colors.HexColor('#f7f7f7')]),
                ('LEFTPADDING',   (0, 0), (-1,-1), 4),
                ('RIGHTPADDING',  (0, 0), (-1,-1), 4),
                ('TOPPADDING',    (0, 0), (-1,-1), 3),
                ('BOTTOMPADDING', (0, 0), (-1,-1), 3),
                ('VALIGN',        (0, 0), (-1,-1), 'MIDDLE'),
            ]))
            story.append(tw)
            story.append(Spacer(1, 8))
            table_buf = []
        i += 1
        continue

    # ── headings ──────────────────────────────────────────────────────────────
    if re.match(r'^# [^#]', line):
        story.append(Paragraph(inline(line[2:]), h1))
        i += 1; continue
    if re.match(r'^## [^#]', line):
        story.append(HRFlowable(width='100%', thickness=0.5,
                                color=colors.grey, spaceAfter=4))
        story.append(Paragraph(inline(line[3:]), h2))
        i += 1; continue
    if re.match(r'^### ', line):
        story.append(Paragraph(inline(line[4:]), h3))
        i += 1; continue

    # ── italic caption line (starts with *) ───────────────────────────────────
    if re.match(r'^\*[^*]', line) and line.rstrip().endswith('*'):
        story.append(Paragraph(f'<i>{inline(line.strip("*").strip())}</i>', cap_s))
        i += 1; continue

    # ── bullet ────────────────────────────────────────────────────────────────
    if line.startswith('- '):
        story.append(Paragraph('&bull; ' + inline(line[2:]), bul_s))
        i += 1; continue

    # ── numbered list ─────────────────────────────────────────────────────────
    nl_m = re.match(r'^(\d+)\. (.*)', line)
    if nl_m:
        story.append(Paragraph(f'{nl_m.group(1)}. {inline(nl_m.group(2))}', bul_s))
        i += 1; continue

    # ── horizontal rule ───────────────────────────────────────────────────────
    if re.match(r'^---+\s*$', line):
        story.append(HRFlowable(width='100%', thickness=0.7,
                                color=colors.grey,
                                spaceBefore=6, spaceAfter=6))
        i += 1; continue

    # ── blank line ────────────────────────────────────────────────────────────
    if not line.strip():
        story.append(Spacer(1, 5))
        i += 1; continue

    # ── normal paragraph ──────────────────────────────────────────────────────
    story.append(Paragraph(inline(line), body))
    i += 1

doc.build(story)
print(f'Written: {os.path.join(HERE, "PAPER_REPORT.pdf")}')
