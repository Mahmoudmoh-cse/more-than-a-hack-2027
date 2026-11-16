"""Build the technical PDF from the reviewed Markdown, without match-data access."""

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                                Preformatted, Table, TableStyle)


ROOT = Path(__file__).resolve().parents[1]


def markup(text):
    text = escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', text)
    return text


def main():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyReport', fontName='Helvetica',fontSize=9.1,
                              leading=12.3,spaceAfter=7,textColor=colors.HexColor('#233346')))
    styles.add(ParagraphStyle(name='CodeReport',fontName='Courier',fontSize=7.7,
                              leading=11,spaceAfter=10,backColor=colors.HexColor('#eef3f7'),
                              borderPadding=7))
    styles['Title'].fontSize = 28
    styles['Title'].leading = 32
    styles['Title'].alignment = TA_LEFT
    styles['Title'].textColor = colors.HexColor('#123b5d')
    styles['Heading1'].fontSize = 17
    styles['Heading1'].leading = 21
    styles['Heading1'].spaceAfter = 13
    styles['Heading2'].fontSize = 11.5
    styles['Heading2'].leading = 15
    styles['Heading2'].spaceBefore = 9
    for name in ('Heading1','Heading2'):
        styles[name].textColor = colors.HexColor('#123b5d')
    story = []
    source = (ROOT/'GOALGORITHM_TECHNICAL_REPORT.md').read_text(encoding='utf-8')
    for block in re.split(r'\n\s*\n',source.strip()):
        if block.strip() == '<!-- pagebreak -->':
            story.append(PageBreak())
        elif block.startswith('```'):
            story.append(Preformatted('\n'.join(block.splitlines()[1:-1]),styles['CodeReport']))
        elif block.startswith('|'):
            rows = []
            for line in block.splitlines():
                if re.match(r'^\|[\s|:-]+\|$',line):
                    continue
                rows.append([Paragraph(markup(value.strip()),styles['BodyReport'])
                             for value in line.strip('|').split('|')])
            table = Table(rows,colWidths=[235,128,128],hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eef5')),
                ('LINEBELOW',(0,0),(-1,0),1,colors.HexColor('#123b5d')),
                ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#cad5df')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),
                ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
                ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),2)]))
            story.extend([table,Spacer(1,10)])
        elif block.startswith('#'):
            for line in block.splitlines():
                count = len(line)-len(line.lstrip('#'))
                style = {1:'Title',2:'Heading1',3:'Heading2'}[min(count,3)]
                story.append(Paragraph(markup(line[count:].strip()),styles[style]))
        else:
            story.append(Paragraph(markup(' '.join(block.splitlines())),styles['BodyReport']))
    def page(canvas, doc):
        canvas.saveState()
        width,height=A4
        canvas.setStrokeColor(colors.HexColor('#cad5df'))
        canvas.line(48,45,width-48,45)
        canvas.setFont('Helvetica',8)
        canvas.setFillColor(colors.HexColor('#526679'))
        canvas.drawString(48,31,'GOALGORITHM  |  PHASE 1  |  IMPLEMENTATION AND SYNTHETIC QA')
        canvas.drawRightString(width-48,31,str(doc.page))
        canvas.restoreState()
    document = SimpleDocTemplate(str(ROOT/'Goalgorithm.pdf'),pagesize=A4,
                                 leftMargin=48,rightMargin=48,topMargin=43,bottomMargin=58,
                                 title='Goalgorithm - Phase 1 Technical Report',author='Goalgorithm')
    document.build(story,onFirstPage=page,onLaterPages=page)
    print('Created Goalgorithm.pdf from the technical Markdown.')


if __name__ == '__main__':
    main()
