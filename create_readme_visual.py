"""
Create a visually striking README poster for PM2.5 CMAQ Data Fusion project.
Precision Data design philosophy: Swiss grid, clean hierarchy, spatial authority.
"""
import os

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm, cm
    from reportlab.lib.colors import HexColor
    from reportlab.pdfgen import canvas
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def create_with_reportlab(output_path):
    """Create PDF poster using ReportLab."""
    w, h = A4  # 210mm x 297mm
    c = canvas.Canvas(output_path, pagesize=A4)

    # Colors
    BG = HexColor('#0D1117')
    SURFACE = HexColor('#161B22')
    BORDER = HexColor('#30363D')
    TEXT = HexColor('#E6EDF3')
    MUTED = HexColor('#8B949E')
    ACCENT = HexColor('#58A6FF')
    GREEN = HexColor('#3FB950')
    ORANGE = HexColor('#D29922')

    # Background
    c.setFillColor(BG)
    c.rect(0, 0, w, h, fill=1, stroke=0)

    # Top accent line
    c.setFillColor(ACCENT)
    c.rect(0, h - 4*mm, w, 4*mm, fill=1, stroke=0)

    # Title area
    y = h - 35*mm
    c.setFillColor(TEXT)
    c.setFont('Helvetica-Bold', 32)
    c.drawCentredString(w/2, y, 'PM2.5 · CMAQ Data Fusion')

    y -= 12*mm
    c.setFillColor(MUTED)
    c.setFont('Helvetica', 13)
    c.drawCentredString(w/2, y, '自动化 PM2.5 数据融合研究系统')

    y -= 10*mm
    c.setFont('Helvetica', 10)
    c.setFillColor(ACCENT)
    flow = '文献分析  →  方案设计  →  代码实现  →  测试验证  →  论文生成'
    c.drawCentredString(w/2, y, flow)

    # Divider
    y -= 12*mm
    c.setStrokeColor(BORDER)
    c.setLineWidth(0.5)
    c.line(25*mm, y, w - 25*mm, y)

    # Pipeline section
    y -= 15*mm
    c.setFillColor(TEXT)
    c.setFont('Helvetica-Bold', 16)
    c.drawCentredString(w/2, y, 'Pipeline Architecture')

    # Pipeline boxes
    y -= 18*mm
    box_w = 35*mm
    box_h = 22*mm
    gap = 5*mm
    total_w = 4 * box_w + 3 * gap
    start_x = (w - total_w) / 2

    phases = [
        ('Phase 2', '文献分析', 'Claude CLI'),
        ('Phase 3', '方案设计', 'Claude CLI'),
        ('Phase 4', '代码实现', 'Claude CLI'),
        ('Phase 5', '测试验证', 'Python 直跑'),
    ]

    for i, (phase, name, method) in enumerate(phases):
        x = start_x + i * (box_w + gap)
        # Box
        c.setFillColor(SURFACE)
        c.setStrokeColor(BORDER)
        c.setLineWidth(0.8)
        c.roundRect(x, y, box_w, box_h, 3*mm, fill=1, stroke=1)
        # Phase label
        c.setFillColor(ACCENT)
        c.setFont('Helvetica-Bold', 8)
        c.drawCentredString(x + box_w/2, y + box_h - 7*mm, phase)
        # Name
        c.setFillColor(TEXT)
        c.setFont('Helvetica', 10)
        c.drawCentredString(x + box_w/2, y + box_h - 14*mm, name)
        # Method
        c.setFillColor(MUTED)
        c.setFont('Helvetica', 7)
        c.drawCentredString(x + box_w/2, y + 3*mm, method)

        # Arrow
        if i < 3:
            ax = x + box_w + 1*mm
            ay = y + box_h/2
            c.setStrokeColor(ACCENT)
            c.setLineWidth(1.2)
            c.line(ax, ay, ax + gap - 2*mm, ay)
            # Arrowhead
            c.setFillColor(ACCENT)
            c.beginPath()
            c.moveTo(ax + gap - 2*mm, ay)
            c.lineTo(ax + gap - 4*mm, ay + 1.5*mm)
            c.lineTo(ax + gap - 4*mm, ay - 1.5*mm)
            c.close()
            c.fill()

    # Key Results section
    y -= 30*mm
    c.setFillColor(TEXT)
    c.setFont('Helvetica-Bold', 16)
    c.drawCentredString(w/2, y, 'Key Results')

    # Results table
    y -= 12*mm
    col_x = [30*mm, 80*mm, 115*mm, 150*mm]
    headers = ['Method', 'Stage1 R²', 'Stage2 R²', 'Stage3 R²']

    # Header row bg
    c.setFillColor(SURFACE)
    c.rect(25*mm, y - 3*mm, w - 50*mm, 8*mm, fill=1, stroke=0)

    c.setFillColor(MUTED)
    c.setFont('Helvetica-Bold', 9)
    for i, header in enumerate(headers):
        c.drawString(col_x[i], y, header)

    # Data rows
    rows = [
        ('VNA (baseline)', '0.9034', '0.8408', '0.9031'),
        ('PolyRK', '0.9105', '0.8474', '0.9060'),
        ('AdvancedRK', '0.9162', '0.8526', '0.9129'),
    ]

    for j, (method, s1, s2, s3) in enumerate(rows):
        y -= 9*mm
        if j == 2:  # Best method highlighted
            c.setFillColor(HexColor('#1A2332'))
            c.rect(25*mm, y - 3*mm, w - 50*mm, 8*mm, fill=1, stroke=0)

        c.setFont('Helvetica-Bold' if j == 2 else 'Helvetica', 9)
        c.setFillColor(GREEN if j == 2 else TEXT)
        c.drawString(col_x[0], y, method)
        c.setFillColor(GREEN if j == 2 else ACCENT)
        c.setFont('Helvetica', 9)
        c.drawString(col_x[1], y, s1)
        c.drawString(col_x[2], y, s2)
        c.drawString(col_x[3], y, s3)

    # Innovation criteria
    y -= 20*mm
    c.setFillColor(TEXT)
    c.setFont('Helvetica-Bold', 14)
    c.drawCentredString(w/2, y, 'Innovation Criteria')

    y -= 10*mm
    criteria = [
        ('R²', '≥ best baseline + 0.01'),
        ('RMSE', '≤ best baseline'),
        ('|MB|', '≤ best baseline'),
    ]

    for metric, req in criteria:
        c.setFillColor(SURFACE)
        c.roundRect(35*mm, y - 3*mm, 50*mm, 7*mm, 2*mm, fill=1, stroke=0)
        c.setFillColor(ACCENT)
        c.setFont('Helvetica-Bold', 9)
        c.drawCentredString(60*mm, y, metric)
        c.setFillColor(MUTED)
        c.setFont('Helvetica', 9)
        c.drawString(92*mm, y, req)
        y -= 10*mm

    # Profiles section
    y -= 5*mm
    c.setFillColor(TEXT)
    c.setFont('Helvetica-Bold', 14)
    c.drawCentredString(w/2, y, 'Profiles')

    y -= 10*mm
    profiles = [
        ('full', '0-6', '完整流程'),
        ('skip-download', '0,2-6', '跳过下载'),
        ('design-verify', '3-5', '设计+验证'),
        ('verify-only', '5', '只跑验证'),
        ('code-iterate', '4-5', '编码迭代'),
    ]

    for name, phases_str, desc in profiles:
        c.setFillColor(SURFACE)
        c.roundRect(30*mm, y - 3*mm, 45*mm, 7*mm, 2*mm, fill=1, stroke=0)
        c.setFillColor(ORANGE)
        c.setFont('Helvetica-Bold', 8)
        c.drawCentredString(52.5*mm, y, name)
        c.setFillColor(MUTED)
        c.setFont('Helvetica', 8)
        c.drawString(80*mm, y, phases_str)
        c.setFillColor(TEXT)
        c.setFont('Helvetica', 8)
        c.drawString(105*mm, y, desc)
        y -= 9*mm

    # Bottom divider
    y -= 8*mm
    c.setStrokeColor(BORDER)
    c.setLineWidth(0.5)
    c.line(25*mm, y, w - 25*mm, y)

    # Footer
    y -= 10*mm
    c.setFillColor(MUTED)
    c.setFont('Helvetica', 9)
    c.drawCentredString(w/2, y, '湖南大学 · 机械与运载工程学院')

    y -= 7*mm
    c.setFont('Helvetica', 7)
    c.drawCentredString(w/2, y, 'Academic Research Use')

    # Bottom accent line
    c.setFillColor(ACCENT)
    c.rect(0, 0, w, 3*mm, fill=1, stroke=0)

    c.save()
    print(f"PDF created: {output_path}")


def create_with_pil(output_path):
    """Create PNG poster using Pillow."""
    # A4 at 150 DPI
    w, h = 1240, 1754
    img = Image.new('RGB', (w, h), '#0D1117')
    draw = ImageDraw.Draw(img)

    # Try to load a font
    try:
        font_bold = ImageFont.truetype("arial.ttf", 42)
        font_med = ImageFont.truetype("arial.ttf", 22)
        font_sm = ImageFont.truetype("arial.ttf", 16)
        font_xs = ImageFont.truetype("arial.ttf", 13)
    except:
        font_bold = ImageFont.load_default()
        font_med = font_bold
        font_sm = font_bold
        font_xs = font_bold

    # Top accent
    draw.rectangle([0, 0, w, 8], fill='#58A6FF')

    # Title
    y = 60
    draw.text((w//2, y), 'PM2.5 · CMAQ Data Fusion', fill='#E6EDF3', font=font_bold, anchor='mt')

    y += 55
    draw.text((w//2, y), '自动化 PM2.5 数据融合研究系统', fill='#8B949E', font=font_med, anchor='mt')

    y += 40
    draw.text((w//2, y), '文献分析  →  方案设计  →  代码实现  →  测试验证  →  论文生成', fill='#58A6FF', font=font_sm, anchor='mt')

    # Divider
    y += 35
    draw.line([(60, y), (w-60, y)], fill='#30363D', width=1)

    # Pipeline
    y += 30
    draw.text((w//2, y), 'Pipeline Architecture', fill='#E6EDF3', font=font_med, anchor='mt')

    y += 45
    box_w, box_h = 250, 70
    gap = 20
    total = 4 * box_w + 3 * gap
    sx = (w - total) // 2

    phases = [
        ('Phase 2', '文献分析', 'Claude CLI'),
        ('Phase 3', '方案设计', 'Claude CLI'),
        ('Phase 4', '代码实现', 'Claude CLI'),
        ('Phase 5', '测试验证', 'Python 直跑'),
    ]

    for i, (phase, name, method) in enumerate(phases):
        x = sx + i * (box_w + gap)
        draw.rounded_rectangle([x, y, x+box_w, y+box_h], radius=8, fill='#161B22', outline='#30363D')
        draw.text((x + box_w//2, y + 15), phase, fill='#58A6FF', font=font_xs, anchor='mt')
        draw.text((x + box_w//2, y + 35), name, fill='#E6EDF3', font=font_sm, anchor='mt')
        draw.text((x + box_w//2, y + 55), method, fill='#8B949E', font=font_xs, anchor='mt')

        if i < 3:
            ax = x + box_w + 5
            ay = y + box_h // 2
            draw.line([(ax, ay), (ax + gap - 10, ay)], fill='#58A6FF', width=2)
            # Arrowhead
            draw.polygon([(ax+gap-10, ay), (ax+gap-16, ay-5), (ax+gap-16, ay+5)], fill='#58A6FF')

    # Key Results
    y += 110
    draw.text((w//2, y), 'Key Results', fill='#E6EDF3', font=font_med, anchor='mt')

    y += 40
    cols = [80, 400, 600, 800]
    headers = ['Method', 'Stage1 R²', 'Stage2 R²', 'Stage3 R²']

    # Header bg
    draw.rectangle([60, y-5, w-60, y+25], fill='#161B22')
    for i, h_text in enumerate(headers):
        draw.text((cols[i], y), h_text, fill='#8B949E', font=font_sm)

    rows = [
        ('VNA (baseline)', '0.9034', '0.8408', '0.9031', '#E6EDF3'),
        ('PolyRK', '0.9105', '0.8474', '0.9060', '#3FB950'),
        ('AdvancedRK', '0.9162', '0.8526', '0.9129', '#3FB950'),
    ]

    for method, s1, s2, s3, color in rows:
        y += 32
        if method == 'AdvancedRK':
            draw.rectangle([60, y-5, w-60, y+25], fill='#1A2332')
        draw.text((cols[0], y), method, fill=color, font=font_sm)
        draw.text((cols[1], y), s1, fill='#58A6FF', font=font_sm)
        draw.text((cols[2], y), s2, fill='#58A6FF', font=font_sm)
        draw.text((cols[3], y), s3, fill='#58A6FF', font=font_sm)

    # Innovation Criteria
    y += 60
    draw.text((w//2, y), 'Innovation Criteria', fill='#E6EDF3', font=font_med, anchor='mt')

    y += 35
    criteria = [
        ('R²', '≥ best baseline + 0.01'),
        ('RMSE', '≤ best baseline'),
        ('|MB|', '≤ best baseline'),
    ]
    for metric, req in criteria:
        draw.rounded_rectangle([100, y-5, 400, y+25], radius=6, fill='#161B22')
        draw.text((250, y), metric, fill='#58A6FF', font=font_sm, anchor='mt')
        draw.text((430, y), req, fill='#8B949E', font=font_sm)
        y += 35

    # Profiles
    y += 20
    draw.text((w//2, y), 'Profiles', fill='#E6EDF3', font=font_med, anchor='mt')

    y += 35
    profiles = [
        ('full', '0-6', '完整流程'),
        ('skip-download', '0,2-6', '跳过下载'),
        ('design-verify', '3-5', '设计+验证'),
        ('verify-only', '5', '只跑验证'),
        ('code-iterate', '4-5', '编码迭代'),
    ]
    for name, ph, desc in profiles:
        draw.rounded_rectangle([80, y-5, 370, y+25], radius=6, fill='#161B22')
        draw.text((225, y), name, fill='#D29922', font=font_sm, anchor='mt')
        draw.text((400, y), ph, fill='#8B949E', font=font_sm)
        draw.text((520, y), desc, fill='#E6EDF3', font=font_sm)
        y += 32

    # Footer
    y += 30
    draw.line([(60, y), (w-60, y)], fill='#30363D', width=1)
    y += 20
    draw.text((w//2, y), '湖南大学 · 机械与运载工程学院', fill='#8B949E', font=font_sm, anchor='mt')
    y += 25
    draw.text((w//2, y), 'Academic Research Use', fill='#8B949E', font=font_xs, anchor='mt')

    # Bottom accent
    draw.rectangle([0, h-6, w, h], fill='#58A6FF')

    img.save(output_path, quality=95)
    print(f"PNG created: {output_path}")


if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))

    if HAS_REPORTLAB:
        create_with_reportlab(os.path.join(base, 'readme-poster.pdf'))
    elif HAS_PIL:
        create_with_pil(os.path.join(base, 'readme-poster.png'))
    else:
        print("Need reportlab or Pillow. Install with:")
        print("  pip install reportlab")
        print("  or")
        print("  pip install Pillow")
