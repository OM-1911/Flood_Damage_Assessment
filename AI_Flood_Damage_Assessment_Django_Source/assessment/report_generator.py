import io
import os
from datetime import datetime
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY


def _get_severity_colors(severity):
    sev = (severity or '').lower()
    if 'low' in sev:
        return colors.HexColor('#16a34a'), colors.HexColor('#dcfce7')
    if 'mod' in sev:
        return colors.HexColor('#d97706'), colors.HexColor('#fef3c7')
    if 'high' in sev:
        return colors.HexColor('#dc2626'), colors.HexColor('#fee2e2')
    return colors.HexColor('#991b1b'), colors.HexColor('#fecaca')


def _get_fitted_image(img_path, max_w, max_h, placeholder_style):
    if not img_path or not os.path.exists(img_path):
        return Paragraph('<b>Image Unavailable</b>', placeholder_style)
    try:
        with PILImage.open(img_path) as im:
            orig_w, orig_h = im.size
        if orig_w <= 0 or orig_h <= 0:
            return Paragraph('<b>Invalid Image</b>', placeholder_style)
        ratio = min(max_w / orig_w, max_h / orig_h)
        return RLImage(img_path, width=orig_w * ratio, height=orig_h * ratio)
    except Exception:
        return Paragraph('<b>Error Loading Image</b>', placeholder_style)


def generate_assessment_pdf(assessment_obj):
    """
    Generates a professional, comprehensive PDF report containing all three images
    (original, segmentation mask, overlay) and the full assessment summary.
    Returns bytes of the generated PDF.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=32,
    )
    content_width = A4[0] - 72  # 595.27 - 72 = 523.27 pt

    styles = getSampleStyleSheet()

    # Custom styles
    brand_style = ParagraphStyle(
        'BrandHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a')
    )
    subhead_style = ParagraphStyle(
        'BrandSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#475569')
    )
    badge_style = ParagraphStyle(
        'MetaBadge',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        alignment=TA_RIGHT,
        textColor=colors.HexColor('#2563eb')
    )
    metric_label_style = ParagraphStyle(
        'MetricLabel',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#64748b')
    )
    metric_val_style = ParagraphStyle(
        'MetricVal',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#0f172a')
    )
    card_title_style = ParagraphStyle(
        'CardTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1e293b')
    )
    card_sub_style = ParagraphStyle(
        'CardSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#64748b')
    )
    section_heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0f172a')
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1e293b')
    )
    table_cell_norm = ParagraphStyle(
        'TableCellNorm',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#334155')
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7,
        leading=9.5,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor('#64748b')
    )

    story = []

    # 1. Header Banner
    date_str = (
        assessment_obj.created_at.strftime("%B %d, %Y - %I:%M %p")
        if getattr(assessment_obj, 'created_at', None)
        else datetime.now().strftime("%B %d, %Y - %I:%M %p")
    )
    obj_pk = getattr(assessment_obj, 'pk', 0) or 0
    header_data = [
        [
            Paragraph('<b>AI FLOOD DAMAGE ASSESSMENT REPORT</b>', brand_style),
            Paragraph(f'<b>ASSESSMENT #{obj_pk}</b><br/>Status: <b>Completed</b>', badge_style)
        ],
        [
            Paragraph(f'Deep Learning Aerial & Ground Imagery Analysis | Date: {date_str}', subhead_style),
            Paragraph(f'Report ID: FLD-{obj_pk:05d}', ParagraphStyle('IDSub', parent=subhead_style, alignment=TA_RIGHT))
        ]
    ]
    header_table = Table(header_data, colWidths=[content_width * 0.72, content_width * 0.28])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 2),
        ('TOPPADDING', (0, 1), (-1, 1), 0),
        ('BOTTOMPADDING', (0, 1), (-1, 1), 6),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563eb'), spaceBefore=2, spaceAfter=10))

    # 2. Executive Metric Cards
    sev_text_color, sev_bg_color = _get_severity_colors(assessment_obj.severity)
    sev_val_style = ParagraphStyle(
        'SevVal',
        parent=metric_val_style,
        textColor=sev_text_color
    )

    col_w = content_width / 3.0
    metrics_data = [
        [
            Paragraph('ESTIMATED FLOOD COVERAGE', metric_label_style),
            Paragraph('SEVERITY RATING', metric_label_style),
            Paragraph('SURFACE STATUS', metric_label_style),
        ],
        [
            Paragraph(f'{assessment_obj.flood_percentage:.2f}%', metric_val_style),
            Paragraph(f'{assessment_obj.severity}', sev_val_style),
            Paragraph('Flooded Area Detected' if assessment_obj.flood_percentage > 0.5 else 'Dry Surface / Normal', metric_val_style),
        ]
    ]
    metrics_table = Table(metrics_data, colWidths=[col_w] * 3)
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (1, 0), (1, 1), sev_bg_color),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 2),
        ('TOPPADDING', (0, 1), (-1, 1), 2),
        ('BOTTOMPADDING', (0, 1), (-1, 1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 12))

    # 3. All Three Images Gallery
    story.append(Paragraph('Imagery Evidence & Segmentation Analysis', section_heading_style))
    story.append(Spacer(1, 6))

    img_box_w = content_width / 3.0
    max_w = img_box_w - 12
    max_h = 120

    def _get_path_safe(field):
        try:
            return field.path if field and hasattr(field, 'path') else None
        except Exception:
            return None

    img_orig_path = _get_path_safe(getattr(assessment_obj, 'uploaded_image', None))
    img_mask_path = _get_path_safe(getattr(assessment_obj, 'mask_image', None))
    img_overlay_path = _get_path_safe(getattr(assessment_obj, 'result_image', None))

    img1 = _get_fitted_image(img_orig_path, max_w, max_h, card_sub_style)
    img2 = _get_fitted_image(img_mask_path, max_w, max_h, card_sub_style)
    img3 = _get_fitted_image(img_overlay_path, max_w, max_h, card_sub_style)

    images_data = [
        [
            Paragraph('<b>1. Original Photo</b>', card_title_style),
            Paragraph('<b>2. Segmentation Mask</b>', card_title_style),
            Paragraph('<b>3. Flood Overlay</b>', card_title_style),
        ],
        [
            Paragraph('Input Image (RGB Scene)', card_sub_style),
            Paragraph('AI Pixel Map (Cyan=Water, Dark=Dry)', card_sub_style),
            Paragraph('Visual Detection Composite', card_sub_style),
        ],
        [img1, img2, img3]
    ]

    images_table = Table(images_data, colWidths=[img_box_w] * 3)
    images_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 1),
        ('TOPPADDING', (0, 1), (-1, 1), 0),
        ('BOTTOMPADDING', (0, 1), (-1, 1), 4),
        ('TOPPADDING', (0, 2), (-1, 2), 4),
        ('BOTTOMPADDING', (0, 2), (-1, 2), 6),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 2), (-1, 2), 'MIDDLE'),
    ]))
    story.append(images_table)
    story.append(Spacer(1, 12))

    # 4. Detailed Technical Assessment & Safety Recommendations
    story.append(Paragraph('Detailed Findings & Impact Assessment', section_heading_style))
    story.append(Spacer(1, 5))

    # Severity advice
    sev = (assessment_obj.severity or '').lower()
    if 'low' in sev:
        safety_text = "Surface water coverage is minimal or negligible. Continue monitoring local weather updates and official civil defense advisories. Inspect drainage outlets and keep emergency kits accessible."
    elif 'mod' in sev:
        safety_text = "Moderate localized water accumulation observed. Prepare essential medicines, drinking water, and important documents. Avoid driving through flooded streets or submerged underpasses."
    elif 'high' in sev:
        safety_text = "Significant inundation detected across scene. Follow official local authority evacuation orders. Stay on elevated ground; never attempt to wade or drive across submerged roadways."
    else:
        safety_text = "Severe widespread flooding detected. Critical hazard to structures and transport. Comply immediately with emergency evacuation notices and contact local disaster relief services if trapped."

    details_data = [
        [
            Paragraph('Estimated Water Coverage', table_cell_bold),
            Paragraph(f'<b>{assessment_obj.flood_percentage:.2f}%</b> of total analyzed image surface area', table_cell_norm),
        ],
        [
            Paragraph('Severity Classification', table_cell_bold),
            Paragraph(f'<b>{assessment_obj.severity} Impact</b> - evaluated from semantic pixel counts', table_cell_norm),
        ],
        [
            Paragraph('Recommended Safety Actions', table_cell_bold),
            Paragraph(safety_text, table_cell_norm),
        ],
    ]

    details_table = Table(details_data, colWidths=[content_width * 0.32, content_width * 0.68])
    details_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f1f5f9')),
        ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#ffffff')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4.5),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(details_table)
    story.append(Spacer(1, 10))

    # 5. Disclaimer & System Note
    disclaimer_text = (
        "<b>DISCLAIMER:</b> This report is generated automatically by an artificial intelligence computer vision system "
        "trained on semantic water segmentation. These findings represent visual approximations from uploaded imagery "
        "and do not substitute for on-ground geotechnical evaluations, hydrological depth surveys, or official "
        "instructions issued by civil defense, municipal authorities, or emergency management services."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cbd5e1'), spaceBefore=2, spaceAfter=4))

    footer_data = [
        [
            Paragraph(f'FloodAI System | Report FLD-{obj_pk:05d}', ParagraphStyle('FootL', parent=subhead_style, fontSize=7.5)),
            Paragraph(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ParagraphStyle('FootR', parent=subhead_style, fontSize=7.5, alignment=TA_RIGHT))
        ]
    ]
    footer_table = Table(footer_data, colWidths=[content_width * 0.5, content_width * 0.5])
    footer_table.setStyle(TableStyle([
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(footer_table)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
