import os
from pathlib import Path

import cv2
from django.conf import settings
from django.shortcuts import get_object_or_404, redirect, render

from .ml_model import predict_flood
from .models import FloodAssessment


ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024


def home(request):
    total = FloodAssessment.objects.count()
    return render(request, 'assessment/home.html', {'total_assessments': total})


def assessment(request):
    if request.method != 'POST':
        return render(request, 'assessment/assessment.html')

    uploaded_file = request.FILES.get('image')
    if not uploaded_file:
        return render(request, 'assessment/assessment.html', {'error': 'Please select an image.'})

    extension = Path(uploaded_file.name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return render(request, 'assessment/assessment.html', {'error': 'Please upload JPG, JPEG, PNG, or WEBP.'})

    if uploaded_file.size > MAX_UPLOAD_SIZE:
        return render(request, 'assessment/assessment.html', {'error': 'Maximum upload size is 10 MB.'})

    obj = FloodAssessment.objects.create(uploaded_image=uploaded_file)

    try:
        mask, visual_mask, overlay, flood_percentage, severity, accuracy = predict_flood(obj.uploaded_image.path)
    except Exception as exc:
        obj.delete()
        return render(request, 'assessment/assessment.html', {'error': str(exc)})

    result_dir = Path(settings.MEDIA_ROOT) / 'results'
    mask_dir = result_dir / 'masks'
    overlay_dir = result_dir / 'overlays'
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    mask_name = f'mask_{obj.pk}.png'
    overlay_name = f'overlay_{obj.pk}.jpg'
    mask_path = mask_dir / mask_name
    overlay_path = overlay_dir / overlay_name

    # Save visual segmentation mask robustly via binary stream
    success_mask, buf_mask = cv2.imencode('.png', visual_mask, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if success_mask:
        with open(mask_path, 'wb') as f:
            f.write(buf_mask.tobytes())
    else:
        cv2.imwrite(str(mask_path), visual_mask)

    # Save overlay robustly via binary stream
    success_overlay, buf_overlay = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if success_overlay:
        with open(overlay_path, 'wb') as f:
            f.write(buf_overlay.tobytes())
    else:
        cv2.imwrite(str(overlay_path), overlay)

    obj.mask_image.name = f'results/masks/{mask_name}'
    obj.result_image.name = f'results/overlays/{overlay_name}'
    obj.flood_percentage = flood_percentage
    obj.accuracy = accuracy
    obj.severity = severity
    obj.save(update_fields=['mask_image', 'result_image', 'flood_percentage', 'accuracy', 'severity'])

    return render(
        request,
        'assessment/result.html',
        {'assessment': obj}
    )


from django.http import HttpResponse
from .report_generator import generate_assessment_pdf


def assessment_detail(request, pk):
    obj = get_object_or_404(FloodAssessment, pk=pk)
    return render(request, 'assessment/result.html', {'assessment': obj})


def download_assessment(request, pk):
    obj = get_object_or_404(FloodAssessment, pk=pk)
    pdf_bytes = generate_assessment_pdf(obj)
    response = HttpResponse(pdf_bytes, content_type='application/pdf')
    filename = f"flood_assessment_report_{obj.pk}.pdf"
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response



def history(request):
    assessments = FloodAssessment.objects.all()[:20]
    return render(request, 'assessment/history.html', {'assessments': assessments})
