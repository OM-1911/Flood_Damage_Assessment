from django.db import models


class FloodAssessment(models.Model):
    uploaded_image = models.ImageField(upload_to='uploads/')
    mask_image = models.ImageField(upload_to='results/masks/', blank=True, null=True)
    result_image = models.ImageField(upload_to='results/overlays/', blank=True, null=True)
    flood_percentage = models.FloatField(default=0.0)
    accuracy = models.FloatField(default=0.0)
    severity = models.CharField(max_length=30, default='Unknown')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'Assessment #{self.pk} - {self.severity}'
