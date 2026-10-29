from django.contrib import admin
from .models import FloodAssessment


@admin.register(FloodAssessment)
class FloodAssessmentAdmin(admin.ModelAdmin):
    list_display = ('id', 'flood_percentage', 'severity', 'created_at')
    list_filter = ('severity', 'created_at')
    readonly_fields = ('created_at',)
