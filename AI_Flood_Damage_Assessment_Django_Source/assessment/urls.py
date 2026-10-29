from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('assessment/', views.assessment, name='assessment'),
    path('assessment/<int:pk>/', views.assessment_detail, name='assessment_detail'),
    path('assessment/<int:pk>/download/', views.download_assessment, name='download_assessment'),
    path('history/', views.history, name='history'),
]
