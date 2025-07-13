from django.urls import path
from .views import GetMetricsView


urlpatterns = [
    path("api/metrics/", GetMetricsView.as_view(), name='get_metrics')
]