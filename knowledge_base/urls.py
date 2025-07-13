from django.urls import path

from knowledge_base import views


urlpatterns = [
    path('api/upload-content/', views.ContentUploadView.as_view(), name='upload_content')
]