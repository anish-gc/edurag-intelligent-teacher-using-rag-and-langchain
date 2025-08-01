
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static

from django.conf import settings
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include("ai_tutor.urls")),
    path('', include("monitoring.urls")),
    path('', include("knowledge_base.urls")),
]
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

# if settings.DEBUG:
#     urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
#     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
# else:
#     # For production, still add static URLs as fallback
#     urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
#     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)