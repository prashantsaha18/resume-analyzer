from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
router = DefaultRouter()
router.register(r"", views.ResumeViewSet, basename="resume")
urlpatterns = [
    path("upload-pdf/", views.upload_pdf, name="upload-pdf"),
    path("", include(router.urls)),
]