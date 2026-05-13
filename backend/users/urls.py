from django.urls import path
from . import views
urlpatterns = [
    path("profile/",   views.ProfileView.as_view(), name="profile"),
    path("sync/",      views.sync_user,             name="sync-user"),
    path("dashboard/", views.dashboard_stats,        name="dashboard-stats"),
]