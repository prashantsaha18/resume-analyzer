from django.contrib import admin
from .models import Analysis, JobMatch
@admin.register(Analysis)
class AnalysisAdmin(admin.ModelAdmin):
    list_display  = ["resume","ats_score","content_score","created_at"]
    search_fields = ["resume__full_name","resume__user__email"]
    readonly_fields = ["created_at"]
@admin.register(JobMatch)
class JobMatchAdmin(admin.ModelAdmin):
    list_display  = ["resume","job_title","match_score","created_at"]
    readonly_fields = ["created_at"]