from django.contrib import admin
from .models import Resume, ResumeFile
@admin.register(Resume)
class ResumeAdmin(admin.ModelAdmin):
    list_display  = ["full_name","user","name","template","version_number","last_score","updated_at"]
    search_fields = ["full_name","user__email","name"]
@admin.register(ResumeFile)
class ResumeFileAdmin(admin.ModelAdmin):
    list_display = ["resume","file_type","uploaded_at"]