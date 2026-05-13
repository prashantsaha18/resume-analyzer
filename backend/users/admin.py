from django.contrib import admin
from .models import User
@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display    = ["email","display_name","plan","date_joined","is_active"]
    list_filter     = ["plan","is_active"]
    search_fields   = ["email","display_name","firebase_uid"]
    readonly_fields = ["firebase_uid","date_joined","last_login","credits_used"]