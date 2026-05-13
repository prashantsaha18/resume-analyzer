from django.urls import path
from . import views
urlpatterns = [
    path("conversations/",              views.conversations,       name="conversations"),
    path("conversations/<int:pk>/",     views.conversation_detail, name="conversation-detail"),
    path("conversations/<int:pk>/message/", views.send_message,   name="send-message"),
    path("quick/",                      views.quick_chat,          name="quick-chat"),
]