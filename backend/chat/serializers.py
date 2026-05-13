from rest_framework import serializers
from .models import Conversation, Message

class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model  = Message
        fields = ["id","role","content","created_at"]
        read_only_fields = ["id","created_at"]

class ConversationSerializer(serializers.ModelSerializer):
    messages      = MessageSerializer(many=True, read_only=True)
    message_count = serializers.SerializerMethodField()
    class Meta:
        model  = Conversation
        fields = ["id","title","created_at","updated_at","messages","message_count"]
        read_only_fields = ["id","created_at","updated_at"]
    def get_message_count(self, obj):
        return obj.messages.count()

class ConversationListSerializer(serializers.ModelSerializer):
    message_count = serializers.SerializerMethodField()
    last_message  = serializers.SerializerMethodField()
    class Meta:
        model  = Conversation
        fields = ["id","title","created_at","updated_at","message_count","last_message"]
    def get_message_count(self, obj):
        return obj.messages.count()
    def get_last_message(self, obj):
        m = obj.messages.last()
        return m.content[:80] if m else ""