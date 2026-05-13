from rest_framework import serializers
from .models import User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model  = User
        fields = ["id","firebase_uid","email","display_name","photo_url","plan","credits_used","date_joined"]
        read_only_fields = ["id","firebase_uid","date_joined","credits_used"]

class UserUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model  = User
        fields = ["display_name","photo_url"]