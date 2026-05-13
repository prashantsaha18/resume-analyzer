from rest_framework import serializers
from .models import Analysis, JobMatch
class AnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model  = Analysis
        fields = "__all__"
        read_only_fields = ["id","created_at"]
class JobMatchSerializer(serializers.ModelSerializer):
    class Meta:
        model  = JobMatch
        fields = "__all__"
        read_only_fields = ["id","created_at"]