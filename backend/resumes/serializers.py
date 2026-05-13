from rest_framework import serializers
from .models import Resume, ResumeFile

class ResumeFileSerializer(serializers.ModelSerializer):
    class Meta:
        model  = ResumeFile
        fields = ["id","file","file_type","extracted_text","uploaded_at"]
        read_only_fields = ["id","extracted_text","uploaded_at"]

class ResumeSerializer(serializers.ModelSerializer):
    files = ResumeFileSerializer(many=True, read_only=True)
    class Meta:
        model  = Resume
        fields = ["id","name","template","version_number","is_primary","full_name","email","phone",
                  "location","linkedin_url","github_url","portfolio_url","summary","education",
                  "experience","projects","certifications","technical_skills","soft_skills",
                  "languages","awards","created_at","updated_at","last_score","files"]
        read_only_fields = ["id","created_at","updated_at","version_number"]
    def create(self, validated_data):
        user = self.context["request"].user
        count = Resume.objects.filter(user=user).count()
        validated_data["user"] = user
        validated_data["version_number"] = count + 1
        return super().create(validated_data)

class ResumeListSerializer(serializers.ModelSerializer):
    class Meta:
        model  = Resume
        fields = ["id","name","template","version_number","is_primary","full_name","last_score","created_at","updated_at"]