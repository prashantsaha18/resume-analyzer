from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import UserSerializer, UserUpdateSerializer

class ProfileView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request):
        return Response(UserSerializer(request.user).data)
    def patch(self, request):
        s = UserUpdateSerializer(request.user, data=request.data, partial=True)
        if s.is_valid():
            s.save()
            return Response(UserSerializer(request.user).data)
        return Response(s.errors, status=400)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def sync_user(request):
    return Response({"message": "synced", "user": UserSerializer(request.user).data})

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def dashboard_stats(request):
    from resumes.models import Resume
    from analysis.models import Analysis
    user = request.user
    latest = Analysis.objects.filter(resume__user=user).order_by("-created_at").first()
    resume_count = Resume.objects.filter(user=user).count()
    return Response({
        "resume_count":    resume_count,
        "version_count":   resume_count,
        "analysis_count":  Analysis.objects.filter(resume__user=user).count(),
        "latest_ats_score": latest.ats_score if latest else 0,
        "plan":            user.plan,
        "credits_used":    user.credits_used,
    })