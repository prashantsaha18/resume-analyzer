import logging
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from resumes.models import Resume
from .models import Analysis
from .serializers import AnalysisSerializer

logger = logging.getLogger(__name__)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def analyze_resume(request):
    resume_text     = request.data.get("resume_text", "").strip()
    resume_id       = request.data.get("resume_id")
    job_description = request.data.get("job_description", "")
    if not resume_text:
        return Response({"error": "resume_text is required"}, status=400)
    try:
        from ml_engine.router import analyze_resume as ai_analyze
        data = ai_analyze(resume_text, job_description)
        if resume_id:
            try:
                resume = Resume.objects.get(id=resume_id, user=request.user)
                obj = Analysis.objects.create(
                    resume=resume, resume_text=resume_text, job_description=job_description,
                    ats_score=data.get("ats_score",0), formatting_score=data.get("formatting_score",0),
                    content_score=data.get("content_score",0), skills_score=data.get("skills_score",0),
                    keywords_score=data.get("keywords_score",0), readability_score=data.get("readability_score",0),
                    found_keywords=data.get("found_keywords",[]), missing_keywords=data.get("missing_keywords",[]),
                    strengths=data.get("strengths",[]), weaknesses=data.get("weaknesses",[]),
                    suggestions=data.get("suggestions",[]), grammar_issues=data.get("grammar_issues",[]),
                    action_verbs=data.get("action_verbs",{}), overall_feedback=data.get("overall_feedback",""),
                )
                resume.last_score = data.get("ats_score", 0)
                resume.save(update_fields=["last_score"])
                data["analysis_id"] = obj.id
            except Resume.DoesNotExist:
                pass
        return Response(data)
    except RuntimeError as e:
        return Response({"error": str(e), "hint": "Run: python manage.py train_models"}, status=503)
    except Exception as e:
        logger.error(f"analyze_resume: {e}", exc_info=True)
        return Response({"error": str(e)}, status=500)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def ats_keyword_match(request):
    resume_text = request.data.get("resume_text","").strip()
    job_desc    = request.data.get("job_description","").strip()
    if not resume_text or not job_desc:
        return Response({"error": "Both fields required"}, status=400)
    try:
        from ml_engine.router import ats_match
        return Response(ats_match(resume_text, job_desc))
    except RuntimeError as e:
        return Response({"error": str(e)}, status=503)
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def skill_gap_analysis(request):
    resume_text = request.data.get("resume_text","").strip()
    job_desc    = request.data.get("job_description","").strip()
    if not resume_text or not job_desc:
        return Response({"error": "Both fields required"}, status=400)
    try:
        from ml_engine.router import skill_gap
        return Response(skill_gap(resume_text, job_desc))
    except RuntimeError as e:
        return Response({"error": str(e)}, status=503)
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def analysis_history(request):
    resume_id = request.query_params.get("resume_id")
    qs = Analysis.objects.filter(resume__user=request.user)
    if resume_id:
        qs = qs.filter(resume_id=resume_id)
    return Response(AnalysisSerializer(qs[:20], many=True).data)