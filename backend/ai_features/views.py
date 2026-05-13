import logging
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

logger = logging.getLogger(__name__)

def _get(request, key):
    return request.data.get(key, "").strip()

def _run(fn, *args):
    try:
        return Response(fn(*args))
    except RuntimeError as e:
        return Response({"error": str(e), "hint": "Run: python manage.py train_models"}, status=503)
    except Exception as e:
        logger.error(f"ai_features error: {e}", exc_info=True)
        return Response({"error": str(e)}, status=500)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def improve_resume(request):
    text = _get(request, "resume_text")
    role = _get(request, "target_role")
    if not text: return Response({"error": "resume_text required"}, status=400)
    from ml_engine.router import improve_resume as fn
    try:
        return Response({"improved_resume": fn(text, role)})
    except RuntimeError as e:
        return Response({"error": str(e)}, status=503)
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def enhance_bullet(request):
    bullet = _get(request, "bullet")
    ctx    = _get(request, "context")
    if not bullet: return Response({"error": "bullet required"}, status=400)
    from ml_engine.router import enhance_bullet as fn
    return _run(fn, bullet, ctx)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def detect_weaknesses(request):
    text = _get(request, "resume_text")
    if not text: return Response({"error": "resume_text required"}, status=400)
    from ml_engine.router import detect_weaknesses_ai as fn
    return _run(fn, text)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def check_grammar(request):
    text = _get(request, "resume_text")
    if not text: return Response({"error": "resume_text required"}, status=400)
    from ml_engine.router import check_grammar_ai as fn
    return _run(fn, text)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def generate_cover_letter(request):
    text    = _get(request, "resume_text")
    company = _get(request, "company")
    role    = _get(request, "role")
    tone    = _get(request, "tone") or "professional"
    jd      = _get(request, "job_description")
    if not text or not company or not role:
        return Response({"error": "resume_text, company, role required"}, status=400)
    from ml_engine.router import generate_cover_letter as fn
    try:
        return Response({"cover_letter": fn(text, company, role, tone, jd)})
    except RuntimeError as e:
        return Response({"error": str(e)}, status=503)
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def recommend_job_roles(request):
    text = _get(request, "resume_text")
    if not text: return Response({"error": "resume_text required"}, status=400)
    from ml_engine.router import recommend_roles as fn
    return _run(fn, text)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def job_apply_assistant(request):
    text  = _get(request, "resume_text")
    title = _get(request, "job_title")
    jd    = _get(request, "job_description")
    if not text or not jd:
        return Response({"error": "resume_text and job_description required"}, status=400)
    from ml_engine.router import job_apply_analysis as fn
    return _run(fn, text, title, jd)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def detect_fake_resume(request):
    text = _get(request, "resume_text")
    if not text: return Response({"error": "resume_text required"}, status=400)
    from ml_engine.router import detect_fake as fn
    return _run(fn, text)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def generate_portfolio(request):
    text  = _get(request, "resume_text")
    theme = _get(request, "theme") or "dark"
    if not text: return Response({"error": "resume_text required"}, status=400)
    from ml_engine.router import generate_portfolio_html as fn
    try:
        return Response({"portfolio_html": fn(text, theme)})
    except RuntimeError as e:
        return Response({"error": str(e)}, status=503)
    except Exception as e:
        return Response({"error": str(e)}, status=500)