import logging
from rest_framework import status, viewsets
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
from .models import Resume, ResumeFile
from .serializers import ResumeSerializer, ResumeListSerializer

logger = logging.getLogger(__name__)

class ResumeViewSet(viewsets.ModelViewSet):
    permission_classes = [IsAuthenticated]
    parser_classes     = [JSONParser, MultiPartParser, FormParser]

    def get_queryset(self):
        return Resume.objects.filter(user=self.request.user)

    def get_serializer_class(self):
        return ResumeListSerializer if self.action == "list" else ResumeSerializer

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=True, methods=["post"], url_path="duplicate")
    def duplicate(self, request, pk=None):
        original = self.get_object()
        original.pk = None
        original.name = f"{original.name} (Copy)"
        original.is_primary = False
        original.version_number = Resume.objects.filter(user=request.user).count() + 1
        original.save()
        return Response(ResumeSerializer(original, context={"request": request}).data, status=201)

    @action(detail=True, methods=["post"], url_path="set-primary")
    def set_primary(self, request, pk=None):
        Resume.objects.filter(user=request.user).update(is_primary=False)
        r = self.get_object(); r.is_primary = True; r.save()
        return Response({"message": "Primary updated"})

    @action(detail=False, methods=["get"], url_path="primary")
    def primary(self, request):
        r = Resume.objects.filter(user=request.user, is_primary=True).first() or \
            Resume.objects.filter(user=request.user).first()
        if not r:
            return Response({"message": "No resume found"}, status=404)
        return Response(ResumeSerializer(r, context={"request": request}).data)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def upload_pdf(request):
    if "file" not in request.FILES:
        return Response({"error": "No file provided"}, status=400)
    f = request.FILES["file"]
    if not f.name.lower().endswith(".pdf"):
        return Response({"error": "Only PDF files supported"}, status=400)
    try:
        import fitz
        doc  = fitz.open(stream=f.read(), filetype="pdf")
        text = "\n".join(page.get_text("text") for page in doc).strip()
        doc.close()
        if not text:
            return Response({"error": "Could not extract text from PDF"}, status=400)
        return Response({"text": text, "pages": doc.page_count if hasattr(doc,"page_count") else 1,
                         "word_count": len(text.split())})
    except Exception as e:
        logger.error(f"PDF error: {e}")
        return Response({"error": str(e)}, status=500)