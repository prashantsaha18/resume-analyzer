import logging
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .models import Conversation, Message
from .serializers import ConversationSerializer, ConversationListSerializer, MessageSerializer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert career coach with 15+ years of experience.
Help users with resume writing, ATS optimization, job searching, interview prep,
salary negotiation, and career development. Be concise and actionable."""

@api_view(["GET","POST"])
@permission_classes([IsAuthenticated])
def conversations(request):
    if request.method == "GET":
        convs = Conversation.objects.filter(user=request.user)
        return Response(ConversationListSerializer(convs, many=True).data)
    conv = Conversation.objects.create(user=request.user, title=request.data.get("title","New Conversation"))
    return Response(ConversationSerializer(conv).data, status=201)

@api_view(["GET","DELETE"])
@permission_classes([IsAuthenticated])
def conversation_detail(request, pk):
    try:
        conv = Conversation.objects.get(pk=pk, user=request.user)
    except Conversation.DoesNotExist:
        return Response({"error": "Not found"}, status=404)
    if request.method == "DELETE":
        conv.delete()
        return Response(status=204)
    return Response(ConversationSerializer(conv).data)

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def send_message(request, pk):
    try:
        conv = Conversation.objects.get(pk=pk, user=request.user)
    except Conversation.DoesNotExist:
        return Response({"error": "Not found"}, status=404)
    user_text = request.data.get("message","").strip()
    if not user_text:
        return Response({"error": "message required"}, status=400)
    Message.objects.create(conversation=conv, role="user", content=user_text)
    history = [
        {"role": m.role if m.role == "user" else "assistant", "parts": [m.content]}
        for m in conv.messages.order_by("-created_at")[1:9]
    ]
    history.reverse()
    try:
        from ml_engine.router import chat_reply
        ai_text = chat_reply(user_text, history)
    except Exception as e:
        logger.error(f"chat error: {e}")
        ai_text = "I encountered an error. Please try again."
    ai_msg = Message.objects.create(conversation=conv, role="assistant", content=ai_text)
    if conv.messages.count() <= 2:
        conv.title = user_text[:60]
        conv.save(update_fields=["title","updated_at"])
    return Response({
        "user_message": {"role": "user",      "content": user_text},
        "ai_message":   MessageSerializer(ai_msg).data,
    })

@api_view(["POST"])
@permission_classes([IsAuthenticated])
def quick_chat(request):
    message = request.data.get("message","").strip()
    if not message:
        return Response({"error": "message required"}, status=400)
    try:
        from ml_engine.router import chat_reply
        return Response({"reply": chat_reply(message, [])})
    except Exception as e:
        return Response({"error": str(e)}, status=500)