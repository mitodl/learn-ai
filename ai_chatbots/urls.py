"""Standard urls for ai_chatbots app"""

from django.urls import include, re_path
from rest_framework.routers import SimpleRouter

from ai_chatbots import views

router = SimpleRouter()
router.register(
    r"chat_sessions",
    views.UserChatSessionsViewSet,
    basename="chat_sessions",
)
router.register(
    r"chat_sessions/(?P<thread_id>[A-Za-z0-9_\-]+)/messages",
    views.ChatMessageViewSet,
    basename="chat_session_messages",
)
router.register(r"llm_models", views.LLMModelViewSet, basename="llm_models")

app_name = "ai"
urlpatterns = [
    re_path(r"^api/v0/", include((router.urls, "v0"))),
]
