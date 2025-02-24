"""Standard urls for ai_chatbots app"""

from django.urls import include, path, re_path
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

v0_urls = [
    *router.urls,
    path(
        r"tutor_problem/",
        views.TutorProblemView.as_view(),
        name="tutor_problem",
    ),
]

app_name = "ai"
urlpatterns = [
    re_path(r"^api/v0/", include((v0_urls, "v0"))),
]
