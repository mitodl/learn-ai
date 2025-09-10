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
router.register(r"llm_models", views.LLMModelViewSet, basename="llm_models")
router.register(r"prompts", views.SystemPromptViewSet, basename="prompts")

app_name = "ai"
v0_urls = [
    *router.urls,
    path(
        r"get_transcript_edx_module_id/",
        views.GetTranscriptBlockId.as_view(),
        name="get_transcript_edx_module_id",
    ),
    path(
        r"problem_set_list/",
        views.ProblemSetList.as_view(),
        name="problem_set_list",
    ),
    path(
        r"ab_test_choice/",
        views.ABTestChoiceView.as_view(),
        name="ab_test_choice",
    ),
]

urlpatterns = [
    re_path(r"^api/v0/", include((v0_urls, "v0"))),
]
