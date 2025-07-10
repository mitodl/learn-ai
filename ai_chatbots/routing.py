"""Routing for the ai_chatbots app."""

from django.urls import re_path

from ai_chatbots import consumers
from users.consumers import UserMetaHttpConsumer

http_patterns = [
    re_path(
        r"http/recommendation_agent/",
        consumers.RecommendationBotHttpConsumer.as_asgi(),
        name="recommendation_agent_sse",
    ),
    re_path(
        r"http/syllabus_agent/",
        consumers.SyllabusBotHttpConsumer.as_asgi(),
        name="syllabus_agent_sse",
    ),
    re_path(
        r"http/canvas_syllabus_agent/",
        consumers.CanvasSyllabusBotHttpConsumer.as_asgi(),
        name="canvas_syllabus_agent_sse",
    ),
    re_path(
        r"http/video_gpt_agent/",
        consumers.VideoGPTBotHttpConsumer.as_asgi(),
        name="video_gpt_agent_sse",
    ),
    # This gets two routes - user_meta doesn't require auth (in the APISIX settings)
    # and login does.
    re_path(
        r"^http/(user_meta|login)/$",
        UserMetaHttpConsumer.as_asgi(),
        name="user_meta",
    ),
    re_path(
        r"http/tutor_agent/",
        consumers.TutorBotHttpConsumer.as_asgi(),
        name="tutor_agent_sse",
    ),
]
