from django.core.asgi import get_asgi_application
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
    # This gets two routes - user_meta doesn't require auth (in the APISIX settings)
    # and login does.
    re_path(
        r"^http/(user_meta|login)/$",
        UserMetaHttpConsumer.as_asgi(),
        name="user_meta",
    ),
    re_path(
        r"",
        get_asgi_application(),
    ),
]
