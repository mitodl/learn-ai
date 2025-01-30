from django.urls import re_path

from ai_chatbots import consumers

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
]
