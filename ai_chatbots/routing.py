from django.urls import re_path

from ai_chatbots import consumers

websocket_patterns = [
    # websocket URLs go here
    re_path(
        r"ws/recommendation_agent/",
        consumers.RecommendationBotWSConsumer.as_asgi(),
        name="recommendation_agent_ws",
    )
]

http_patterns = [
    re_path(
        r"http/recommendation_agent/",
        consumers.RecommendationBotHttpConsumer.as_asgi(),
        name="recommendation_agent_sse",
    ),
        re_path(
        r"http/tutor_agent/",
        consumers.TutorBotHttpConsumer.as_asgi(),
        name="tutor_agent_sse",
    ),
]
