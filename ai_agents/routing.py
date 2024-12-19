from django.urls import re_path

from ai_agents import consumers

websocket_patterns = [
    # websocket URLs go here
    re_path(
        r"ws/recommendation_agent/",
        consumers.RecommendationAgentWSConsumer.as_asgi(),
        name="recommendation_agent_ws",
    ),
]

http_patterns = [
    re_path(
        r"sse/recommendation_agent/",
        consumers.RecommendationAgentSSEConsumer.as_asgi(),
        name="recommendation_agent_sse",
    ),
]
