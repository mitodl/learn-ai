from django.urls import re_path

from ai_agents import consumers

websocket_urlpatterns = [
    # websocket URLs go here
    re_path(
        r"ws/recommendation_agent/",
        consumers.RecommendationAgentConsumer.as_asgi(),
        name="recommendation_agent",
    ),
]
