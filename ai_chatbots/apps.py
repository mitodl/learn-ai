"""ai_chatbots app config"""

from django.apps import AppConfig


class AiChatConfig(AppConfig):
    """AI Chat Appconfig"""

    name = "ai_chatbots"

    def ready(self):
        """Register the app's system checks"""
        from ai_chatbots import checks  # noqa: F401
