import pytest
from llama_cloud import MessageRole

from ai_chatbots.factories import ChatMessageFactory


@pytest.fixture(autouse=True)
def ai_settings(settings):
    """Assign default AI settings"""
    settings.AI_PROXY = None
    settings.AI_PROXY_URL = None
    return settings


@pytest.fixture
def chat_history():
    """Return one round trip chat history for testing."""
    return [
        ChatMessageFactory.create(role=MessageRole.USER),
        ChatMessageFactory.create(role=MessageRole.ASSISTANT),
        ChatMessageFactory.create(role=MessageRole.USER),
        ChatMessageFactory.create(role=MessageRole.ASSISTANT),
    ]
