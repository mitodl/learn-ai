import pytest
from llama_cloud import MessageRole

from ai_agents.factories import ChatMessageFactory


@pytest.fixture
def chat_history():
    """Return one round trip chat history for testing."""
    return [
        ChatMessageFactory.create(role=MessageRole.USER),
        ChatMessageFactory.create(role=MessageRole.ASSISTANT),
        ChatMessageFactory.create(role=MessageRole.USER),
        ChatMessageFactory.create(role=MessageRole.ASSISTANT),
    ]
