"""Test factory classes for ai_chatbots tests"""

import factory
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.ai import AIMessageChunk


class BaseMessageFactory(factory.Factory):
    """Factory for generating BaseMessage instances."""

    content = factory.Faker("sentence")

    class Meta:
        model = BaseMessage


class SystemMessageFactory(BaseMessageFactory):
    """Factory for generating SystemChatMessage instances."""

    class Meta:
        model = SystemMessage


class HumanMessageFactory(BaseMessageFactory):
    """Factory for generating HumanChatMessage instances."""

    class Meta:
        model = HumanMessage


class AIMessageChunkFactory(BaseMessageFactory):
    """Factory for generating AIMessageChunk instances."""

    class Meta:
        model = AIMessageChunk
