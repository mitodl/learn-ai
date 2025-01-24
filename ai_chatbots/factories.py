"""Test factory classes for ai_chatbots tests"""

import factory
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
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
    """Factory for generating HumantMessage instances."""

    class Meta:
        model = HumanMessage


class ToolMessageFactory(BaseMessageFactory):
    """Factory for generating ToolMessage instances."""

    tool_call_id = factory.Faker("uuid4")
    values = {"messages": [factory.SubFactory(SystemMessageFactory)]}

    class Meta:
        model = ToolMessage


class AIMessageChunkFactory(BaseMessageFactory):
    """Factory for generating AIMessageChunk instances."""

    class Meta:
        model = AIMessageChunk
