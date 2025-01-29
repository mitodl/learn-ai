"""Test factory classes for ai_chatbots tests"""

import factory
from factory.fuzzy import FuzzyChoice
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.ai import AIMessageChunk

from ai_chatbots.chatbots import SyllabusAgentState


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
    values = {
        "messages": [
            factory.SubFactory(HumanMessageFactory),
        ]
    }
    status: FuzzyChoice(["success", "error"])

    class Meta:
        model = ToolMessage


class AIMessageChunkFactory(BaseMessageFactory):
    """Factory for generating AIMessageChunk instances."""

    class Meta:
        model = AIMessageChunk


class SyllabusAgentStateFactory(factory.Factory):
    """Factory for generating SyllabusAgentState instances."""

    messages = [factory.SubFactory(HumanMessageFactory)]
    course_id = [factory.Faker("uuid4")]
    collection_name = [factory.Faker("word")]

    class Meta:
        model = SyllabusAgentState
