"""Test factory classes for ai_chatbots tests"""

from random import randint
from uuid import uuid4

import factory
from factory.django import DjangoModelFactory
from factory.fuzzy import FuzzyChoice, FuzzyText
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.ai import AIMessage, AIMessageChunk
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from ai_chatbots import models
from ai_chatbots.chatbots import (
    RecommendationAgentState,
    ResourceRecommendationBot,
    SyllabusAgentState,
    VideoGPTAgentState,
)
from main.factories import UserFactory


def generate_user_metadata() -> dict:
    """Generate metadata for a user message."""
    return {
        "step": randint(-1, 100),  # noqa: S311
        "writes": {
            "__start__": {
                "messages": [
                    {
                        "kwargs": {
                            "type": "human",
                            "content": FuzzyText().fuzz(),
                        }
                    }
                ]
            }
        },
    }


def generate_agent_metadata() -> dict:
    """Generate metadata for an agent message."""
    return {
        "step": randint(-1, 100),  # noqa: S311
        "writes": {
            "agent": {
                "messages": [
                    {
                        "kwargs": {
                            "type": "ai",
                            "content": FuzzyText().fuzz(),
                        }
                    }
                ]
            }
        },
    }


def generate_tool_metadata() -> dict:
    """Generate metadata for a tool message."""
    return {
        "step": randint(-1, 100),  # noqa: S311
        "writes": {
            "tools": {
                "messages": [
                    {
                        "kwargs": {
                            "name": "search_courses",
                            "type": "tool",
                            "content": FuzzyText().fuzz(),
                        }
                    }
                ]
            }
        },
    }


def generate_sample_checkpoint_data() -> dict:
    """Generate sample checkpoint data."""
    return {
        "v": 1,
        "ts": "2025-02-09T15:09:06.378971+00:00",
        "id": uuid4().hex,
        "channel_values": {
            "messages": [
                {
                    "lc": 1,
                    "type": "constructor",
                    "id": ["langchain", "schema", "messages", "SystemMessage"],
                    "kwargs": {
                        "content": "Answer questions",
                        "type": "system",
                        "id": "0788c616-da2c-4d08-8f92-52aefb94d474",
                    },
                },
                {
                    "lc": 1,
                    "type": "constructor",
                    "id": ["langchain", "schema", "messages", "HumanMessage"],
                    "kwargs": {
                        "content": "Tell me about ocean currents",
                        "type": "human",
                        "id": "d6231cfe-2e7c-4f93-8797-559f9addfcdd",
                    },
                },
                {
                    "lc": 1,
                    "type": "constructor",
                    "id": ["langchain", "schema", "messages", "AIMessage"],
                    "kwargs": {
                        "content": "Ocean currents are.....",
                        "type": "ai",
                        "id": "run-a9af5489-2a49-4bc9-b5d9-22c43bd5f23f",
                        "tool_calls": [],
                        "invalid_tool_calls": [],
                    },
                },
            ],
            "agent": "agent",
        },
        "channel_versions": {
            "__start__": 2,
            "messages": 3,
            "start:agent": 3,
            "agent": 3,
        },
        "versions_seen": {
            "__input__": {},
            "__start__": {"__start__": 1},
            "agent": {"start:agent": 2},
        },
        "pending_sends": [],
    }


class BaseMessageFactory(factory.Factory):
    """Factory for generating BaseMessage instances."""

    content = factory.Faker("sentence")
    id = factory.Faker("uuid4")

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


class AIMessageFactory(BaseMessageFactory):
    """Factory for generating HumantMessage instances."""

    class Meta:
        model = AIMessage


class AIMessageChunkFactory(BaseMessageFactory):
    """Factory for generating AIMessageChunk instances."""

    class Meta:
        model = AIMessageChunk


class RecommendationAgentStateFactory(factory.Factory):
    """Factory for generating RecommendationAgent instances."""

    messages = [factory.SubFactory(HumanMessageFactory)]
    search_url = [factory.Faker("url")]

    class Meta:
        model = RecommendationAgentState


class SyllabusAgentStateFactory(factory.Factory):
    """Factory for generating SyllabusAgentState instances."""

    messages = [factory.SubFactory(HumanMessageFactory)]
    course_id = [factory.Faker("uuid4")]
    collection_name = [factory.Faker("word")]
    exclude_canvas = ["True"]

    class Meta:
        model = SyllabusAgentState


class VideoGPTAgentStateFactory(factory.Factory):
    """Factory for generating VideoGPTAgentState instances."""

    messages = [factory.SubFactory(HumanMessageFactory)]
    transcript_asset_id = [factory.Faker("uuid4")]

    class Meta:
        model = VideoGPTAgentState


class UserChatSessionFactory(DjangoModelFactory):
    """Factory for generating UserChatSession instances."""

    user = factory.SubFactory(UserFactory)
    thread_id = factory.Faker("uuid4")
    title = factory.Faker("sentence")
    agent = FuzzyChoice(
        [ResourceRecommendationBot.__name__, SyllabusAgentState.__name__]
    )

    class Meta:
        model = models.UserChatSession


class CheckpointFactory(DjangoModelFactory):
    """Factory for Langgraph checkpoints"""

    session = factory.SubFactory(UserChatSessionFactory)
    thread_id = uuid4().hex
    checkpoint_ns = FuzzyText()
    checkpoint_id = FuzzyText()
    parent_checkpoint_id = FuzzyText()
    checkpoint = generate_sample_checkpoint_data()
    metadata = FuzzyChoice(
        [generate_user_metadata(), generate_agent_metadata(), generate_tool_metadata()]
    )
    type = "msgpack"

    class Meta:
        model = models.DjangoCheckpoint

    class Params:
        is_human = factory.Trait(metadata=generate_user_metadata())
        is_agent = factory.Trait(metadata=generate_agent_metadata())
        is_tool = factory.Trait(metadata=generate_tool_metadata())


class CheckpointWriteFactory(DjangoModelFactory):
    """Factory for Langgraph checkpoint writes"""

    session = factory.SubFactory(UserChatSessionFactory)
    thread_id = uuid4().hex
    checkpoint_ns = FuzzyText()
    checkpoint_id = FuzzyText()
    task_id = FuzzyText()
    idx = randint(0, 100)  # noqa: S311
    channel = FuzzyText()
    type = "msgpack"
    blob = JsonPlusSerializer().dumps_typed(
        ("msgpack", generate_sample_checkpoint_data())
    )[1]
    task_path = FuzzyText()

    class Meta:
        model = models.DjangoCheckpointWrite


class LLMModelFactory(DjangoModelFactory):
    """Factory for LLMModel instances."""

    provider = FuzzyChoice(["openai", "anthropic", "meta", "deepseek", "google"])
    name = FuzzyText()
    litellm_id = FuzzyText()

    class Meta:
        model = models.LLMModel
