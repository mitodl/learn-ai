import json
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from asgiref.sync import sync_to_async

from ai_chatbots.factories import (
    HumanMessageFactory,
    SyllabusAgentStateFactory,
    SystemMessageFactory,
    ToolMessageFactory,
    VideoGPTAgentStateFactory,
)
from main.factories import UserFactory


@pytest.fixture
async def async_user():
    """Return a user for the agent."""
    return await sync_to_async(UserFactory.create)()


@pytest.fixture
def sync_user():
    """Return a user for the agent."""
    return UserFactory.create()


@pytest.fixture(autouse=True)
def ai_settings(settings, mocker):
    """Assign default AI settings"""
    settings.AI_PROXY = None
    settings.AI_PROXY_URL = None
    settings.OPENAI_API_KEY = "test_key"
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    mocker.patch("channels_redis.core.RedisChannelLayer", return_value=AsyncMock())
    return settings


@pytest.fixture
def chat_history():
    """Return a 2-round trip chat history for testing."""
    return [
        HumanMessageFactory.create(),
        SystemMessageFactory.create(),
        HumanMessageFactory.create(),
        SystemMessageFactory.create(),
    ]


class MockAsyncIterator:
    """An async iterator for testing purposes."""

    def __init__(self, seq):
        self.iter = iter(seq)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.iter)
        except StopIteration:
            raise StopAsyncIteration from StopIteration


@pytest.fixture
def search_results():
    """Return search results for testing."""
    with Path.open("./test_json/search_results.json") as f:
        yield json.loads(f.read())


@pytest.fixture
def content_chunk_results():
    """Return content file vector chunks for testing."""
    with Path.open("./test_json/content_file_chunks.json") as f:
        yield json.loads(f.read())


@pytest.fixture
def video_transcript_content_chunk_results():
    """Return content file vector chunks for testing video GPT."""
    with Path.open("./test_json/video_transcript_chunks.json") as f:
        yield json.loads(f.read())


@pytest.fixture
def syllabus_agent_state():
    """Return a syllabus agent state for testing."""
    return SyllabusAgentStateFactory(
        messages=[
            HumanMessageFactory.create(content="prerequisites"),
            ToolMessageFactory.create(tool_call="search_contentfiles"),
            HumanMessageFactory.create(content="main topics"),
            ToolMessageFactory.create(tool_call="search_contentfiles"),
        ],
        course_id=["MITx+10.00.2x", "MITx+6.00.1x"],
        collection_name=[None, "vector512"],
    )


@pytest.fixture
def video_gpt_agent_state():
    """Return a video gpt agent state for testing."""
    return VideoGPTAgentStateFactory(
        messages=[
            HumanMessageFactory.create(content="What is this video about?"),
            ToolMessageFactory.create(tool_call="get_video_transcript_chunk"),
            HumanMessageFactory.create(content="What topic does the video discuss?"),
            ToolMessageFactory.create(tool_call="get_video_transcript_chunk"),
        ],
        transcript_asset_id=[
            "asset-v1:xPRO+LASERxE3+R15+type@asset+block@469c03c4-581a-4687-a9ca-7a1c4047832d-en"
        ],
    )
