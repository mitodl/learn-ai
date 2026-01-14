import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from asgiref.sync import sync_to_async

from ai_chatbots import utils
from ai_chatbots.checkpointers import AsyncDjangoSaver
from ai_chatbots.factories import (
    HumanMessageFactory,
    SyllabusAgentStateFactory,
    SystemMessageFactory,
    ToolMessageFactory,
    VideoGPTAgentStateFactory,
)
from main.factories import UserFactory


@pytest.fixture(autouse=True)
def _reset_http_clients():
    """Reset HTTP client singletons before and after each test."""
    utils.http_client_manager.reset()
    utils.http_client_manager.reset()


@pytest.fixture
def mock_stdout(mocker):
    """Create mock stdout for testing."""
    return mocker.Mock()


@pytest.fixture(autouse=True)
def mock_settings(settings):
    """Langsmith API should be blank for most tests"""
    os.environ["LANGSMITH_API_KEY"] = ""
    os.environ["LANGSMITH_TRACING"] = "false"
    settings.LANGSMITH_API_KEY = ""
    return settings


@pytest.fixture
async def async_user():
    """Return a user for the agent."""
    return await sync_to_async(UserFactory.create)()


@pytest.fixture
def sync_user():
    """Return a user for the agent."""
    return UserFactory.create()


@pytest.fixture
def django_session():
    """Create a mock Django session for testing."""
    from uuid import uuid4

    session = Mock()
    session.session_key = f"test_session_{uuid4().hex}"
    session.save = Mock()
    return session


@pytest.fixture(autouse=True)
def mock_check_throttles(mocker):
    """Mock check_throttles to avoid needing ConsumerThrottleLimit DB entries."""
    return mocker.patch(
        "ai_chatbots.consumers.BaseBotHttpConsumer.check_throttles",
        return_value=None,
    )


@pytest.fixture(autouse=True)
def ai_settings(settings, mocker):
    """Assign default AI settings"""
    # Reset HTTP client singletons before each test
    from ai_chatbots import utils

    utils.sync_http_client = None
    utils.async_http_client = None

    settings.AI_PROXY = None
    settings.AI_PROXY_URL = None
    settings.AI_PROXY_AUTH_TOKEN = "test_token"  # noqa: S105
    settings.LEARN_ACCESS_TOKEN = "test_token"  # noqa: S105
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


@pytest.fixture
async def mock_checkpointer(mocker) -> AsyncDjangoSaver:  # noqa: ARG001
    """Mock the checkpointer"""
    return await AsyncDjangoSaver.create_with_session(
        uuid4(), "test message", "test_bot"
    )


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
        exclude_canvas=["True", "True"],
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


@pytest.fixture
def mock_httpx_client(mocker):
    """Mock httpx.Client for synchronous HTTP requests using shared client."""

    def _mock_client(json_return_value, status_code=200, patch_path=None):
        """
        Create a mock httpx.Client for testing.

        Args:
            json_return_value: The value to return from response.json()
            status_code: HTTP status code to return (default: 200)
            patch_path: Optional path to patch
        """
        mock_response = mocker.Mock()
        mock_response.json.return_value = json_return_value
        mock_response.status_code = status_code
        mock_response.raise_for_status = mocker.Mock()

        mock_client = mocker.Mock()
        mock_client.get = mocker.Mock(return_value=mock_response)
        mock_client.post = mocker.Mock(return_value=mock_response)
        mock_client.close = mocker.Mock()

        if patch_path:
            return mocker.patch(patch_path, return_value=mock_client)
        return mock_client

    return _mock_client


@pytest.fixture
def mock_httpx_async_client(mocker):
    """Mock httpx.AsyncClient for asynchronous HTTP requests using shared client."""

    def _mock_async_client(json_return_value, status_code=200, patch_path=None):
        """
        Create a mock httpx.AsyncClient for testing.

        Args:
            json_return_value: The value to return from response.json()
            status_code: HTTP status code to return (default: 200)
            patch_path: Optional path to patch
        """
        mock_response = mocker.Mock()
        mock_response.json.return_value = json_return_value
        mock_response.status_code = status_code
        mock_response.raise_for_status = mocker.Mock()

        mock_client = mocker.Mock()
        mock_client.get = mocker.AsyncMock(return_value=mock_response)
        mock_client.post = mocker.AsyncMock(return_value=mock_response)
        mock_client.aclose = mocker.AsyncMock()

        if patch_path:
            return mocker.patch(patch_path, return_value=mock_client)
        return mock_client

    return _mock_async_client
