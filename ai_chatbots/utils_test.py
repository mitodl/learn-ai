"""Tests for ai_chatbots utils"""

import asyncio

import httpx
from django.conf import settings
from named_enum import ExtendedEnum

from ai_chatbots import utils


def test_enum_zip():
    """Test the enum_zip function."""

    class TestEnum(ExtendedEnum):
        """enum test class"""

        foo = "bar"
        fizz = "buzz"
        hello = "world"

    new_enum = utils.enum_zip("New_Enum", TestEnum)
    for item in new_enum:
        assert item.value == item.name


def test_request_with_token(mocker, settings):
    """Test synchronous request_with_token function."""
    settings.LEARN_ACCESS_TOKEN = "test_token_123"  # noqa: S105

    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": "success"}

    mock_client = mocker.Mock()
    mock_client.get = mocker.Mock(return_value=mock_response)

    mocker.patch("ai_chatbots.utils.get_sync_http_client", return_value=mock_client)

    response = utils.request_with_token(
        "https://api.example.com/test", {"param1": "value1"}, timeout=15
    )

    mock_client.get.assert_called_once_with(
        "https://api.example.com/test",
        params={"param1": "value1"},
        headers={"Authorization": "Bearer test_token_123"},
        timeout=15,
    )
    assert response.status_code == 200
    assert response.json() == {"result": "success"}


async def test_async_request_with_token(mocker, settings):
    """Test asynchronous async_request_with_token function."""
    settings.LEARN_ACCESS_TOKEN = "test_token_456"  # noqa: S105

    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": "async_success"}
    mock_response.raise_for_status = mocker.Mock()

    mock_client = mocker.Mock()
    mock_client.get = mocker.AsyncMock(return_value=mock_response)

    mocker.patch("ai_chatbots.utils.get_async_http_client", return_value=mock_client)

    response = await utils.async_request_with_token(
        "https://api.example.com/async", {"param2": "value2"}, timeout=20
    )

    mock_client.get.assert_called_once_with(
        "https://api.example.com/async",
        params={"param2": "value2"},
        headers={"Authorization": "Bearer test_token_456"},
        timeout=20,
    )
    assert response.status_code == 200
    assert response.json() == {"result": "async_success"}


def test_get_sync_http_client_singleton():
    """Test that get_sync_http_client returns the same instance."""
    client1 = utils.get_sync_http_client()
    client2 = utils.get_sync_http_client()

    assert client1 is client2
    assert hasattr(client1, "get")


def test_get_async_http_client_singleton():
    """Test that get_async_http_client returns the same instance."""
    client1 = utils.get_async_http_client()
    client2 = utils.get_async_http_client()

    assert client1 is client2
    assert hasattr(client1, "get")


def test_sync_client_connection_pooling_config():
    """Test that sync client has proper connection pooling configuration."""
    client = utils.get_sync_http_client()

    assert client.timeout.connect == settings.REQUESTS_TIMEOUT

    # Verify client is a properly configured httpx Client instance
    assert isinstance(client, httpx.Client)
    assert not client.is_closed


def test_async_client_connection_pooling_config():
    """Test that async client has proper connection pooling configuration."""
    client = utils.get_async_http_client()

    assert client.timeout.connect == settings.REQUESTS_TIMEOUT

    # Verify client is a properly configured httpx AsyncClient instance
    assert isinstance(client, httpx.AsyncClient)
    assert not client.is_closed


async def test_concurrent_async_requests_pooling(mocker):
    """Test connection pooling with synchronous httpx client."""
    # Track which client instance is used for each call
    client_instances = []

    async def mock_request(request_id):
        # Get the client and track its instance ID
        client = utils.get_async_http_client()
        client.post = mocker.AsyncMock(
            return_value={"request_id": request_id, "client_id": id(client)}
        )
        client_instances.append(id(client))

        # Verify the client is accessible concurrently and not closed
        assert client is not None
        assert not client.is_closed
        return await client.post("/test", json={"request_id": request_id})

    # Simulate 10 concurrent users making requests
    results = await asyncio.gather(*[mock_request(i) for i in range(10)])

    # Verify all requests used the same client instance
    client_ids = {r["client_id"] for r in results}
    assert len(client_ids) == 1
    assert len(results) == 10


async def test_concurrent_sync_requests_pooling(mocker):
    """Test connection pooling with synchronous httpx client."""

    # Verify client supports concurrent usage
    async def mock_request(request_id):
        client = utils.get_sync_http_client()
        # Verify the client is accessible concurrently and not closed
        assert client is not None
        assert not client.is_closed
        client.post = mocker.Mock(
            return_value={"request_id": request_id, "client_id": id(client)}
        )
        return client.post("/test", json={"request_id": request_id})

    # Run 50 concurrent "requests"
    results = await asyncio.gather(*[mock_request(i) for i in range(10)])

    # Verify all requests used the same client instance
    client_ids = {r["client_id"] for r in results}
    assert len(client_ids) == 1
    assert len(results) == 10


def test_collect_answered_tool_call_ids():
    """Should ignore AIMessages and HumanMessages, just get ToolMessage."""
    messages = [
        {"id": ["langchain", "schema", "messages", "HumanMessage"]},
        {
            "id": ["langchain", "schema", "messages", "AIMessage"],
            "kwargs": {"tool_call_id": "ignored"},
        },
        {
            "id": ["langchain", "schema", "messages", "ToolMessage"],
            "kwargs": {"tool_call_id": "call_123"},
        },
    ]
    assert utils.collect_answered_tool_call_ids(messages) == {"call_123"}


def test_collect_answered_tool_call_ids_handles_empty():
    """Should return empty set for empty messages."""
    assert utils.collect_answered_tool_call_ids([]) == set()


def test_collect_answered_tool_call_ids_skips_missing():
    """Should skip ToolMessages without tool_call_id."""
    messages = [
        {"id": ["langchain", "schema", "messages", "ToolMessage"], "kwargs": {}}
    ]
    assert utils.collect_answered_tool_call_ids(messages) == set()


def test_filter_orphaned_tool_calls_removes_orphaned():
    """Should remove tool calls without matching ToolMessage responses."""
    messages = [
        {
            "id": ["langchain", "schema", "messages", "AIMessage"],
            "kwargs": {"tool_calls": [{"id": "call_123"}, {"id": "call_orphaned"}]},
        },
    ]
    modified = utils.filter_orphaned_tool_calls(messages, {"call_123"})
    assert modified is True
    assert messages[0]["kwargs"]["tool_calls"] == [{"id": "call_123"}]


def test_filter_orphaned_tool_calls_removes_all_when_none_answered():
    """Should remove tool_calls key when no tool calls are answered."""
    messages = [
        {
            "id": ["langchain", "schema", "messages", "AIMessage"],
            "kwargs": {"tool_calls": [{"id": "orphaned"}]},
        },
    ]
    modified = utils.filter_orphaned_tool_calls(messages, set())
    assert modified is True
    assert "tool_calls" not in messages[0]["kwargs"]


def test_filter_orphaned_tool_calls_no_modification_when_all_answered():
    """Should return False when all tool calls have responses."""
    messages = [
        {
            "id": ["langchain", "schema", "messages", "AIMessage"],
            "kwargs": {"tool_calls": [{"id": "call_123"}]},
        },
    ]
    modified = utils.filter_orphaned_tool_calls(messages, {"call_123"})
    assert modified is False


def test_filter_orphaned_tool_calls_handles_additional_kwargs():
    """Should also clean additional_kwargs.tool_calls."""
    messages = [
        {
            "id": ["langchain", "schema", "messages", "AIMessage"],
            "kwargs": {
                "tool_calls": [{"id": "orphaned"}],
                "additional_kwargs": {"tool_calls": [{"id": "orphaned"}]},
            },
        },
    ]
    utils.filter_orphaned_tool_calls(messages, set())
    assert "tool_calls" not in messages[0]["kwargs"].get("additional_kwargs", {})
