"""Tests for ai_chatbots utils"""

import asyncio
import json

import httpx
import pytest
from channels.db import database_sync_to_async
from django.conf import settings
from langchain_core.messages import AIMessage
from named_enum import ExtendedEnum

from ai_chatbots import utils
from ai_chatbots.factories import (
    CheckpointFactory,
    HumanMessageFactory,
    SystemMessageFactory,
)


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
        "https://api.example.com/test",
        {"param1": "value1"},
        follow_redirects=False,
        timeout=15,
    )

    mock_client.get.assert_called_once_with(
        "https://api.example.com/test",
        params={"param1": "value1"},
        headers={"Authorization": "Bearer test_token_123"},
        follow_redirects=False,
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


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_save_truncated_checkpoint():
    """Should truncate messages and save to DB."""
    checkpoint_data = {
        "channel_values": {
            "messages": [
                {"kwargs": {"id": "remove_me"}},
                {"kwargs": {"id": "keep_me"}},
                {"kwargs": {"id": "remove_me 2"}},
            ]
        }
    }
    checkpoint = await database_sync_to_async(CheckpointFactory)(
        checkpoint=json.dumps(checkpoint_data)
    )

    await utils.save_truncated_checkpoint(checkpoint.thread_id, {"keep_me"})

    await database_sync_to_async(checkpoint.refresh_from_db)()
    saved = json.loads(checkpoint.checkpoint)
    assert len(saved["channel_values"]["messages"]) == 1
    assert saved["channel_values"]["messages"][0]["kwargs"]["id"] == "keep_me"


@pytest.mark.asyncio
@pytest.mark.django_db(transaction=True)
async def test_save_truncated_checkpoint_no_checkpoint():
    """Should not error when no checkpoint exists."""
    await utils.save_truncated_checkpoint("nonexistent_thread", {"any"})


@pytest.mark.asyncio
@pytest.mark.django_db
async def test_truncate_to_latest_human_message(mock_checkpointer):
    """Should keep only messages from last HumanMessage onward."""
    messages = [
        SystemMessageFactory.create(),
        HumanMessageFactory.create(content="first"),
        AIMessage(content="response1"),
        HumanMessageFactory.create(content="second"),
        AIMessage(content="response2"),
    ]
    result = utils.truncate_to_latest_human_message(messages)
    assert len(result) == 2
    assert result[0].content == "second"
    assert result[1].content == "response2"


@pytest.mark.asyncio
@pytest.mark.django_db
async def test_truncate_to_latest_human_message_missing(mock_checkpointer):
    """Should return empty list when no HumanMessage found."""
    messages = [SystemMessageFactory.create(), AIMessage(content="response")]
    result = utils.truncate_to_latest_human_message(messages)
    assert result == []


@pytest.mark.asyncio
@pytest.mark.django_db
async def test_truncate_to_latest_human_message_empty(mock_checkpointer):
    """Should return empty list for empty input."""
    assert utils.truncate_to_latest_human_message([]) == []


@pytest.mark.asyncio
@pytest.mark.django_db
async def test_truncate_checkpoint_messages_filters_messages(mock_checkpointer):
    """Should filter checkpoint messages to only keep specified IDs."""
    from ai_chatbots.utils import save_truncated_checkpoint

    thread_id = mock_checkpointer.session.thread_id
    keep_msg = HumanMessageFactory.create(content="keep")
    checkpoint_data = {
        "channel_values": {
            "messages": [
                {"kwargs": {"id": "remove_id"}},
                {"kwargs": {"id": keep_msg.id}},
            ]
        }
    }
    checkpoint = await database_sync_to_async(CheckpointFactory)(
        thread_id=thread_id,
        checkpoint=json.dumps(checkpoint_data),
    )

    await save_truncated_checkpoint(thread_id, {keep_msg.id})

    await database_sync_to_async(checkpoint.refresh_from_db)()
    saved_data = json.loads(checkpoint.checkpoint)
    assert len(saved_data["channel_values"]["messages"]) == 1
    assert saved_data["channel_values"]["messages"][0]["kwargs"]["id"] == keep_msg.id
