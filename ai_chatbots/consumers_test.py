"""Tests for ai_chatbots consumers"""

import json
from random import randint
from unittest.mock import AsyncMock

import pytest

from ai_chatbots import consumers
from ai_chatbots.chatbots import ResourceRecommendationBot
from ai_chatbots.conftest import MockAsyncIterator
from ai_chatbots.constants import AI_THREAD_COOKIE_KEY
from ai_chatbots.factories import SystemMessageFactory
from main.factories import UserFactory


@pytest.fixture
def agent_user():
    """Return a user for the agent."""
    return UserFactory.build(
        username=f"test_user_{randint(1, 1000)}"  # noqa: S311
    )


@pytest.fixture
def recommendation_consumer(agent_user):
    """Return a recommendation consumer."""
    consumer = consumers.RecommendationBotHttpConsumer()
    consumer.scope = {"user": agent_user, "cookies": {}, "session": None}
    consumer.channel_name = "test_channel"
    return consumer


@pytest.mark.parametrize(
    ("message", "temperature", "instructions", "model"),
    [
        ("hello", 0.7, "Answer this question as best you can", "gpt-3.5-turbo"),
        ("hello", 0.7, "", "gpt-3.5-turbo"),
        ("hello", 0.6, None, "gpt-4-turbo"),
        ("hello", 0.4, None, "gpt-4o"),
        ("hello", 0.4, None, ""),
        ("hello", None, None, None),
    ],
)
async def test_recommend_agent_handle(  # noqa: PLR0913
    settings,
    mocker,
    mock_http_consumer_send,
    recommendation_consumer,
    message,
    temperature,
    instructions,
    model,
):
    """Test the receive function of the recommendation agent."""
    response = SystemMessageFactory.create()
    mock_completion = mocker.patch(
        "ai_chatbots.chatbots.ResourceRecommendationBot.get_completion",
        return_value=mocker.Mock(
            __aiter__=mocker.Mock(
                return_value=MockAsyncIterator(list(response.content.split(" ")))
            )
        ),
    )
    data = {
        "message": message,
    }
    if temperature:
        data["temperature"] = temperature
    if instructions is not None:
        data["instructions"] = instructions
    if model is not None:
        data["model"] = model
    await recommendation_consumer.handle(json.dumps(data))

    assert recommendation_consumer.bot.user_id.startswith("test_user")
    assert recommendation_consumer.bot.llm.temperature == (
        temperature if temperature else settings.AI_DEFAULT_TEMPERATURE
    )
    assert recommendation_consumer.bot.llm.model_name == (
        model if model else settings.AI_MODEL
    )
    assert recommendation_consumer.bot.instructions == (
        instructions if instructions else ResourceRecommendationBot.INSTRUCTIONS
    )

    mock_completion.assert_called_once_with(message)
    assert (
        mock_http_consumer_send.send_body.call_count
        == len(response.content.split(" ")) + 2
    )
    mock_http_consumer_send.send_body.assert_any_call(
        body=response.content.split(" ")[0].encode("utf-8"),
        more_body=True,
    )
    assert mock_http_consumer_send.send_headers.call_count == 1


@pytest.mark.parametrize("clear_history", [True, False])
async def test_clear_history(
    mocker, mock_http_consumer_send, recommendation_consumer, clear_history
):
    """Test the clear history function of the recommendation agent."""
    recommendation_consumer.scope["cookies"] = {AI_THREAD_COOKIE_KEY: b"1234"}
    mocker.patch(
        "ai_chatbots.chatbots.ResourceRecommendationBot.get_completion",
    )
    await recommendation_consumer.handle(
        json.dumps({"clear_history": clear_history, "message": "hello"})
    )
    args = mock_http_consumer_send.send_headers.call_args_list
    assert (
        recommendation_consumer.scope["cookies"][AI_THREAD_COOKIE_KEY]
        in args[0][-1]["headers"][-1][1]
    ) != clear_history


async def test_http_request(mocker):
    """Test the http request function of the AsyncHttpConsumer"""
    msg = {"body": "test"}
    mock_handle = mocker.patch(
        "ai_chatbots.consumers.RecommendationBotHttpConsumer.handle"
    )
    consumer = consumers.RecommendationBotHttpConsumer()
    await consumer.http_request(msg)
    mock_handle.assert_called_once_with(msg["body"])


@pytest.mark.parametrize("has_layer", [True, False])
async def test_disconnect(mocker, recommendation_consumer, has_layer):
    """Test the disconnect function of the recommendation agent."""
    recommendation_consumer.user_id = "Anonymous"
    mock_layer = mocker.Mock(group_discard=AsyncMock())
    if has_layer:
        recommendation_consumer.channel_layer = mock_layer
    await recommendation_consumer.disconnect()
    assert mock_layer.group_discard.call_count == (1 if has_layer else 0)
