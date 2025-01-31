"""Tests for ai_chatbots consumers"""

import json
from random import randint
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from ai_chatbots import consumers
from ai_chatbots.chatbots import ResourceRecommendationBot, SyllabusBot
from ai_chatbots.conftest import MockAsyncIterator
from ai_chatbots.constants import AI_THREAD_COOKIE_KEY
from ai_chatbots.factories import SystemMessageFactory
from main.factories import UserFactory


@pytest.fixture
def agent_user():
    """Return a user for the agent."""
    return UserFactory.build(
        username=f"test_user_{randint(1, 1000)}",  # noqa: S311
        global_id=f"test_user_{uuid4()!s}",
    )


@pytest.fixture
def recommendation_consumer(agent_user):
    """Return a recommendation consumer."""
    consumer = consumers.RecommendationBotHttpConsumer()
    consumer.scope = {"user": agent_user, "cookies": {}, "session": None}
    consumer.channel_name = "test_channel"
    return consumer


@pytest.fixture
def syllabus_consumer(agent_user):
    """Return a syllabus consumer."""
    consumer = consumers.SyllabusBotHttpConsumer()
    consumer.scope = {"user": agent_user, "cookies": {}, "session": None}
    consumer.channel_name = "test_syllabus_channel"
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

    mock_completion.assert_called_once_with(message, extra_state=None)
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
        recommendation_consumer.room_group_name = "test_name"
    await recommendation_consumer.disconnect()
    assert mock_layer.group_discard.call_count == (1 if has_layer else 0)


async def test_syllabus_create_chatbot(
    mock_http_consumer_send, syllabus_consumer, agent_user
):
    """SyllabusBotHttpConsumer create_chatbot function should return syllabus bot."""
    serializer = consumers.SyllabusChatRequestSerializer(
        data={
            "message": "hello",
            "course_id": "MITx+6.00.1x",
            "collection_name": "vector512",
            "temperature": 0.7,
            "instructions": "Answer this question as best you can",
            "model": "gpt-3.5-turbo",
        }
    )
    serializer.is_valid(raise_exception=True)
    await syllabus_consumer.prepare_response(serializer)
    mock_http_consumer_send.send_headers.assert_called_once()
    chatbot = syllabus_consumer.create_chatbot(serializer)
    assert isinstance(chatbot, SyllabusBot)
    assert chatbot.user_id == agent_user.global_id.replace("-", "_")
    assert chatbot.temperature == 0.7
    assert chatbot.instructions == "Answer this question as best you can"
    assert chatbot.model == "gpt-3.5-turbo"


@pytest.mark.parametrize(
    "request_params",
    [
        {"message": "hello", "course_id": "MITx+6.00.1x"},
        {
            "message": "bonjour",
            "course_id": "MITx+9.00.2x",
            "collection_name": "vector512",
        },
    ],
)
def test_process_extra_state(request_params):
    """Test that the process_extra_state function returns the expected values."""
    consumer = consumers.SyllabusBotHttpConsumer()
    assert consumer.process_extra_state(request_params) == {
        "course_id": [request_params.get("course_id")],
        "collection_name": [request_params.get("collection_name", None)],
    }
