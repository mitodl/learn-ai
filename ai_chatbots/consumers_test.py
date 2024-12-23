"""Tests for ai_chatbots consumers"""

import json
from random import randint

import pytest
from llama_cloud import MessageRole
from llama_index.core.constants import DEFAULT_TEMPERATURE

from ai_chatbots import consumers
from ai_chatbots.chatbots import ResourceRecommendationBot
from ai_chatbots.factories import ChatMessageFactory
from main.factories import UserFactory


@pytest.fixture(autouse=True)
def mock_connect(mocker):
    """Mock the AsyncWebsocketConsumer connect function"""
    return mocker.patch(
        "ai_chatbots.consumers.AsyncWebsocketConsumer.connect",
        new_callable=mocker.AsyncMock,
    )


@pytest.fixture(autouse=True)
def mock_send(mocker):
    """Mock the AsyncWebsocketConsumer connect function"""
    return mocker.patch(
        "ai_chatbots.consumers.AsyncWebsocketConsumer.send",
        new_callable=mocker.AsyncMock,
    )


@pytest.fixture
def agent_user():
    """Return a user for the agent."""
    return UserFactory.build(
        username=f"test_user_{randint(1, 1000)}"  # noqa: S311
    )


@pytest.fixture
def recommendation_consumer(agent_user):
    """Return a recommendation consumer."""
    consumer = consumers.RecommendationBotWSConsumer()
    consumer.scope = {"user": agent_user}
    return consumer


async def test_recommend_agent_connect(
    recommendation_consumer, agent_user, mock_connect
):
    """Test the connect function of the recommendation agent."""
    await recommendation_consumer.connect()

    assert mock_connect.call_count == 1
    assert recommendation_consumer.user_id == agent_user.username
    assert recommendation_consumer.agent.user_id == agent_user.username


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
async def test_recommend_agent_receive(  # noqa: PLR0913
    settings,
    mocker,
    mock_send,
    recommendation_consumer,
    message,
    temperature,
    instructions,
    model,
):
    """Test the receive function of the recommendation agent."""
    response = ChatMessageFactory.create(role=MessageRole.ASSISTANT)
    mock_completion = mocker.patch(
        "ai_chatbots.chatbots.ResourceRecommendationBot.get_completion",
        side_effect=[response.content.split(" ")],
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
    await recommendation_consumer.connect()
    await recommendation_consumer.receive(json.dumps(data))

    assert recommendation_consumer.agent.user_id.startswith("test_user")
    agent_worker = recommendation_consumer.agent.agent.agent_worker
    assert agent_worker._llm.temperature == (  # noqa: SLF001
        temperature if temperature else DEFAULT_TEMPERATURE
    )
    assert recommendation_consumer.agent.agent.agent_worker.prefix_messages[
        0
    ].content == (
        instructions if instructions else ResourceRecommendationBot.INSTRUCTIONS
    )
    assert agent_worker._llm.model == (  # noqa: SLF001
        model if model else settings.AI_MODEL
    )

    mock_completion.assert_called_once_with(message)
    assert mock_send.call_count == len(response.content.split(" ")) + 1
    mock_send.assert_any_call(text_data="!endResponse")


@pytest.mark.parametrize("clear_history", [True, False])
async def test_clear_history(mocker, clear_history, recommendation_consumer):
    """Test the clear history function of the recommendation agent."""
    mock_clear = mocker.patch(
        "ai_chatbots.consumers.ResourceRecommendationBot.clear_chat_history"
    )
    mocker.patch(
        "ai_chatbots.chatbots.ResourceRecommendationBot.get_completion",
    )
    await recommendation_consumer.connect()
    await recommendation_consumer.receive(
        json.dumps({"clear_history": clear_history, "message": "hello"})
    )
    assert mock_clear.call_count == (1 if clear_history else 0)
