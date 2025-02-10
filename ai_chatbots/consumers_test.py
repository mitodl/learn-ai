"""Tests for ai_chatbots consumers"""

import json
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from asgiref.sync import sync_to_async
from django.contrib.auth.models import AnonymousUser

from ai_chatbots import consumers
from ai_chatbots.chatbots import ResourceRecommendationBot, SyllabusBot
from ai_chatbots.conftest import MockAsyncIterator
from ai_chatbots.constants import AI_THREAD_COOKIE_KEY, AI_THREADS_ANONYMOUS_COOKIE_KEY
from ai_chatbots.factories import SystemMessageFactory, UserChatSessionFactory
from ai_chatbots.models import UserChatSession

pytestmark = pytest.mark.django_db


@pytest.fixture
def recommendation_consumer(async_user):
    """Return a recommendation consumer."""
    consumer = consumers.RecommendationBotHttpConsumer()
    consumer.scope = {"user": async_user, "cookies": {}, "session": None}
    consumer.channel_name = "test_channel"
    return consumer


@pytest.fixture
def syllabus_consumer(async_user):
    """Return a syllabus consumer."""
    consumer = consumers.SyllabusBotHttpConsumer()
    consumer.scope = {"user": async_user, "cookies": {}, "session": None}
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
    user = recommendation_consumer.scope["user"]
    user.is_superuser = True
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

    assert (
        recommendation_consumer.bot.user_id
        == recommendation_consumer.scope["user"].global_id
    )
    assert recommendation_consumer.bot.llm.temperature == (
        temperature if temperature else settings.AI_DEFAULT_TEMPERATURE
    )
    assert recommendation_consumer.bot.llm.model == (
        model if model else settings.AI_DEFAULT_RECOMMENDATION_MODEL
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
@pytest.mark.parametrize(
    ("is_anon", "cookie_name"),
    [(False, AI_THREAD_COOKIE_KEY), (True, AI_THREADS_ANONYMOUS_COOKIE_KEY)],
)
async def test_clear_history(  # noqa: PLR0913
    mocker,
    mock_http_consumer_send,
    recommendation_consumer,
    async_user,
    clear_history,
    cookie_name,
    is_anon,
):
    """Test the clear history function of the recommendation agent."""
    mocker.patch(
        "ai_chatbots.chatbots.ResourceRecommendationBot.get_completion",
    )
    bot_cookie = f"{recommendation_consumer.ROOM_NAME}_{cookie_name}"
    latest_thread_id = f"1234{',' if is_anon else ''}"
    recommendation_consumer.scope["cookies"] = {bot_cookie: latest_thread_id}
    recommendation_consumer.scope["user"] = AnonymousUser() if is_anon else async_user
    await recommendation_consumer.handle(
        json.dumps({"clear_history": clear_history, "message": "hello"})
    )
    args = mock_http_consumer_send.send_headers.call_args_list
    if cookie_name == AI_THREAD_COOKIE_KEY:
        # old thread id should not be present at all
        cookie_args = str(args[0][-1]["headers"][-2][1])
        assert (latest_thread_id in cookie_args) != clear_history
    elif is_anon:
        # old thread id should be present in the cookie, but not last in the list
        cookie_args = str(args[0][-1]["headers"][-1][1])
        assert (
            latest_thread_id in cookie_args
            and f"{latest_thread_id};" not in cookie_args
        ) == clear_history
    else:
        # anon thread_ids should have been cleared from cookie
        assert cookie_args == f"{bot_cookie}=; Path=/; HttpOnly"


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
    mocker, mock_http_consumer_send, syllabus_consumer, async_user
):
    """SyllabusBotHttpConsumer create_chatbot function should return syllabus bot."""
    serializer = consumers.SyllabusChatRequestSerializer(
        data={
            "message": "hello",
            "course_id": "MITx+6.00.1x",
            "collection_name": "vector512",
            "temperature": 0.7,
            "model": "gpt-3.5-turbo",
        }
    )
    serializer.is_valid(raise_exception=True)
    await syllabus_consumer.prepare_response(serializer)
    mock_http_consumer_send.send_headers.assert_called_once()
    chatbot = syllabus_consumer.create_chatbot(serializer, mocker.Mock())
    assert isinstance(chatbot, SyllabusBot)
    assert chatbot.user_id == async_user.global_id
    assert chatbot.temperature == 0.7
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


@pytest.mark.parametrize(
    ("user_cookie", "anon_cookie", "is_anon"),
    [
        ("1234", "", False),  # User is authenticated
        ("1234", "", True),  # User recently logged out
        ("", "4567,8910", False),  # User just authenticated
        ("", "4567,9876", True),  # User is anonymous
        ("", "", True),  # Anonymous user 1st message
        ("", "", False),  # Authenticated user 1st message
    ],
)
async def test_assign_thread_cookie(
    syllabus_consumer, async_user, user_cookie, anon_cookie, is_anon
):
    """Test the cookie handling for the consumer."""

    anon_cookie_name = (
        f"{syllabus_consumer.ROOM_NAME}_{AI_THREADS_ANONYMOUS_COOKIE_KEY}"
    )
    user_cookie_name = f"{syllabus_consumer.ROOM_NAME}_{AI_THREAD_COOKIE_KEY}"

    syllabus_consumer.scope["cookies"] = {
        user_cookie_name: user_cookie,
        anon_cookie_name: anon_cookie,
    }

    user = AnonymousUser() if is_anon else async_user
    syllabus_consumer.scope["user"] = user

    # Clear out any old saved sessions
    await UserChatSession.objects.all().adelete()

    # Any thread ids in cookie should already have been saved in the database, so add them here
    if user_cookie:
        # Should already be associated with user
        await UserChatSession.objects.acreate(
            thread_id=user_cookie,
            user=(None if is_anon else user),
            agent=SyllabusBot.__name__,
        )
    elif anon_cookie:
        for thread in anon_cookie.split(","):
            # Should not yet be associated with user
            await UserChatSession.objects.acreate(
                thread_id=thread, user=None, agent=SyllabusBot.__name__
            )

    thread_id, cookies = await syllabus_consumer.assign_thread_cookies(user)

    if anon_cookie:  # User is or just was anonymous
        assert thread_id == anon_cookie.split(",")[-1]
        if not is_anon:  # User just authenticated, associate older sessions with user
            # Clear out anon cookie thread ids and set current thread id to auth user cookie
            assert cookies[1] == f"{anon_cookie_name}=; Path=/; HttpOnly"
            assert cookies[0] == f"{user_cookie_name}={thread_id}; Path=/; HttpOnly"
            for thread in anon_cookie.split(","):
                # All anon thread ids should now be associated with user
                assert await UserChatSession.objects.filter(
                    thread_id=thread, user=user, agent=SyllabusBot.__name__
                ).aexists()
        else:  # User is still anonymous
            for thread in anon_cookie.split(","):
                session = await UserChatSession.objects.aget(
                    thread_id=thread, agent=SyllabusBot.__name__
                )
                assert session.user is None
        # Current thread id should be in the correct cookie depending on user auth status
        assert thread_id in cookies[(1 if is_anon else 0)]
    else:  # User is either not anonymous or has no anon cookies
        assert (thread_id == user_cookie) is (user_cookie != "")
        if (
            is_anon
        ):  # User must have logged out, so current thread id should be in anon cookie
            assert cookies[1] == f"{anon_cookie_name}={thread_id},; Path=/; HttpOnly"
            assert cookies[0] == f"{user_cookie_name}=; Path=/; HttpOnly"


@pytest.mark.parametrize(
    ("is_anon", "is_authorized"), [(True, False), (False, True), (False, False)]
)
async def test_assign_thread_cookie_passed_thread(
    syllabus_consumer, async_user, is_anon, is_authorized
):
    """
    The thread_id in the request should override any cookies for an authorized user.
    """
    original_thread_id = uuid4().hex
    requested_thread_id = uuid4().hex
    user_cookie_name = f"{syllabus_consumer.ROOM_NAME}_{AI_THREAD_COOKIE_KEY}"
    syllabus_consumer.scope["cookies"] = {
        user_cookie_name: original_thread_id,
    }
    user = AnonymousUser() if is_anon else async_user
    await sync_to_async(UserChatSessionFactory.create)(
        thread_id=requested_thread_id, user=(async_user if is_authorized else None)
    )
    cookies = await syllabus_consumer.assign_thread_cookies(
        thread_id=requested_thread_id, user=user
    )
    assert (requested_thread_id in cookies[0]) is (is_authorized and not is_anon)
    assert (original_thread_id in cookies[0]) is (not is_authorized)


async def test_consumer_handle(mocker, mock_http_consumer_send, syllabus_consumer):
    """Test the handle function of the consumer."""
    response = SystemMessageFactory.create().content.split(" ")
    mock_completion = mocker.patch(
        "ai_chatbots.chatbots.SyllabusBot.get_completion",
        return_value=mocker.Mock(
            __aiter__=mocker.Mock(return_value=MockAsyncIterator(list(response)))
        ),
    )
    payload = {
        "message": "what are the prerequisites",
        "course_id": "MITx+6.00.1x",
        "collection_name": "vector512",
    }
    await syllabus_consumer.handle(json.dumps(payload))
    mock_completion.assert_called_once_with(
        payload["message"],
        extra_state={
            "course_id": [payload["course_id"]],
            "collection_name": [payload["collection_name"]],
        },
    )
    assert await UserChatSession.objects.filter(
        thread_id=syllabus_consumer.thread_id,
        user=syllabus_consumer.scope["user"],
        title=payload["message"],
        agent=SyllabusBot.__name__,
    ).aexists()
