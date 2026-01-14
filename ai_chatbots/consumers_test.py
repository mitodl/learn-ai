"""Tests for ai_chatbots consumers"""

import json
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib.auth.models import AnonymousUser
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode
from rest_framework.exceptions import ValidationError

from ai_chatbots import consumers, prompts
from ai_chatbots.chatbots import SyllabusBot, VideoGPTBot
from ai_chatbots.conftest import MockAsyncIterator
from ai_chatbots.constants import (
    AI_SESSION_COOKIE_KEY,
    AI_THREAD_COOKIE_KEY,
    AI_THREADS_ANONYMOUS_COOKIE_KEY,
)
from ai_chatbots.factories import SystemMessageFactory, UserChatSessionFactory
from ai_chatbots.models import UserChatSession
from main.exceptions import AsyncThrottled
from main.factories import UserFactory

pytestmark = pytest.mark.django_db


@pytest.fixture
def recommendation_consumer(async_user, django_session):
    """Return a recommendation consumer."""
    consumer = consumers.RecommendationBotHttpConsumer()
    consumer.scope = {
        "user": async_user,
        "cookies": {AI_SESSION_COOKIE_KEY: "test_session_key"},
        "session": django_session,
    }
    consumer.channel_name = "test_channel"
    return consumer


@pytest.fixture
def syllabus_consumer(async_user, django_session):
    """Return a syllabus consumer."""
    consumer = consumers.SyllabusBotHttpConsumer()
    consumer.scope = {
        "user": async_user,
        "cookies": {AI_SESSION_COOKIE_KEY: "test_session_key"},
        "session": django_session,
    }
    consumer.channel_name = "test_syllabus_channel"
    return consumer


@pytest.fixture
def canvas_syllabus_consumer(async_user, django_session):
    """Return a syllabus canvas consumer."""
    consumer = consumers.CanvasSyllabusBotHttpConsumer()
    consumer.scope = {
        "user": async_user,
        "cookies": {AI_SESSION_COOKIE_KEY: "test_session_key"},
        "session": django_session,
    }
    consumer.channel_name = "test_syllabus_canvas_channel"
    return consumer


@pytest.fixture
def tutor_consumer(async_user, django_session):
    """Return a tutor consumer."""
    consumer = consumers.TutorBotHttpConsumer()
    consumer.scope = {
        "user": async_user,
        "cookies": {AI_SESSION_COOKIE_KEY: "test_session_key"},
        "session": django_session,
    }
    consumer.channel_name = "test_tutor_channel"
    return consumer


@pytest.fixture
def canvas_tutor_consumer(async_user, django_session):
    """Return a canvas tutor consumer."""
    consumer = consumers.CanvasTutorBotHttpConsumer()
    consumer.scope = {
        "user": async_user,
        "cookies": {AI_SESSION_COOKIE_KEY: "test_session_key"},
        "session": django_session,
    }
    consumer.channel_name = "test_tutor_channel"
    return consumer


@pytest.fixture
def video_gpt_consumer(async_user, django_session):
    """Return a video gpt consumer."""
    consumer = consumers.VideoGPTBotHttpConsumer()
    consumer.scope = {
        "user": async_user,
        "cookies": {AI_SESSION_COOKIE_KEY: "test_session_key"},
        "session": django_session,
    }
    consumer.channel_name = "test_video_gpt_channel"
    return consumer


@pytest.fixture
def test_session_key():
    """Return a unique test session key."""
    from uuid import uuid4

    return f"test_session_{uuid4().hex}"


@pytest.fixture(autouse=True)
def mock_chatbot_completion(mocker):
    """Mock chatbot completion for testing."""
    response = ["test", "response"]
    return mocker.patch(
        "ai_chatbots.chatbots.SyllabusBot.get_completion",
        return_value=mocker.Mock(
            __aiter__=mocker.Mock(return_value=MockAsyncIterator(response))
        ),
    )


@pytest.fixture
def anonymous_consumer_setup(mocker):
    """Create a consumer setup for anonymous user testing."""

    from django.contrib.auth.models import AnonymousUser

    def create_consumer(session_key, cookies=None):
        consumer = consumers.SyllabusBotHttpConsumer()
        mock_session = mocker.Mock()
        mock_session.session_key = session_key
        mock_session.save = mocker.Mock()

        consumer.scope = {
            "user": AnonymousUser(),
            "cookies": cookies or {},
            "session": mock_session,
        }
        consumer.channel_name = "test_channel"
        return consumer

    return create_consumer


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
    mock_http_consumer_send.send_headers.assert_called_once()
    assert recommendation_consumer.bot.temperature == (
        temperature if temperature else settings.AI_DEFAULT_TEMPERATURE
    )
    assert recommendation_consumer.bot.model == (
        model if model else settings.AI_DEFAULT_RECOMMENDATION_MODEL
    )
    assert recommendation_consumer.bot.instructions == (
        instructions if instructions else prompts.PROMPT_RECOMMENDATION
    )
    default_state = {"search_url": [settings.AI_MIT_SEARCH_URL]}
    mock_completion.assert_called_once_with(message, extra_state=default_state)
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
    unique_thread_id = uuid4().hex

    session = await sync_to_async(UserChatSessionFactory.create)(
        thread_id=unique_thread_id,
        user=(None if is_anon else async_user),
        agent=recommendation_consumer.ROOM_NAME,
        dj_session_key="test_session_key",
    )
    thread_id_bytes = urlsafe_base64_encode(force_bytes(session.thread_id))
    recommendation_consumer.scope["cookies"] = {
        bot_cookie: thread_id_bytes,
        AI_SESSION_COOKIE_KEY: "test_session_key",
    }
    recommendation_consumer.scope["user"] = AnonymousUser() if is_anon else async_user
    await recommendation_consumer.handle(
        json.dumps({"clear_history": clear_history, "message": "hello"})
    )
    args = mock_http_consumer_send.send_headers.call_args_list
    headers = args[0][-1]["headers"]

    # Find the specific cookie by name rather than relying on position
    target_cookie = None
    for header_name, header_value in headers:
        if header_name == b"Set-Cookie":
            cookie_str = header_value.decode()
            if cookie_str.startswith(f"{bot_cookie}="):
                target_cookie = cookie_str
                break

    assert target_cookie is not None, f"Cookie {bot_cookie} not found in headers"

    if cookie_name == AI_THREAD_COOKIE_KEY and not is_anon:
        # old thread id should not be present at all
        assert (thread_id_bytes in target_cookie) != clear_history
    elif is_anon and cookie_name == AI_THREADS_ANONYMOUS_COOKIE_KEY:
        # old thread id should be present in the anon cookie
        assert (thread_id_bytes in target_cookie) != clear_history
    else:
        # anon thread_ids should have been cleared from cookie
        assert target_cookie == f"{bot_cookie}=;Path=/;"


async def test_http_request_complete_body(mocker):
    """Test the http request function with a complete message body"""
    mock_handle = mocker.patch(
        "ai_chatbots.consumers.RecommendationBotHttpConsumer.handle"
    )
    consumer = consumers.RecommendationBotHttpConsumer()
    msg = {"body": b"test message without chunks"}
    with pytest.raises(consumers.StopConsumer):
        await consumer.http_request(msg)
    mock_handle.assert_called_once_with(b"test message without chunks")


async def test_http_request_chunked_body(mocker):
    """Test the http request function with a chunked message body"""
    mock_handle = mocker.patch(
        "ai_chatbots.consumers.RecommendationBotHttpConsumer.handle"
    )
    consumer = consumers.RecommendationBotHttpConsumer()

    # Test chunked message body with multiple chunks
    first_chunk = {"body": b"test", "more_body": True}
    second_chunk = {"body": b" message", "more_body": True}
    third_chunk = {"body": b" with chunks", "more_body": False}

    await consumer.http_request(first_chunk)
    await consumer.http_request(second_chunk)

    with pytest.raises(consumers.StopConsumer):
        await consumer.http_request(third_chunk)

    # Verify handle was called with the complete body from all chunks
    mock_handle.assert_called_once_with(b"test message with chunks")


async def test_http_request_error(mocker):
    """Test that exceptions in the http_request function are logged."""
    mock_log = mocker.patch("ai_chatbots.consumers.log.exception")
    mock_handle = mocker.patch(
        "ai_chatbots.consumers.RecommendationBotHttpConsumer.handle",
        side_effect=Exception("Test exception"),
    )
    consumer = consumers.RecommendationBotHttpConsumer()
    with pytest.raises(consumers.StopConsumer):
        await consumer.http_request({"body": b"test"})
    mock_handle.assert_called_once()
    mock_log.assert_called_once_with("Error in handling consumer http_request")


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
    chatbot = await sync_to_async(syllabus_consumer.create_chatbot)(
        serializer, mocker.Mock()
    )
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
def test_syllabus_process_extra_state(syllabus_consumer, request_params):
    """Test that the process_extra_state function returns the expected values."""

    assert syllabus_consumer.process_extra_state(request_params) == {
        "course_id": [request_params.get("course_id")],
        "collection_name": [request_params.get("collection_name", None)],
        "exclude_canvas": ["True"],
    }


def test_canvas_syllabus_process_extra_state(canvas_syllabus_consumer):
    """Test that the canvas syllabus process_extra_state function returns False for exclude_canvas."""
    assert canvas_syllabus_consumer.process_extra_state(
        {"message": "hello", "course_id": "MITx+6.00.1x"}
    ) == {
        "course_id": ["MITx+6.00.1x"],
        "collection_name": [None],
        "exclude_canvas": ["False"],
    }


async def test_canvas_syllabus_create_chatbot(mocker, canvas_syllabus_consumer):
    """The correct chatbot class should be assigned to self.chatbot"""
    serializer = consumers.SyllabusChatRequestSerializer(
        data={
            "message": "test",
            "course_id": "MITx+6.00.1x",
            "model": "gpt-3.5-turbo",
        }
    )
    serializer.is_valid()
    await canvas_syllabus_consumer.prepare_response(serializer)
    bot = await sync_to_async(canvas_syllabus_consumer.create_chatbot)(
        serializer, mocker.Mock()
    )
    assert bot.__class__ == consumers.CanvasSyllabusBot


@pytest.mark.parametrize(
    ("user_cookie", "anon_cookie", "is_anon"),
    [
        ("1234", "", False),  # User is authenticated
        ("1234", "", True),  # User recently logged out
        ("", "4567", False),  # User just authenticated
        ("", "4567", True),  # User is anonymous
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

    encoded_user_cookie = urlsafe_base64_encode(force_bytes(user_cookie))
    encoded_anon_cookie = urlsafe_base64_encode(force_bytes(anon_cookie))

    syllabus_consumer.scope["cookies"] = {
        user_cookie_name: encoded_user_cookie,
        anon_cookie_name: encoded_anon_cookie,
        AI_SESSION_COOKIE_KEY: "test_session_key",
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
            dj_session_key="test_session_key",
        )
    elif anon_cookie:
        for thread in anon_cookie.split(","):
            # Should not yet be associated with user
            await UserChatSession.objects.acreate(
                thread_id=thread,
                user=None,
                agent=SyllabusBot.__name__,
                dj_session_key="test_session_key",
            )

    thread_id, cookies = await syllabus_consumer.assign_thread_cookies(user)

    encoded_thread_id = urlsafe_base64_encode(force_bytes(thread_id))
    if anon_cookie:  # User is or just was anonymous
        assert thread_id == anon_cookie
        if not is_anon:  # User just authenticated, associate older sessions with user
            # Clear out anon cookie thread ids and set current thread id to auth user cookie
            assert (
                str(cookies[1])
                == f"{anon_cookie_name}=;Path=/;Max-Age={settings.AI_CHATBOTS_COOKIE_MAX_AGE};"
            )
            assert (
                str(cookies[0])
                == f"{user_cookie_name}={encoded_thread_id};Path=/;Max-Age={settings.AI_CHATBOTS_COOKIE_MAX_AGE};"
            )
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
        assert encoded_thread_id in str(cookies[(1 if is_anon else 0)])
    else:  # User is either not anonymous or has no anon cookies
        assert (thread_id == user_cookie) is (user_cookie != "")
        if (
            is_anon
        ):  # User must have logged out, so current thread id should be in anon cookie
            assert (
                str(cookies[1])
                == f"{anon_cookie_name}={encoded_thread_id};Path=/;Max-Age={settings.AI_CHATBOTS_COOKIE_MAX_AGE};"
            )
            assert (
                str(cookies[0])
                == f"{user_cookie_name}=;Path=/;Max-Age={settings.AI_CHATBOTS_COOKIE_MAX_AGE};"
            )


async def test_assign_thread_cookie_changed_account(syllabus_consumer, async_user):
    """Test the cookie handling if a user logs in under a different account."""
    old_thread_id = uuid4().hex
    user_cookie_name = f"{syllabus_consumer.ROOM_NAME}_{AI_THREAD_COOKIE_KEY}"

    encoded_user_cookie = urlsafe_base64_encode(force_bytes(old_thread_id))

    syllabus_consumer.scope["cookies"] = {
        user_cookie_name: encoded_user_cookie,
        AI_SESSION_COOKIE_KEY: "test_session_key",
    }

    syllabus_consumer.scope["user"] = async_user

    # Clear out any old saved sessions
    await UserChatSession.objects.all().adelete()

    # Associate old thread id with a different user
    await UserChatSession.objects.acreate(
        thread_id=old_thread_id,
        user=await sync_to_async(UserFactory.create)(),
        agent=SyllabusBot.__name__,
        dj_session_key="different_session_key",
    )

    thread_id, cookies = await syllabus_consumer.assign_thread_cookies(async_user)
    encoded_thread_id = urlsafe_base64_encode(force_bytes(thread_id))

    # Should have gotten assigned a new thread id
    assert thread_id != old_thread_id
    assert cookies[0].value == encoded_thread_id


@pytest.mark.parametrize(
    ("is_anon", "is_authorized", "same_key"),
    [
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, False),
    ],
)
async def test_assign_thread_cookie_passed_thread(
    syllabus_consumer, async_user, is_anon, is_authorized, same_key
):
    """
    The thread_id in the request should override any cookies for an authorized user.
    """
    original_thread_id = uuid4().hex
    encoded_original_id = urlsafe_base64_encode(force_bytes(original_thread_id))
    requested_thread_id = uuid4().hex
    encoded_requested_id = urlsafe_base64_encode(force_bytes(requested_thread_id))
    session_key = "abc123"

    user_cookie_name = f"{syllabus_consumer.ROOM_NAME}_{AI_THREAD_COOKIE_KEY}"
    syllabus_consumer.scope["cookies"] = {
        user_cookie_name: encoded_original_id,
        AI_SESSION_COOKIE_KEY: session_key if same_key else "other_key",
    }

    user = AnonymousUser() if is_anon else async_user
    await sync_to_async(UserChatSessionFactory.create)(
        thread_id=requested_thread_id,
        user=(async_user if is_authorized else None),
        dj_session_key=session_key,
    )
    assigned_thread_id, cookies = await syllabus_consumer.assign_thread_cookies(
        thread_id=requested_thread_id, user=user
    )
    if is_anon and same_key:
        assert await UserChatSession.objects.filter(
            user=None, thread_id=requested_thread_id, dj_session_key=session_key
        ).aexists()
    assert assigned_thread_id == (
        requested_thread_id
        if is_authorized or (is_anon and same_key)
        else original_thread_id
    )
    assert (encoded_requested_id == cookies[0].value) is (is_authorized and not is_anon)
    assert (encoded_requested_id == cookies[1].value) is (is_anon and same_key)


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
            "exclude_canvas": ["True"],
        },
    )
    assert await UserChatSession.objects.filter(
        thread_id=syllabus_consumer.thread_id,
        user=syllabus_consumer.scope["user"],
        title=payload["message"],
        agent=SyllabusBot.__name__,
    ).aexists()


@pytest.mark.parametrize(
    ("error_class", "expected_status", "headers_sent"),
    [
        (ValidationError, 400, True),
        (json.JSONDecodeError, 400, False),
        (Exception, 500, True),
        (ValueError, 500, False),
    ],
)
async def test_handle_errors(
    mocker, syllabus_consumer, error_class, expected_status, headers_sent
):
    """Test that the handle function sends the correct error response."""
    mock_send = mocker.patch(
        "ai_chatbots.consumers.AsyncHttpConsumer.send", new_callable=AsyncMock
    )

    # Raise an error in the right place depending on headers_sent
    error_message = "Major malfunction"
    error = (
        error_class(error_message, doc="doc", pos=0)
        if error_class == json.JSONDecodeError
        else error_class(error_message)
    )
    if headers_sent:
        mocker.patch(
            "ai_chatbots.consumers.SyllabusBot.get_completion",
            side_effect=error,
        )
    else:
        mocker.patch(
            "ai_chatbots.consumers.SyllabusBotHttpConsumer.create_chatbot",
            side_effect=error,
        )

    expected_msg = json.dumps({"error": {"message": str(error)}})

    await syllabus_consumer.handle('{"message": "hello", "course_id": "MITx+6.00.1x"}')
    # If error occurs after headers are sent, the status will be 200, that ship has sailed
    if headers_sent:
        mock_send.assert_any_call(
            {
                "type": "http.response.body",
                "body": f"<!-- {expected_msg} -->".encode(),
                "more_body": True,
            }
        )
    else:
        mock_send.assert_any_call(
            {
                "type": "http.response.start",
                "status": expected_status,
                "headers": [
                    (b"Cache-Control", b"no-cache"),
                    (
                        b"Content-Type",
                        b"application/json",
                    ),
                    (b"Connection", b"close"),
                ],
            }
        )
        mock_send.assert_any_call(
            {
                "type": "http.response.body",
                "body": expected_msg.encode(),
                "more_body": True,
            }
        )


@pytest.mark.parametrize(
    ("wait_time", "formatted_time"),
    [
        (35, "35s"),
        (68, "1m 8s"),
        (3600, "1h"),
        (86400, "1d"),
        (90061, "1d 1h"),
        (360000, "4d 4h"),
    ],
)
async def test_rate_limit_message(mocker, wait_time, formatted_time, django_session):
    """Test that a user gets a rate limit message if they are rate limited."""
    mocker.patch(
        "ai_chatbots.consumers.SyllabusBotHttpConsumer.check_throttles",
        side_effect=AsyncThrottled(wait_time),
    )
    mocker.patch("ai_chatbots.consumers.AsyncHttpConsumer.send", new_callable=AsyncMock)
    mock_send_chunk = mocker.patch(
        "ai_chatbots.consumers.SyllabusBotHttpConsumer.send_chunk"
    )
    consumer = consumers.SyllabusBotHttpConsumer()
    consumer.scope = {
        "user": AnonymousUser(),
        "session": django_session,
        "cookies": {AI_SESSION_COOKIE_KEY: "test_session_key"},
    }
    await consumer.handle('{"message": "hello", "course_id": "MITx+6.00.1x"}')
    mock_send_chunk.assert_any_call(
        f"You have reached the maximum number of chat requests.\
                \nPlease try again in {formatted_time}."
    )


async def test_tutor_agent_handle(
    mocker,
    mock_http_consumer_send,
    tutor_consumer,
):
    """Test the receive function of the recommendation agent."""
    response = SystemMessageFactory.create()
    user = tutor_consumer.scope["user"]
    user.is_superuser = True
    mocker.patch(
        "ai_chatbots.chatbots.get_problem_from_edx_block",
        new_callable=AsyncMock,
        return_value=("problem_xml", "problem_set_xml"),
    )
    mock_completion = mocker.patch(
        "ai_chatbots.chatbots.TutorBot.get_completion",
        return_value=mocker.Mock(
            __aiter__=mocker.Mock(
                return_value=MockAsyncIterator(list(response.content.split(" ")))
            )
        ),
    )
    message = "What should i try next?"
    data = {
        "message": message,
        "edx_module_id": "block1",
        "block_siblings": ["block1", "block2"],
    }

    await tutor_consumer.handle(json.dumps(data))

    mock_http_consumer_send.send_headers.assert_called_once()

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


async def canvas_test_tutor_agent_handle(
    mocker,
    mock_http_consumer_send,
    canvas_tutor_consumer,
):
    """Test the receive function of the recommendation agent."""
    response = SystemMessageFactory.create()
    user = canvas_tutor_consumer.scope["user"]
    user.is_superuser = True
    mocker.patch(
        "ai_chatbots.chatbots.get_canvas_problem_set",
        new_callable=AsyncMock,
        return_value="problem_set",
    )
    mock_completion = mocker.patch(
        "ai_chatbots.chatbots.TutorBot.get_completion",
        return_value=mocker.Mock(
            __aiter__=mocker.Mock(
                return_value=MockAsyncIterator(list(response.content.split(" ")))
            )
        ),
    )
    message = "What should i try next?"
    data = {
        "message": message,
        "run_readable_id": "run1",
        "problem_set_title": "Problem Set 1",
    }

    await canvas_tutor_consumer.handle(json.dumps(data))

    mock_http_consumer_send.send_headers.assert_called_once()

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

    user_chat = await UserChatSession.objects.aget(
        thread_id=canvas_tutor_consumer.thread_id,
        user=canvas_tutor_consumer.scope["user"],
        title=data["message"],
        agent="TutorBot",
    )

    assert user_chat is not None

    assert user_chat.object_id == "run1 - Problem Set 1"


async def test_video_gpt_create_chatbot(
    mocker, mock_http_consumer_send, video_gpt_consumer, async_user
):
    """VideoGPTBotHttpConsumer create_chatbot function should return VideoGPTBot."""
    serializer = consumers.VideoGPTRequestSerializer(
        data={
            "message": "What is this video about?",
            "transcript_asset_id": "asset-v1:xPRO+LASERxE3+R15+type@asset+block@469c03c4-581a-4687-a9ca-7a1c4047832d-en",
            "temperature": 0.7,
            "model": "gpt-3.5-turbo",
        }
    )
    serializer.is_valid(raise_exception=True)
    await video_gpt_consumer.prepare_response(serializer)
    chatbot = await sync_to_async(video_gpt_consumer.create_chatbot)(
        serializer, mocker.Mock()
    )
    assert isinstance(chatbot, VideoGPTBot)
    assert chatbot.user_id == async_user.global_id
    assert chatbot.temperature == 0.7
    assert chatbot.model == "gpt-3.5-turbo"


async def test_video_agent_consumer_handle(
    mocker, mock_http_consumer_send, video_gpt_consumer
):
    """Test the handle function of the video gpt consumer."""
    response = SystemMessageFactory.create().content.split(" ")
    mock_completion = mocker.patch(
        "ai_chatbots.chatbots.VideoGPTBot.get_completion",
        return_value=mocker.Mock(
            __aiter__=mocker.Mock(return_value=MockAsyncIterator(list(response)))
        ),
    )
    payload = {
        "message": "what is this video about?",
        "transcript_asset_id": "asset-v1:xPRO+LASERxE3+R15+type@asset+block@469c03c4-581a-4687-a9ca-7a1c4047832d-en",
    }
    await video_gpt_consumer.handle(json.dumps(payload))
    mock_completion.assert_called_once_with(
        payload["message"],
        extra_state={
            "transcript_asset_id": [payload["transcript_asset_id"]],
        },
    )
    assert await UserChatSession.objects.filter(
        thread_id=video_gpt_consumer.thread_id,
        user=video_gpt_consumer.scope["user"],
        title=payload["message"],
        agent=VideoGPTBot.__name__,
    ).aexists()


async def test_anonymous_user_session_cookie_assignment(
    mocker, mock_http_consumer_send, test_session_key
):
    """Test that anonymous users get assigned a non-blank session cookie and UserChatSession is created."""
    from django.contrib.auth.models import AnonymousUser

    from ai_chatbots.constants import AI_SESSION_COOKIE_KEY

    # Create a consumer with an anonymous user and no session cookie
    consumer = consumers.SyllabusBotHttpConsumer()
    mock_session = mocker.Mock()
    mock_session.session_key = test_session_key
    mock_session.save = mocker.Mock()

    consumer.scope = {
        "user": AnonymousUser(),
        "cookies": {},  # No session cookie initially
        "session": mock_session,
    }
    consumer.channel_name = "test_channel"

    # Clear any existing sessions
    await UserChatSession.objects.all().adelete()

    payload = {
        "message": "test message",
        "course_id": "MITx+6.00.1x",
    }

    await consumer.handle(json.dumps(payload))

    # Verify the session cookie was assigned
    headers = mock_http_consumer_send.send_headers.call_args[1]["headers"]
    session_cookie = None
    for header in headers:
        if header[0] == b"Set-Cookie" and AI_SESSION_COOKIE_KEY.encode() in header[1]:
            session_cookie = header[1].decode()
            break

    assert session_cookie is not None
    assert f"{AI_SESSION_COOKIE_KEY}=" in session_cookie
    # Extract the cookie value (everything after the = and before any ;)
    cookie_value = session_cookie.split(f"{AI_SESSION_COOKIE_KEY}=")[1].split(";")[0]
    assert cookie_value != ""
    assert cookie_value is not None

    # Verify UserChatSession was created with correct dj_session_key
    session = await UserChatSession.objects.aget(
        thread_id=consumer.thread_id,
        user=None,  # Should be None for anonymous user
        agent="SyllabusBot",
    )
    assert session.dj_session_key == cookie_value
    assert session.dj_session_key != ""


async def test_anonymous_user_login_session_association(  # noqa: PLR0913
    mocker,
    mock_http_consumer_send,
    async_user,
    anonymous_consumer_setup,
    test_session_key,
    client,
):
    """
    Test that when an anonymous user logs in, their sessions are associated
    with their user account via Consumer.handle flow.
    """
    from ai_chatbots.constants import (
        AI_SESSION_COOKIE_KEY,
        AI_THREADS_ANONYMOUS_COOKIE_KEY,
    )

    # Clear any existing sessions
    await UserChatSession.objects.all().adelete()

    # Create multiple anonymous sessions with the same session key
    session_data = []

    for i in range(3):
        anon_consumer = anonymous_consumer_setup(test_session_key)
        payload = {
            "message": f"Anonymous question {i+1}",
            "course_id": "MITx+6.00.1x",
        }
        await anon_consumer.handle(json.dumps(payload))
        session_data.append(
            {"thread_id": anon_consumer.thread_id, "message": payload["message"]}
        )

    # Create 1 anonymous session with a different session key
    different_session_key = f"different_session_{test_session_key}_different"
    different_anon_consumer = anonymous_consumer_setup(different_session_key)
    different_payload = {
        "message": "Different session anonymous question",
        "course_id": "MITx+6.00.1x",
    }
    await different_anon_consumer.handle(json.dumps(different_payload))
    different_thread_id = different_anon_consumer.thread_id

    all_anon_sessions = await sync_to_async(list)(
        UserChatSession.objects.filter(user=None)
    )
    assert len(all_anon_sessions) == 4

    test_session_anon_sessions = await sync_to_async(list)(
        UserChatSession.objects.filter(dj_session_key=test_session_key, user=None)
    )
    assert len(test_session_anon_sessions) == 3

    different_session_anon_sessions = await sync_to_async(list)(
        UserChatSession.objects.filter(dj_session_key=different_session_key, user=None)
    )
    assert len(different_session_anon_sessions) == 1

    # Log in user with anon cookie containing some thread IDs
    logged_in_consumer = consumers.SyllabusBotHttpConsumer()
    mock_session = mocker.Mock()
    mock_session.session_key = "new_session_after_login"
    mock_session.save = mocker.Mock()

    # Set up anon cookie to trigger the association logic
    anon_cookie_name = (
        f"{logged_in_consumer.ROOM_NAME}_{AI_THREADS_ANONYMOUS_COOKIE_KEY}"
    )
    anon_cookie_value = f"{session_data[0]['thread_id']},{session_data[1]['thread_id']}"
    encoded_anon_cookie = urlsafe_base64_encode(force_bytes(anon_cookie_value))

    await sync_to_async(client.force_login)(async_user)
    logged_in_consumer.scope = {
        "user": async_user,  # Now logged in
        "cookies": {
            AI_SESSION_COOKIE_KEY: test_session_key,
            anon_cookie_name: encoded_anon_cookie,
        },
        "session": mock_session,
    }
    logged_in_consumer.channel_name = "test_channel"

    await logged_in_consumer.handle(
        json.dumps(
            {
                "message": "First message after login",
                "course_id": "MITx+6.00.1x",
            }
        )
    )

    # All anonymous sessions with the same session key associated with the user
    associated_sessions = await sync_to_async(list)(
        UserChatSession.objects.filter(dj_session_key=test_session_key, user=async_user)
    )
    assert len(associated_sessions) == 3
    original_thread_ids = [s["thread_id"] for s in session_data]
    associated_thread_ids = [session.thread_id for session in associated_sessions]
    for thread_id in original_thread_ids:
        assert thread_id in associated_thread_ids

    # No anonymous sessions remain with this session key
    remaining_anon_sessions = await sync_to_async(list)(
        UserChatSession.objects.filter(dj_session_key=test_session_key, user=None)
    )
    assert len(remaining_anon_sessions) == 0

    # Sessions with different session keys are not associated
    different_session_sessions = await sync_to_async(list)(
        UserChatSession.objects.filter(dj_session_key=different_session_key)
    )
    assert len(different_session_sessions) == 1
    assert different_session_sessions[0].user is None  # Should still be anonymous
    assert different_session_sessions[0].thread_id == different_thread_id


@pytest.mark.parametrize(
    ("dj_session_key", "should_be_updated"),
    [
        ("", False),  # empty string session key should be excluded
        ("user_session_key_value", True),  # valid session key should be updated
    ],
)
async def test_assign_thread_cookies_session_key_filtering(
    syllabus_consumer, async_user, dj_session_key, should_be_updated
):
    """
    Test that anon user login associations exclude sessions with empty dj_session_key.

    """
    test_session_key = "user_session_key_value"
    anon_thread_id = uuid4().hex
    anon_cookie_name = (
        f"{syllabus_consumer.ROOM_NAME}_{AI_THREADS_ANONYMOUS_COOKIE_KEY}"
    )
    encoded_anon_cookie = urlsafe_base64_encode(force_bytes(anon_thread_id))

    syllabus_consumer.scope["cookies"] = {
        anon_cookie_name: encoded_anon_cookie,
        AI_SESSION_COOKIE_KEY: test_session_key,
    }
    syllabus_consumer.scope["user"] = async_user
    syllabus_consumer.session_key = test_session_key

    await UserChatSession.objects.all().adelete()
    await UserChatSession.objects.acreate(
        thread_id=anon_thread_id,
        user=None,
        agent=SyllabusBot.__name__,
        dj_session_key=dj_session_key,
    )

    await syllabus_consumer.assign_thread_cookies(async_user)

    if should_be_updated:
        assert await UserChatSession.objects.filter(
            thread_id=anon_thread_id, user=async_user
        ).aexists()
    else:
        assert await UserChatSession.objects.filter(
            thread_id=anon_thread_id, user=None
        ).aexists()


@pytest.mark.asyncio
async def test_disconnect_closes_litellm_clients(mocker, recommendation_consumer):
    """Test that disconnect properly closes LiteLLM async clients."""
    # Mock litellm.close_litellm_async_clients
    mock_close = mocker.patch(
        "ai_chatbots.consumers.litellm.close_litellm_async_clients"
    )
    mock_close.return_value = AsyncMock()

    # Mock channel layer
    recommendation_consumer.channel_layer = mocker.Mock()
    recommendation_consumer.channel_layer.group_discard = AsyncMock()
    recommendation_consumer.room_group_name = "test_room"

    # Call disconnect
    await recommendation_consumer.disconnect()

    # Verify litellm cleanup was called
    mock_close.assert_called_once()


@pytest.mark.asyncio
async def test_disconnect_handles_litellm_exception(mocker, recommendation_consumer):
    """Test that disconnect handles exceptions from LiteLLM cleanup gracefully."""
    # Mock litellm.close_litellm_async_clients to raise an exception
    mock_close = mocker.patch(
        "ai_chatbots.consumers.litellm.close_litellm_async_clients",
        side_effect=Exception("Test exception"),
    )

    # Mock channel layer
    recommendation_consumer.channel_layer = mocker.Mock()
    recommendation_consumer.channel_layer.group_discard = AsyncMock()
    recommendation_consumer.room_group_name = "test_room"

    # Call disconnect - should not raise exception
    await recommendation_consumer.disconnect()

    # Verify litellm cleanup was attempted
    mock_close.assert_called_once()
    # Verify channel cleanup still happened despite exception
    recommendation_consumer.channel_layer.group_discard.assert_called_once()


@pytest.mark.asyncio
async def test_disconnect_without_channel_layer(mocker, recommendation_consumer):
    """Test disconnect works when channel_layer is not set."""
    # Mock litellm
    mock_close = mocker.patch(
        "ai_chatbots.consumers.litellm.close_litellm_async_clients"
    )
    mock_close.return_value = AsyncMock()

    # Don't set channel_layer
    if hasattr(recommendation_consumer, "channel_layer"):
        delattr(recommendation_consumer, "channel_layer")

    # Call disconnect - should not raise exception
    await recommendation_consumer.disconnect()

    # Verify litellm cleanup was called
    mock_close.assert_called_once()
