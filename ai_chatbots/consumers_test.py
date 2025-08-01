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
from ai_chatbots.constants import AI_THREAD_COOKIE_KEY, AI_THREADS_ANONYMOUS_COOKIE_KEY
from ai_chatbots.factories import SystemMessageFactory, UserChatSessionFactory
from ai_chatbots.models import UserChatSession
from main.exceptions import AsyncThrottled
from main.factories import UserFactory

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


@pytest.fixture
def tutor_consumer(async_user):
    """Return a tutor consumer."""
    consumer = consumers.TutorBotHttpConsumer()
    consumer.scope = {"user": async_user, "cookies": {}, "session": None}
    consumer.channel_name = "test_tutor_channel"
    return consumer


@pytest.fixture
def canvas_tutor_consumer(async_user):
    """Return a canvas tutor consumer."""
    consumer = consumers.CanvasTutorBotHttpConsumer()
    consumer.scope = {"user": async_user, "cookies": {}, "session": None}
    consumer.channel_name = "test_tutor_channel"
    return consumer


@pytest.fixture
def video_gpt_consumer(async_user):
    """Return a video gpt consumer."""
    consumer = consumers.VideoGPTBotHttpConsumer()
    consumer.scope = {"user": async_user, "cookies": {}, "session": None}
    consumer.channel_name = "test_video_gpt_channel"
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
    mock_http_consumer_send.send_headers.assert_called_once()
    assert recommendation_consumer.bot.llm.temperature == (
        temperature if temperature else settings.AI_DEFAULT_TEMPERATURE
    )
    assert recommendation_consumer.bot.llm.model == (
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
    latest_thread_id = urlsafe_base64_encode(force_bytes("1234"))
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
        assert (latest_thread_id in cookie_args) != clear_history
    else:
        # anon thread_ids should have been cleared from cookie
        cookie_args = str(args[0][-1]["headers"][-2][1])
        assert cookie_args == f"{bot_cookie}=;Path=/;"


async def test_http_request(mocker):
    """Test the http request function of the AsyncHttpConsumer"""
    msg = {"body": "test"}
    mock_handle = mocker.patch(
        "ai_chatbots.consumers.RecommendationBotHttpConsumer.handle"
    )
    consumer = consumers.RecommendationBotHttpConsumer()
    await consumer.http_request(msg)
    mock_handle.assert_called_once_with(msg["body"])


async def test_http_request_error(mocker):
    """Test that exceptions in the http_request function are logged."""
    mock_log = mocker.patch("ai_chatbots.consumers.log.exception")
    mock_handle = mocker.patch(
        "ai_chatbots.consumers.RecommendationBotHttpConsumer.handle",
        side_effect=Exception("Test exception"),
    )
    consumer = consumers.RecommendationBotHttpConsumer()
    await consumer.http_request({"body": "test"})
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
def test_syllabus_process_extra_state(syllabus_consumer, request_params):
    """Test that the process_extra_state function returns the expected values."""

    assert syllabus_consumer.process_extra_state(request_params) == {
        "course_id": [request_params.get("course_id")],
        "collection_name": [request_params.get("collection_name", None)],
        "exclude_canvas": ["True"],
    }


def test_canvas_process_extra_state(syllabus_consumer, async_user):
    """Test that the canvas syllabus process_extra_state function returns False for exclude_canvas."""
    consumer = consumers.CanvasSyllabusBotHttpConsumer()
    consumer.scope = {"user": async_user, "cookies": {}, "session": None}
    consumer.channel_name = "test_syllabus_channel"

    assert consumer.process_extra_state(
        {"message": "hello", "course_id": "MITx+6.00.1x"}
    ) == {
        "course_id": ["MITx+6.00.1x"],
        "collection_name": [None],
        "exclude_canvas": ["False"],
    }


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

    syllabus_consumer.scope["cookies"] = {user_cookie_name: encoded_user_cookie}

    syllabus_consumer.scope["user"] = async_user

    # Clear out any old saved sessions
    await UserChatSession.objects.all().adelete()

    # Associate old thread id with a different user
    await UserChatSession.objects.acreate(
        thread_id=old_thread_id,
        user=await sync_to_async(UserFactory.create)(),
        agent=SyllabusBot.__name__,
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
    }
    syllabus_consumer.session_key = session_key if same_key else "other_key"

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
async def test_rate_limit_message(mocker, wait_time, formatted_time):
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
    consumer.scope = {"user": AnonymousUser()}
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
    chatbot = video_gpt_consumer.create_chatbot(serializer, mocker.Mock())
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
