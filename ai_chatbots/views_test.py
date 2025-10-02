"""Tests for ai_chatbots views"""

import json
from random import randint
from types import SimpleNamespace
from uuid import uuid4

import pytest
from django.test import RequestFactory
from django.urls import reverse
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode
from rest_framework import status
from rest_framework.exceptions import ErrorDetail
from rest_framework.test import APIClient

from ai_chatbots.constants import AI_SESSION_COOKIE_KEY, AI_THREADS_ANONYMOUS_COOKIE_KEY
from ai_chatbots.factories import (
    ChatResponseRatingFactory,
    CheckpointFactory,
    LLMModelFactory,
    UserChatSessionFactory,
)
from ai_chatbots.models import ChatResponseRating, LLMModel
from ai_chatbots.prompts import CHATBOT_PROMPT_MAPPING, parse_prompt
from ai_chatbots.serializers import ChatMessageSerializer, UserChatSessionSerializer
from ai_chatbots.views import get_transcript_block_id
from main.factories import UserFactory
from main.test_utils import assert_json_equal

pytestmark = pytest.mark.django_db


@pytest.fixture
def test_users():
    """Create several test users"""
    return UserFactory.create_batch(2)


@pytest.fixture
def test_sessions(test_users):
    """Create several test sessions for different users"""
    return [
        session
        for user in test_users
        for session in UserChatSessionFactory.create_batch(2, user=user)
    ]


@pytest.fixture
def test_session_w_messages():
    """Create a session containing a few messages"""
    chat_session = UserChatSessionFactory()
    return SimpleNamespace(
        session=chat_session,
        messages=[
            CheckpointFactory.create(thread_id=chat_session.thread_id, is_human=True),
            CheckpointFactory.create(thread_id=chat_session.thread_id, is_tool=True),
            CheckpointFactory.create(thread_id=chat_session.thread_id, is_agent=True),
        ],
    )


@pytest.fixture
def test_anonymous_session():
    """Create a anonymous session containing a few messages"""
    chat_session = UserChatSessionFactory(user=None, dj_session_key=uuid4().hex)
    return SimpleNamespace(
        session=chat_session,
        messages=[
            CheckpointFactory.create(thread_id=chat_session.thread_id, is_human=True),
            CheckpointFactory.create(thread_id=chat_session.thread_id, is_tool=True),
            CheckpointFactory.create(thread_id=chat_session.thread_id, is_agent=True),
        ],
    )


@pytest.fixture
def factory():
    return RequestFactory()


@pytest.fixture
def user_session():
    return UserChatSessionFactory()


@pytest.fixture
def agent_checkpoint(user_session):
    return CheckpointFactory(
        session=user_session, thread_id=user_session.thread_id, is_agent=True
    )


@pytest.fixture
def human_checkpoint(user_session):
    return CheckpointFactory(
        session=user_session, thread_id=user_session.thread_id, is_human=True
    )


def test_user_chat_sessions_view_list(client, test_sessions, test_users):
    """Test UserChatSessionsViewSet list results and filtering by user"""
    for user in test_users:
        client.force_login(user)
        response = client.get(reverse("ai:v0:chat_sessions-list"))
        assert response.status_code == 200
        for session in test_sessions:
            assert session.user is not None
            assert session.created_on is not None
        assert_json_equal(
            response.json()["results"],
            [
                UserChatSessionSerializer(instance=session).data
                for session in sorted(
                    test_sessions, key=lambda session: session.created_on, reverse=True
                )
                if session.user and session.user == user
            ],
        )


def test_user_chat_sessions_view_list_403(client):
    """Anon users should not have access to chat sessions"""
    response = client.get(reverse("ai:v0:chat_sessions-list"))
    assert response.status_code == 403


def test_user_chat_sessions_view_detail(client, test_sessions):
    """Test UserChatSessionsViewSet detail results"""
    session = test_sessions[0]
    client.force_login(session.user)
    response = client.get(
        reverse("ai:v0:chat_sessions-detail", args=[session.thread_id])
    )
    assert response.status_code == 200
    assert session.user == session.user
    assert_json_equal(response.json(), UserChatSessionSerializer(instance=session).data)


def test_user_chat_sessions_view_detail_403_anon(client, test_sessions):
    """Anon users should not have access to chat session"""
    response = client.get(
        reverse("ai:v0:chat_sessions-detail", args=[test_sessions[0].thread_id])
    )
    assert response.status_code == 403


def test_user_chat_sessions_view_detail_404_other_user(client, test_sessions):
    """Users should not have access to other users' chat sessions"""
    other_user = UserFactory.create()
    client.force_login(other_user)
    response = client.get(
        reverse("ai:v0:chat_sessions-detail", args=[test_sessions[0].thread_id])
    )
    assert response.status_code == 404


def test_thread_messages_view(client, test_session_w_messages):
    """Test ChatMessageViewSet list results"""
    session = test_session_w_messages.session
    client.force_login(session.user)
    # Some arbitray messages from another thread/user
    CheckpointFactory.create_batch(3)
    response = client.get(
        reverse("ai:v0:chat_session_messages-list", args=[session.thread_id])
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    expected = sorted(
        [
            ChatMessageSerializer(instance=checkpoint).data
            for checkpoint in test_session_w_messages.messages
            if (
                checkpoint.metadata["writes"].get("agent")
                or checkpoint.metadata["writes"].get("__start__")
            )
        ],
        key=lambda message: message["step"],
    )
    for result in results:
        assert result["role"] in ["human", "agent"]
        assert result["content"] is not None
    assert_json_equal(response.json()["results"], expected)


def test_thread_messages_view_403(client, test_session_w_messages):
    """Test ChatMessageViewSet returns a 403 for a unauthorized user"""
    session = test_session_w_messages.session
    client.force_login(UserFactory.create())
    response = client.get(
        reverse("ai:v0:chat_session_messages-list", args=[session.thread_id])
    )
    assert response.status_code == 403


def test_rate_agent_message_success(client, user_session, agent_checkpoint):
    """Test successfully rating an agent message"""
    # Use direct ViewSet approach since URL routing has issues
    client.force_login(user_session.user)
    response = client.post(
        f"/api/v0/chat_sessions/{user_session.thread_id}/messages/{agent_checkpoint.id}/rate/",
        data={"rating": "like"},
        content_type="application/json",
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.data["rating"] == "like"

    # Verify rating was saved
    rating = ChatResponseRating.objects.get(checkpoint=agent_checkpoint)
    assert rating.rating == "like"


@pytest.mark.parametrize("has_object_id", [True, False])
def test_rate_agent_message_anon_success(client, test_anonymous_session, has_object_id):
    """Test successfully rating an agent message for an anonymous user"""
    if has_object_id:
        test_anonymous_session.session.object_id = uuid4().hex
        test_anonymous_session.session.save()
    cookie_value = urlsafe_base64_encode(
        force_bytes(
            f"{test_anonymous_session.session.thread_id}{f'|{test_anonymous_session.session.object_id}' if test_anonymous_session.session.object_id else ''}"
        )
    )
    response = client.post(
        f"/api/v0/chat_sessions/{test_anonymous_session.session.thread_id}/messages/{test_anonymous_session.messages[2].id}/rate/",
        data={"rating": "dislike", "rating_reason": "Partially inaccurate"},
        content_type="application/json",
        HTTP_COOKIE=f"{test_anonymous_session.session.agent}_{AI_THREADS_ANONYMOUS_COOKIE_KEY}={cookie_value}",
    )
    assert response.status_code == status.HTTP_200_OK
    assert response.data["rating"] == "dislike"
    assert response.data["rating_reason"] == "Partially inaccurate"

    # Verify rating was saved
    rating = ChatResponseRating.objects.get(
        checkpoint=test_anonymous_session.messages[2]
    )
    assert rating.rating == "dislike"
    assert rating.rating_reason == "Partially inaccurate"


def test_rate_agent_message_other_user_403_wrong_cookie(
    client, test_anonymous_session, agent_checkpoint
):
    """Test that rating an agent message fails for unauthorized users"""
    wrong_cookie_value = urlsafe_base64_encode(force_bytes(f"{uuid4().hex}|1000"))
    client.force_login(UserFactory.create())
    response = client.post(
        f"/api/v0/chat_sessions/{test_anonymous_session.session.thread_id}/messages/{test_anonymous_session.messages[2].id}/rate/",
        data={"rating": "like"},
        content_type="application/json",
        HTTP_COOKIE=f"{test_anonymous_session.session.agent}_{AI_THREADS_ANONYMOUS_COOKIE_KEY}={wrong_cookie_value}",
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_rate_agent_message_other_user_403_no_cookie(
    client, test_anonymous_session, agent_checkpoint
):
    """Test that rating an agent message fails for unauthorized users without any cookie"""
    client.force_login(UserFactory.create())
    response = client.post(
        f"/api/v0/chat_sessions/{test_anonymous_session.session.thread_id}/messages/{test_anonymous_session.messages[2].id}/rate/",
        data={"rating": "like"},
        content_type="application/json",
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_rate_agent_message_anon_403(client, test_anonymous_session):
    """Test that rating an agent message for another anonymous user fails"""
    response = client.post(
        f"/api/v0/chat_sessions/{test_anonymous_session.session.thread_id}/messages/{test_anonymous_session.messages[2].id}/rate/",
        data={"rating": "like"},
        content_type="application/json",
        HTTP_COOKIE=f"{AI_SESSION_COOKIE_KEY}=invalid_session_key",
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_rate_human_message_fails(client: APIClient, user_session, human_checkpoint):
    """Test that rating a human message fails"""
    user = user_session.user
    client.force_login(user)

    response = client.post(
        f"/api/v0/chat_sessions/{user_session.thread_id}/messages/{human_checkpoint.id}/rate/",
        data={"rating": "like"},
        content_type="application/json",
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.data == {
        "rating": [ErrorDetail(string="Can only rate agent responses", code="invalid")]
    }


def test_rate_nonexistent_checkpoint_fails(client, user_session):
    """Test rating a non-existent checkpoint fails"""
    client.force_login(user_session.user)

    response = client.post(
        f"/api/v0/chat_sessions/{user_session.thread_id}/messages/0/rate/",
        data={"rating": "like"},
        content_type="application/json",
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_update_existing_rating(client, user_session, agent_checkpoint):
    """Test updating an existing rating"""
    ChatResponseRatingFactory(checkpoint=agent_checkpoint, rating="like")
    client.force_login(user_session.user)

    response = client.post(
        f"/api/v0/chat_sessions/{user_session.thread_id}/messages/{agent_checkpoint.id}/rate/",
        data=json.dumps({"rating": "dislike"}),
        content_type="application/json",
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.data["rating"] == "dislike"

    # Verify only one rating exists and it's updated
    ratings = ChatResponseRating.objects.filter(checkpoint=agent_checkpoint)
    assert ratings.count() == 1
    assert ratings.first().rating == "dislike"


def test_llm_model_viewset(client):
    """Test LLMModelViewSet"""
    providers = ["openai", "google"]
    [
        LLMModelFactory.create_batch(randint(1, 3), provider=provider)  # noqa: S311
        for provider in providers
    ]
    llm_queryset = LLMModel.objects.filter(enabled=True)
    response = client.get("/api/v0/llm_models/")
    assert response.status_code == 200
    assert len(response.json()) == llm_queryset.count()

    # Test filtering by provider
    for provider in providers:
        response = client.get("/api/v0/llm_models/", {"provider": provider})
        assert response.status_code == 200
        for result in response.json():
            assert result["provider"] == provider
        assert (
            len(response.json())
            == LLMModel.objects.filter(provider=provider, enabled=True).count()
        )


def test_llm_model_viewset_enabled_only(client):
    """Test that LLMModelViewSet does nto return disabled models"""
    disabled_model = LLMModelFactory.create(enabled=False)
    response = client.get("/api/v0/llm_models/")
    assert response.status_code == 200
    assert disabled_model.litellm_id not in [
        model["litellm_id"] for model in response.json()
    ]


def test_get_transcript_block_id():
    """Test that get_transcript_block_id returns the transcript block ID correctly"""
    contentfile = {
        "edx_module_id": "block-v1:MITxT+3.012Sx+3T2024+type@video+block@ab8c1a02e9804e75aff98835dd03c28d",
        "content": '<video url_name="ab8c1a02e9804e75aff98835dd03c28d" sub="" transcripts="{&quot;en&quot;: &quot;df9493c5-4765-4b0c-a7bb-7ea732fea90e-en.srt&quot;}" display_name="State of Matter and Bonding" download_track="true" download_video="true" edx_video_id="df9493c5-4765-4b0c-a7bb-7ea732fea90e" html5_sources="[]" youtube_id_1_0="">\n  <video_asset client_video_id="3012_State_of_Matter_and_Bonding.mp4" duration="0.0" image="">\n    <encoded_video profile="hls" url="https://d3tsb3m56iwvoq.cloudfront.net/transcoded/51371d122b294c0ab27df648cf6068ca/video__index.m3u8" file_size="0" bitrate="0"/>\n    <transcripts>\n      <transcript language_code="en" file_format="srt" provider="Custom"/>\n    </transcripts>\n  </video_asset>\n  <transcript language="en" src="df9493c5-4765-4b0c-a7bb-7ea732fea90e-en.srt"/>\n</video>',
    }

    expected_block_id = "asset-v1:MITxT+3.012Sx+3T2024+type@asset+block@df9493c5-4765-4b0c-a7bb-7ea732fea90e-en.srt"
    assert get_transcript_block_id(contentfile) == expected_block_id


def test_list_all_prompts(client):
    """Test getting all system prompts."""
    response = client.get("/api/v0/prompts/")
    assert response.status_code == 200

    results = response.json()
    assert len(results) == 4

    prompt_names = [item["prompt_name"] for item in results]
    assert sorted(prompt_names) == sorted(CHATBOT_PROMPT_MAPPING.keys())

    for item in results:
        prompt_name = item["prompt_name"]
        assert item["prompt_value"] == parse_prompt(
            CHATBOT_PROMPT_MAPPING.get(prompt_name), prompt_name
        )


@pytest.mark.parametrize("prompt_name", CHATBOT_PROMPT_MAPPING.keys())
def test_get_specific_prompt(client, prompt_name):
    """Test getting a specific system prompt."""
    response = client.get(f"/api/v0/prompts/{prompt_name}/")
    assert response.status_code == 200

    result = response.json()
    assert result["prompt_name"] == prompt_name
    assert result["prompt_value"] == CHATBOT_PROMPT_MAPPING.get(prompt_name)


def test_invalid_prompt_name(client):
    """A non-existent prompt returns 404."""
    response = client.get("/api/v0/prompts/invalid_prompt/")
    assert response.status_code == 404
