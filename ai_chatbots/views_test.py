"""Tests for ai_chatbots views"""

from types import SimpleNamespace

import pytest
from django.urls import reverse
from open_learning_ai_tutor.problems import get_pb_sol

from ai_chatbots.factories import CheckpointFactory, UserChatSessionFactory
from ai_chatbots.serializers import ChatMessageSerializer, UserChatSessionSerializer
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


def test_tutor_problem_view(client):
    """Test TutorProblemVeiew"""

    response = client.get(reverse("ai:v0:tutor_problem"), {"problem_code": "A1P1"})
    assert response.status_code == 200
    problem, solution = get_pb_sol("A1P1")

    assert response.json()["problem"] == problem
    assert response.json()["solution"] == solution

    response = client.get(reverse("ai:v0:tutor_problem"), {"problem_code": "invalid"})
    assert response.status_code == 404
    assert response.json()["error"] == "Problem code invalid not found."
