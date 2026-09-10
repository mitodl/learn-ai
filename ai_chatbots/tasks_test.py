"""Test for ai_chatbots tasks"""

from datetime import timedelta
from uuid import uuid4

import pytest
from freezegun import freeze_time

from ai_chatbots.factories import (
    CheckpointFactory,
    TutorBotOutputFactory,
    UserChatSessionFactory,
)
from ai_chatbots.models import DjangoCheckpoint, TutorBotOutput, UserChatSession
from ai_chatbots.tasks import delete_stale_sessions, extract_learner_memory
from main import settings
from main.utils import now_in_utc


@pytest.mark.django_db
def test_delete_stale_sessions():
    """Test delete_stale_sessions"""
    with freeze_time(
        lambda: (
            now_in_utc() - timedelta(days=settings.AI_CHATBOTS_SESSION_EXPIRY_DAYS + 1)
        )
    ):
        expired_chats = UserChatSessionFactory.create_batch(
            4, user=None, dj_session_key=uuid4().hex
        )
        expired_tutor_outputs = [
            TutorBotOutputFactory(session=chat) for chat in expired_chats[:2]
        ]
        expired_checkpoints = [
            CheckpointFactory(session=chat) for chat in expired_chats
        ]
    recent_chats = UserChatSessionFactory.create_batch(
        3, created_on=now_in_utc() - timedelta(days=1)
    )
    valid_tutor_outputs = [TutorBotOutputFactory(session=chat) for chat in recent_chats]
    valid_checkpoints = [CheckpointFactory(session=chat) for chat in recent_chats]
    delete_stale_sessions.apply()

    assert (
        UserChatSession.objects.filter(
            id__in=[chat.id for chat in expired_chats]
        ).count()
        == 0
    )
    assert (
        TutorBotOutput.objects.filter(
            id__in=[output.id for output in expired_tutor_outputs]
        ).count()
        == 0
    )
    assert (
        DjangoCheckpoint.objects.filter(
            id__in=[cp.id for cp in expired_checkpoints]
        ).count()
        == 0
    )
    assert (
        UserChatSession.objects.filter(
            id__in=[chat.id for chat in recent_chats]
        ).count()
        == 3
    )
    for output in valid_tutor_outputs:
        assert TutorBotOutput.objects.filter(id=output.id).exists()
    for cp in valid_checkpoints:
        assert DjangoCheckpoint.objects.filter(id=cp.id).exists()


def test_extract_learner_memory_invokes_manager_for_user(mocker):
    """The task feeds the exchange to the memory manager under the learner's id"""
    manager = mocker.patch("ai_chatbots.tasks.get_memory_manager").return_value
    extract_learner_memory("gid-1", "I only want online courses")
    manager.invoke.assert_called_once()
    payload, kwargs = manager.invoke.call_args.args[0], manager.invoke.call_args.kwargs
    # Only the learner's words: the bot's reply is never a source of learner facts
    assert [(m.type, m.content) for m in payload["messages"]] == [
        ("human", "I only want online courses")
    ]
    assert kwargs["config"]["configurable"]["langgraph_user_id"] == "gid-1"


def test_extract_learner_memory_logs_and_swallows_errors(mocker):
    """Extraction is best-effort: failures are logged, never raised"""
    mocker.patch(
        "ai_chatbots.tasks.get_memory_manager", side_effect=RuntimeError("boom")
    )
    log = mocker.patch("ai_chatbots.tasks.log")
    extract_learner_memory("gid-1", "hi")
    log.exception.assert_called_once()
