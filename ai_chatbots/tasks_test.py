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


def test_extract_learner_memory_task_delegates(mocker):
    """The task hands everything to the memory module"""
    extract = mocker.patch("ai_chatbots.tasks.memory.extract_learner_memory")
    extract_learner_memory("gid-1", "TutorBot", "t-1", "plain english please", "Sure.")
    extract.assert_called_once_with(
        "gid-1", "TutorBot", "t-1", "plain english please", "Sure."
    )


def test_extract_learner_memory_logs_and_swallows_errors(mocker):
    """Extraction is best-effort: failures are logged, never raised"""
    mocker.patch(
        "ai_chatbots.tasks.memory.extract_learner_memory",
        side_effect=RuntimeError("boom"),
    )
    log = mocker.patch("ai_chatbots.tasks.log")
    extract_learner_memory("gid-1", "TutorBot", "t-1", "hi", "hello")
    log.exception.assert_called_once()
