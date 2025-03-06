"""Test for ai_chatbots tasks"""

from datetime import timedelta
from uuid import uuid4

import pytest
from freezegun import freeze_time

from ai_chatbots.factories import UserChatSessionFactory
from ai_chatbots.models import UserChatSession
from ai_chatbots.tasks import delete_stale_sessions
from main import settings
from main.utils import now_in_utc


@pytest.mark.django_db
def test_delete_stale_sessions():
    """Test delete_stale_sessions"""
    with freeze_time(
        lambda: now_in_utc()
        - timedelta(days=settings.AI_CHATBOTS_SESSION_EXPIRY_DAYS + 1)
    ):
        expired_chats = UserChatSessionFactory.create_batch(
            3, user=None, dj_session_key=uuid4().hex
        )
        old_user_chats = UserChatSessionFactory.create_batch(3)
    recent_chats = UserChatSessionFactory.create_batch(
        3, created_on=now_in_utc() - timedelta(days=1)
    )
    delete_stale_sessions.apply()

    assert (
        UserChatSession.objects.filter(
            id__in=[chat.id for chat in expired_chats]
        ).count()
        == 0
    )
    for chat_list in [old_user_chats, recent_chats]:
        assert (
            UserChatSession.objects.filter(
                id__in=[chat.id for chat in chat_list]
            ).count()
            == 3
        )
