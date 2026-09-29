"""Tasks for AI chatbots"""

from datetime import timedelta

from django.conf import settings
from django.db.models import Count

from ai_chatbots.chatbots import SearchSummaryBot
from ai_chatbots.models import DjangoCheckpoint, UserChatSession
from main.celery import app
from main.utils import now_in_utc


@app.task
def delete_stale_sessions():
    """
    Delete any old anonymous chat sessions, and search summary sessions
    that never got a follow-up message.
    """
    cutoff_dt = now_in_utc() - timedelta(days=settings.AI_CHATBOTS_SESSION_EXPIRY_DAYS)
    UserChatSession.objects.filter(created_on__lt=cutoff_dt, user=None).delete()

    summary_cutoff_dt = now_in_utc() - timedelta(
        days=settings.AI_SEARCH_SUMMARY_EXPIRY_DAYS
    )
    # Each user message starts a new graph run, which saves an "input" checkpoint
    followed_up_sessions = (
        DjangoCheckpoint.objects.filter(
            metadata__source="input", session__agent=SearchSummaryBot.__name__
        )
        .values("session_id")
        .annotate(inputs=Count("id"))
        .filter(inputs__gt=1)
        .values("session_id")
    )
    UserChatSession.objects.filter(
        agent=SearchSummaryBot.__name__, created_on__lt=summary_cutoff_dt
    ).exclude(id__in=followed_up_sessions).delete()
