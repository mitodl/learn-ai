"""Tasks for AI chatbots"""

from datetime import timedelta

from django.conf import settings

from ai_chatbots.models import UserChatSession
from main.celery import app
from main.utils import now_in_utc


@app.task
def delete_stale_sessions():
    """Delete any old anonymous chat sessions"""
    cutoff_dt = now_in_utc() - timedelta(days=settings.AI_CHATBOTS_SESSION_EXPIRY_DAYS)
    UserChatSession.objects.filter(created_on__lt=cutoff_dt, user=None).delete()
