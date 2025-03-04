"""Tasks for AI chatbots"""

from datetime import timedelta

from ai_chatbots.models import UserChatSession
from main.celery import app
from main.utils import now_in_utc


@app.task
def delete_stale_sessions():
    """Delete any old anonymous chat sessions"""
    yesterday = now_in_utc() - timedelta(days=1)
    UserChatSession.objects.filter(created_on__lt=yesterday, user=None).delete()
