"""Tasks for AI chatbots"""

import logging
from datetime import timedelta

from django.conf import settings

from ai_chatbots import memory
from ai_chatbots.models import UserChatSession
from main.celery import app
from main.utils import now_in_utc

log = logging.getLogger(__name__)


@app.task
def delete_stale_sessions():
    """Delete any old anonymous chat sessions"""
    cutoff_dt = now_in_utc() - timedelta(days=settings.AI_CHATBOTS_SESSION_EXPIRY_DAYS)
    UserChatSession.objects.filter(created_on__lt=cutoff_dt, user=None).delete()


@app.task
def extract_learner_memory(global_id: str, bot_name: str, message: str, response: str):
    """
    Revise the learner's memory from one exchange. Best-effort.

    ponytail: runs on every response and sees one exchange with no thread context.
    Add the per-user 15 minute throttle and a last-processed marker (then feed the
    thread's recent messages) when volume matters.
    """
    try:
        memory.extract_learner_memory(global_id, bot_name, message, response)
    except Exception:
        log.exception("Memory extraction failed for %s", global_id)
