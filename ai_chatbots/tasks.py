"""Tasks for AI chatbots"""

import logging
from datetime import timedelta

from django.conf import settings
from langchain_core.messages import HumanMessage

from ai_chatbots.memory import get_memory_manager
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
def extract_learner_memory(global_id: str, message: str):
    """
    Revise the learner's memory document from one learner message. Best-effort.

    Only the learner's words go in: given the bot's reply too, gpt-4o-mini recorded
    recommended course titles as the learner's interests despite instructions.
    ponytail: runs on every response and sees one message with no thread context,
    so "the second one" means nothing. Add the per-user 15 minute throttle and a
    last-processed marker (then feed the learner's recent messages) when volume matters.
    """
    try:
        get_memory_manager().invoke(
            {"messages": [HumanMessage(content=message)]},
            config={"configurable": {"langgraph_user_id": global_id}},
        )
    except Exception:
        log.exception("Memory extraction failed for %s", global_id)
