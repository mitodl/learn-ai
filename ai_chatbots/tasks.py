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


@app.task(
    soft_time_limit=settings.AI_MEMORY_TASK_TIME_LIMIT,
    time_limit=settings.AI_MEMORY_TASK_TIME_LIMIT + 30,
)
def process_learner_memory(user_id: int):
    """Revise one learner's notes from their pending turns. Best-effort."""
    try:
        outcome = memory.process_learner_memory(user_id)
        log.info("Learner memory for user %s: %s", user_id, outcome)
    except Exception:
        log.exception("Memory extraction failed for user %s", user_id)


def schedule_learner_memory(user_id: int) -> None:
    """Run extraction after the batching delay."""
    process_learner_memory.apply_async(
        (user_id,), countdown=settings.AI_MEMORY_DELAY_SECONDS
    )


@app.task
def requeue_stale_memory_turns():
    """Recover pending turns whose task was lost or found the lock held."""
    for user_id in memory.stale_users():
        process_learner_memory.delay(user_id)
    stuck = memory.stuck_turn_count()
    if stuck:
        log.error("%d learner memory turns exceeded the retry limit", stuck)
