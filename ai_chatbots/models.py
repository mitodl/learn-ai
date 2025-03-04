"""ai_chabots models"""

from django.db import models

from main import settings
from main.models import TimestampedModel


class UserChatSession(TimestampedModel):
    """Associate LangGraph threads with users"""

    thread_id = models.TextField(unique=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True
    )
    title = models.CharField(max_length=255, blank=True)
    agent = models.CharField(max_length=128, blank=True)

    def __str__(self):
        return f"{self.user.global_id if self.user else "anonymous"}-{self.thread_id}:{self.title}"


class DjangoCheckpointWrite(models.Model):
    """
    Temporary checkpoint data.
    """

    session = models.ForeignKey(UserChatSession, on_delete=models.CASCADE, null=True)
    thread_id = models.TextField()
    checkpoint_ns = models.TextField()
    checkpoint_id = models.TextField()
    task_id = models.TextField()
    idx = models.IntegerField()
    channel = models.TextField()
    type = models.TextField(blank=True, null=True)  # noqa: DJ001
    blob = models.BinaryField()
    task_path = models.TextField()

    class Meta:
        unique_together = (
            ("thread_id", "checkpoint_ns", "checkpoint_id", "task_id", "idx"),
        )

    def __str__(self):
        return f"{self.thread_id}-{self.checkpoint_id}-{self.idx}"


class DjangoCheckpoint(models.Model):
    """
    Checkpoint created by the DjangoSaver checkpointer class.
    """

    session = models.ForeignKey(UserChatSession, on_delete=models.CASCADE, null=True)
    thread_id = models.TextField()
    checkpoint_ns = models.TextField()
    checkpoint_id = models.TextField()
    parent_checkpoint_id = models.TextField(blank=True, null=True)  # noqa: DJ001
    type = models.TextField(blank=True, null=True)  # noqa: DJ001
    checkpoint = models.JSONField()
    metadata = models.JSONField()

    class Meta:
        unique_together = (("thread_id", "checkpoint_ns", "checkpoint_id"),)

    def __str__(self):
        return f"{self.thread_id}-{self.checkpoint_id}-{self.type}"



class TutorBotOutput(models.Model):
    """
    Store  chat history and internal state for the tutor chatbot
    """
    thread_id = models.TextField()
    chat_json = models.JSONField()

    def __str__(self):
        return f"{self.thread_id}"
