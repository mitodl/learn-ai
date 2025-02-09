"""
Models for langgraph checkpoint tables.  These models were reverse-engineered
via inspectdb after running langgraph.checkpoint.postgres.aio.PostgresSaver.setup()
Their data should not be modified by Django, only by langgraph.
"""

from django.db import models


class DjLangCheckpointWrite(models.Model):
    """
    Generated by inspectdb from langgraph.checkpoint.postgres.aio.PostgresSaver.setup()
    """

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


class DjLangCheckpoint(models.Model):
    """
    Generated by inspectdb from langgraph.checkpoint.postgres.aio.PostgresSaver.setup()
    """

    thread_id = models.TextField()
    checkpoint_ns = models.TextField()
    checkpoint_id = models.TextField()
    parent_checkpoint_id = models.TextField(blank=True, null=True)  # noqa: DJ001
    type = models.TextField(blank=True, null=True)  # noqa: DJ001
    checkpoint = models.BinaryField()
    metadata = models.JSONField()

    class Meta:
        unique_together = (("thread_id", "checkpoint_ns", "checkpoint_id"),)

    def __str__(self):
        return f"{self.thread_id}-{self.checkpoint_id}-{self.type}"
