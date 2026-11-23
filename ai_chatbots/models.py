"""ai_chabots models"""

import re

from django.core.exceptions import ValidationError
from django.db import models

from ai_chatbots.constants import ChatResponseScore
from main import settings
from main.models import TimestampedModel


class UserChatSession(TimestampedModel):
    """Associate LangGraph threads with users"""

    thread_id = models.TextField(unique=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True
    )
    dj_session_key = models.CharField(max_length=512, blank=True, db_index=True)
    title = models.CharField(max_length=255, blank=True)
    agent = models.CharField(max_length=128, blank=True, db_index=True)
    object_id = models.CharField(max_length=256, blank=True, db_index=True)
    # Set once a ticket is filed for this thread, so a learner who keeps typing
    # cannot open a second one.
    support_ticket_reference = models.CharField(max_length=64, blank=True, default="")
    support_ticket_email = models.EmailField(blank=True, default="")

    def __str__(self):
        user_id = self.user.global_id if self.user else self.dj_session_key
        return f"{user_id}-{self.thread_id}"


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

    created_on = models.DateTimeField(
        auto_now_add=True, db_index=True, null=True
    )  # UTC
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
        indexes = [
            # Composite index for queries that use all three fields
            models.Index(
                fields=["thread_id", "checkpoint_ns", "checkpoint_id"],
                name="checkpoint_lookup_idx",
            ),
            # Individual index for thread_id queries
            models.Index(fields=["thread_id"], name="thread_lookup_idx"),
        ]

    def __str__(self):
        return f"{self.thread_id}-{self.checkpoint_id}-{self.type}"


class TutorBotOutput(models.Model):
    """
    Store  chat history and internal state for the tutor chatbot
    """

    created_on = models.DateTimeField(
        auto_now_add=True, db_index=True, null=True
    )  # UTC
    session = models.ForeignKey(UserChatSession, on_delete=models.CASCADE, null=True)
    thread_id = models.TextField()
    chat_json = models.JSONField()
    edx_module_id = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return f"{self.thread_id}"


class LLMModel(models.Model):
    litellm_id = models.CharField(max_length=512, primary_key=True)
    provider = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    enabled = models.BooleanField(default=True)
    temperature = models.FloatField(default=None, null=True)
    reasoning_effort = models.CharField(max_length=16, blank=True)

    def __str__(self):
        return f"{self.provider} - {self.name}"


class ChatResponseRating(models.Model):
    """Store user ratings for AI chatbot responses"""

    RATING_CHOICES = [
        (ChatResponseScore.like.value, ChatResponseScore.like.value.capitalize()),
        (ChatResponseScore.dislike.value, ChatResponseScore.dislike.value.capitalize()),
        (ChatResponseScore.no_rating.value, "No Rating"),
    ]

    checkpoint = models.OneToOneField(
        DjangoCheckpoint,
        on_delete=models.CASCADE,
        related_name="rating",
        primary_key=True,
    )
    rating = models.CharField(
        max_length=10, choices=RATING_CHOICES, db_index=True, blank=True
    )
    rating_reason = models.TextField(blank=True)
    created_on = models.DateTimeField(auto_now_add=True)
    updated_on = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["rating"], name="rating_value_idx"),
        ]

    def __str__(self):
        return f"{self.checkpoint.checkpoint_id}-{self.rating}"


# The widget's own defaults, in smoot-design ContactUsLauncher.
SUPPORT_WIDGET_SEND_DEFAULT = "Send request"
SUPPORT_WIDGET_CANCEL_DEFAULT = "Cancel"
MARKDOWN_IMAGE = re.compile(r"!\[")


class SupportWidgetCopy(TimestampedModel):
    """
    The Contact us widget's text, edited by product in the admin. One row. A blank
    field leaves the widget's built-in default in place.
    """

    launcher_label = models.CharField(
        max_length=40, blank=True, help_text='Blank shows "Contact us".'
    )
    escalate_label = models.CharField(
        max_length=60, blank=True, help_text='Blank shows "Contact the support team".'
    )
    send_label = models.CharField(
        max_length=40,
        blank=True,
        help_text=f'Blank shows "{SUPPORT_WIDGET_SEND_DEFAULT}".',
    )
    cancel_label = models.CharField(
        max_length=40,
        blank=True,
        help_text=f'Blank shows "{SUPPORT_WIDGET_CANCEL_DEFAULT}".',
    )
    placeholder = models.CharField(
        max_length=80, blank=True, help_text='Blank shows "Describe what you need".'
    )
    badge = models.CharField(
        max_length=30, blank=True, help_text='Blank shows "AI ASSISTANT".'
    )
    greeting = models.TextField(
        max_length=1000,
        blank=True,
        help_text="Markdown, no images. Blank shows the built-in greeting.",
    )

    class Meta:
        verbose_name_plural = "support widget copy"

    def __str__(self):
        return "Support widget copy"

    def clean(self):
        for field in self._meta.get_fields():
            if isinstance(field, models.CharField | models.TextField):
                setattr(self, field.name, getattr(self, field.name).strip())

        errors = {}
        for field in ("send_label", "cancel_label", "escalate_label"):
            if getattr(self, field).casefold() == "send":
                errors[field] = (
                    '"Send" is already the name of the button that sends a message.'
                )
        if MARKDOWN_IMAGE.search(self.greeting):
            errors["greeting"] = "Images can't be used in the greeting."
        send = self.send_label or SUPPORT_WIDGET_SEND_DEFAULT
        cancel = self.cancel_label or SUPPORT_WIDGET_CANCEL_DEFAULT
        if send.casefold() == cancel.casefold() and "send_label" not in errors:
            errors["send_label"] = (
                f'The Send and Cancel buttons can\'t both say "{send}".'
            )
        if errors:
            raise ValidationError(errors)
