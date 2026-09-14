"""Django Admin configuration for some AI models."""

from django.contrib import admin

from ai_chatbots.models import (
    LearnerMemoryNote,
    LLMModel,
    PendingMemoryTurn,
    UserChatSession,
)


@admin.register(UserChatSession)
class UserChatSessionAdmin(admin.ModelAdmin):
    """User admin configuration."""

    list_display = ("thread_id", "user", "created_on", "updated_on", "title", "agent")
    list_filter = ("agent", "user")
    search_fields = ("title", "thread_id")
    ordering = ("-updated_on",)
    readonly_fields = ("agent", "thread_id", "created_on", "updated_on", "user")


@admin.register(LLMModel)
class LLMModelAdmin(admin.ModelAdmin):
    """LLM Model admin configuration."""

    list_display = (
        "provider",
        "name",
        "litellm_id",
        "enabled",
    )
    list_filter = (
        "provider",
        "enabled",
    )
    search_fields = ("name", "litellm_id")
    ordering = ("provider", "name", "litellm_id")


@admin.register(LearnerMemoryNote)
class LearnerMemoryNoteAdmin(admin.ModelAdmin):
    """Learned notes, one row per learner and section."""

    list_display = ("user", "key", "updated_on")
    search_fields = ("user__global_id", "user__email", "key")
    readonly_fields = ("created_on", "updated_on")


@admin.register(PendingMemoryTurn)
class PendingMemoryTurnAdmin(admin.ModelAdmin):
    """Exchanges awaiting extraction."""

    list_display = ("user", "bot", "attempts", "created_on")
    search_fields = ("user__global_id", "user__email")
