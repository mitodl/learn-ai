"""Django Admin configuration for some AI models."""

from django.contrib import admin

from ai_chatbots.models import LLMModel, MemoryStoreItem, UserChatSession


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


@admin.register(MemoryStoreItem)
class MemoryStoreItemAdmin(admin.ModelAdmin):
    """Learner memory documents, one per namespace."""

    list_display = ("namespace", "key", "updated_on")
    search_fields = ("namespace",)
    readonly_fields = ("created_on", "updated_on")
