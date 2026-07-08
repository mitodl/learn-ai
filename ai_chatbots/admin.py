"""Django Admin configuration for some AI models."""

from django.contrib import admin

from ai_chatbots.models import ContentFeedback, LLMModel, UserChatSession


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


@admin.register(ContentFeedback)
class ContentFeedbackAdmin(admin.ModelAdmin):
    """Content feedback admin configuration (append-only records)."""

    list_display = (
        "user",
        "course_id",
        "block_type",
        "block_usage_key",
        "sentiment",
        "created_on",
    )
    list_filter = ("sentiment", "course_id")
    search_fields = ("block_usage_key", "comment")
    ordering = ("-created_on",)
    readonly_fields = (
        "user",
        "course_id",
        "course_name",
        "block_usage_key",
        "block_type",
        "block_display_name",
        "unit_title",
        "url",
        "sentiment",
        "comment",
        "created_on",
        "updated_on",
    )

    def has_add_permission(self, request):  # noqa: ARG002
        return False

    def has_delete_permission(self, request, obj=None):  # noqa: ARG002
        return False
