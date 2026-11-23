"""Django Admin configuration for some AI models."""

from django.contrib import admin

from ai_chatbots.models import LLMModel, SupportWidgetCopy, UserChatSession


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


@admin.register(SupportWidgetCopy)
class SupportWidgetCopyAdmin(admin.ModelAdmin):
    """The Contact us widget's text. One row; a blank field keeps the default."""

    readonly_fields = ("updated_on",)

    def has_add_permission(self, request):
        return (
            super().has_add_permission(request)
            and not SupportWidgetCopy.objects.exists()
        )

    def has_delete_permission(self, request, obj=None):  # noqa: ARG002
        return False
