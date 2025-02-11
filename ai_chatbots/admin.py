"""Django Admin configuration for some AI models."""

from django.contrib import admin

from ai_chatbots.models import UserChatSession


@admin.register(UserChatSession)
class UserChatSessionAdmin(admin.ModelAdmin):
    """User admin configuration."""

    list_display = ("thread_id", "user", "created_on", "updated_on", "title", "agent")
    list_filter = ("agent", "user")
    search_fields = ("title", "thread_id")
    ordering = ("-updated_on",)
    readonly_fields = ("agent", "thread_id", "created_on", "updated_on")
