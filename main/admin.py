"""Django Admin configuration for main models."""

from django.contrib import admin

from main.models import ConsumerThrottleLimit


@admin.register(ConsumerThrottleLimit)
class ConsumerThrottleAdmin(admin.ModelAdmin):
    """ConsumerThrottle configuration."""

    list_display = (
        "throttle_key",
        "auth_limit",
        "anon_limit",
        "interval",
    )
    list_filter = ("throttle_key",)
    search_fields = ("throttle_key",)
    ordering = ("throttle_key",)
