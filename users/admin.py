"""Django Admin configuration for the User model."""

from django.contrib import admin

from users.models import User


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    """User admin configuration."""

    list_display = (
        "id",
        "global_id",
        "username",
        "email",
        "name",
        "is_active",
        "is_staff",
        "is_superuser",
    )
    list_filter = ("is_active", "is_staff", "is_superuser")
    search_fields = ("username", "email", "name")
    ordering = ("username",)
    readonly_fields = ("id", "global_id")
    fieldsets = (
        (
            None,
            {
                "fields": (
                    "id",
                    "global_id",
                    "username",
                    "email",
                    "name",
                    "is_active",
                    "is_staff",
                    "is_superuser",
                )
            },
        ),
    )
    add_fieldsets = (
        (
            None,
            {
                "fields": (
                    "username",
                    "email",
                    "name",
                    "is_active",
                    "is_staff",
                    "is_superuser",
                    "password1",
                    "password2",
                )
            },
        ),
    )
    filter_horizontal = ()
    actions = ["deactivate_users", "activate_users"]

    def deactivate_users(self, request, queryset):  # noqa: ARG002
        """Deactivate selected users."""
        queryset.update(is_active=False)

    deactivate_users.short_description = "Deactivate selected users"

    def activate_users(self, request, queryset):  # noqa: ARG002
        """Activate selected users."""
        queryset.update(is_active=True)

    activate_users.short_description = "Activate selected users"
