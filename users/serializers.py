"""Serializers for user data."""

from rest_framework import serializers

from users.models import User


class UserSerializer(serializers.ModelSerializer):
    """Serializer for the User model."""

    class Meta:
        model = User
        fields = (
            "id",
            "global_id",
            "username",
            "email",
            "name",
            "is_active",
            "is_staff",
            "is_superuser",
        )
        read_only_fields = (
            "id",
            "global_id",
            "username",
            "email",
            "name",
            "is_active",
            "is_staff",
            "is_superuser",
        )
