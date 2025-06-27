"""Tests for serializers"""

import pytest
from django.conf import settings

from ai_chatbots.serializers import ChatRequestSerializer
from main.factories import UserFactory


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("is_staff", "is_super"),
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_instructions_permissions(is_staff, is_super):
    """Test that the instructions validation works as expected"""
    user = UserFactory.create(is_staff=is_staff, is_superuser=is_super)
    serializer = ChatRequestSerializer(
        data={"message": "Hello, world!", "instructions": "Do something"},
        context={"user": user},
    )
    assert serializer.is_valid() == (is_staff or is_super)


@pytest.mark.django_db
def test_message_truncation():
    """Test that the message field truncates correctly."""
    user = UserFactory.create(is_staff=True)

    settings.AI_MAX_MESSAGE_LENGTH = 6000

    long_message = "x" * (
        settings.AI_MAX_MESSAGE_LENGTH + 1000
    )  # Exceeds the max length
    serializer = ChatRequestSerializer(
        data={"message": long_message, "instructions": "Do something"},
        context={"user": user},
    )
    assert serializer.is_valid()
    assert len(serializer.validated_data["message"]) == settings.AI_MAX_MESSAGE_LENGTH
