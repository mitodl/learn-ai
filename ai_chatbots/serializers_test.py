"""Tests for serializers"""

import pytest

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
