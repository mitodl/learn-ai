"""Tests for serializers"""

import pytest
from django.conf import settings

from ai_chatbots.factories import ChatResponseRatingFactory, CheckpointFactory
from ai_chatbots.models import ChatResponseRating
from ai_chatbots.serializers import (
    ChatMessageSerializer,
    ChatRatingSerializer,
    ChatRequestSerializer,
    SupportChatRequestSerializer,
)
from main.factories import UserFactory

pytestmark = pytest.mark.django_db


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


def test_valid_rating_serialization():
    """Test serializing valid rating data"""
    checkpoint = CheckpointFactory(is_agent=True)
    data = {"rating": "like"}

    serializer = ChatRatingSerializer(data=data, context={"checkpoint": checkpoint})

    assert serializer.is_valid()
    rating = serializer.save()

    assert rating.rating == "like"
    assert rating.checkpoint == checkpoint


def test_invalid_rating_validation():
    """Test validation of invalid rating values"""
    checkpoint = CheckpointFactory(is_agent=True)
    data = {"rating": "invalid"}

    serializer = ChatRatingSerializer(data=data, context={"checkpoint": checkpoint})

    assert not serializer.is_valid()
    assert "rating" in serializer.errors


def test_update_existing_rating():
    """Test updating an existing rating"""
    checkpoint = CheckpointFactory(is_agent=True)

    # Create initial rating
    ChatResponseRatingFactory(checkpoint=checkpoint, rating="like")

    # Update to dislike, with a reason for the rating
    data = {"rating": "dislike", "rating_reason": "Wildly inaccurate"}
    serializer = ChatRatingSerializer(data=data, context={"checkpoint": checkpoint})

    assert serializer.is_valid()
    updated_rating = serializer.save()
    updated_rating.refresh_from_db()

    assert updated_rating.rating == data["rating"]
    assert updated_rating.rating_reason == data["rating_reason"]
    assert updated_rating.checkpoint == checkpoint

    # Should only be one rating for this checkpoint
    assert ChatResponseRating.objects.filter(checkpoint=checkpoint).count() == 1


def test_empty_rating_validation():
    """Test validation when rating is empty"""
    checkpoint = CheckpointFactory(is_agent=True)
    data = {"rating": ""}

    serializer = ChatRatingSerializer(data=data, context={"checkpoint": checkpoint})

    assert serializer.is_valid()
    assert serializer.validated_data["rating"] == ""


def test_agent_message_with_rating():
    """Test agent message serialization includes rating"""
    checkpoint = CheckpointFactory(is_agent=True)
    ChatResponseRatingFactory(
        checkpoint=checkpoint, rating="like", rating_reason="Easy to understand"
    )

    serializer = ChatMessageSerializer(checkpoint)
    data = serializer.to_representation(checkpoint)

    assert data["role"] == "agent"
    assert data["rating"]["rating"] == "like"
    assert data["rating"]["rating_reason"] == "Easy to understand"


def test_agent_message_without_rating():
    """Test agent message without rating returns None"""
    checkpoint = CheckpointFactory(is_agent=True)

    serializer = ChatMessageSerializer(checkpoint)
    data = serializer.to_representation(checkpoint)

    assert data["role"] == "agent"
    assert data["rating"] is None


def test_human_message_no_rating_field():
    """Test human message doesn't include rating field"""
    checkpoint = CheckpointFactory(is_human=True)

    serializer = ChatMessageSerializer(checkpoint)
    data = serializer.to_representation(checkpoint)

    assert data["role"] == "human"
    assert "rating" not in data


def test_support_serializer_keeps_page_url():
    """The page the learner was on is passed through to the bot state."""
    serializer = SupportChatRequestSerializer(
        data={
            "message": "my video will not play",
            "page_url": "https://learn.mit.edu/courses/18.01/week-3",
        }
    )

    assert serializer.is_valid(), serializer.errors
    assert (
        serializer.validated_data["page_url"]
        == "https://learn.mit.edu/courses/18.01/week-3"
    )


@pytest.mark.parametrize(
    "page_url",
    [
        "javascript:alert(1)",
        "not a url at all",
        f"https://learn.mit.edu/{'x' * 2048}",
    ],
)
def test_support_serializer_rejects_bad_page_url(page_url):
    """page_url reaches a Zendesk ticket, so only real http(s) urls get through."""
    serializer = SupportChatRequestSerializer(
        data={"message": "help", "page_url": page_url}
    )

    assert not serializer.is_valid()
    assert "page_url" in serializer.errors


def test_support_serializer_keeps_user_email():
    """Hosts learn-ai has no session for supply the learner's address themselves."""
    serializer = SupportChatRequestSerializer(
        data={"message": "my video will not play", "user_email": "learner@mit.edu"}
    )

    assert serializer.is_valid(), serializer.errors
    assert serializer.validated_data["user_email"] == "learner@mit.edu"


def test_support_serializer_rejects_a_malformed_user_email():
    """The address becomes the Zendesk requester, so junk must not reach the tool."""
    serializer = SupportChatRequestSerializer(
        data={"message": "help", "user_email": "not an address"}
    )

    assert not serializer.is_valid()
    assert "user_email" in serializer.errors
