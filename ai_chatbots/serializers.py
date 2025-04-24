"""Serializers for the ai_chatbots app"""

from django.conf import settings
from rest_framework import serializers

from ai_chatbots.models import DjangoCheckpoint, LLMModel, UserChatSession


class ChatRequestSerializer(serializers.Serializer):
    """Serializer for chatbot requests"""

    message = serializers.CharField(required=True, allow_blank=False)
    model = serializers.CharField(required=False, allow_blank=True)
    temperature = serializers.FloatField(
        min_value=0.0,
        max_value=1.0,
        required=False,
    )
    instructions = serializers.CharField(required=False, allow_blank=True)
    clear_history = serializers.BooleanField(
        default=False,
    )
    thread_id = serializers.CharField(required=False, allow_blank=True)

    def validate_instructions(self, value):
        """Ensure that the user has permission"""
        user = self.context.get("user")
        if not user or (not user.is_staff and not user.is_superuser):
            err_msg = "You do not have permission to adjust the instructions."
            raise serializers.ValidationError(err_msg)
        return value


class RecommendationChatRequestSerializer(ChatRequestSerializer):
    """
    Serializer for requests sent to the syllabus chatbot.
    """

    search_url = serializers.CharField(
        required=False, allow_blank=False, default=settings.AI_MIT_SEARCH_URL
    )


class SyllabusChatRequestSerializer(ChatRequestSerializer):
    """
    Serializer for requests sent to the syllabus chatbot.
    """

    course_id = serializers.CharField(required=True, allow_blank=False)
    collection_name = serializers.CharField(
        required=False, allow_blank=True, allow_null=True
    )


class VideoGPTRequestSerializer(ChatRequestSerializer):
    """
    Serializer for requests sent to the video GPT chatbot.
    """

    transcript_asset_id = serializers.CharField(required=True, allow_blank=False)


class UserChatSessionSerializer(serializers.ModelSerializer):
    """Serializer for user chat sessions"""

    class Meta:
        model = UserChatSession
        fields = (
            "thread_id",
            "title",
            "user",
            "created_on",
            "updated_on",
        )
        read_only_fields = ("created_on", "updated_on", "thread_id", "user")


class ChatMessageSerializer(serializers.ModelSerializer):
    """
    Serializer for chat messages.  This serializer is used to return just the message,
    content and role, and is intended to backfill chat history in a frontend UI.
    """

    role = serializers.CharField()
    content = serializers.CharField()

    def to_representation(self, instance):
        """Return just the message content and role"""
        role = "agent" if instance.metadata.get("writes", {}).get("agent") else "human"
        return {
            "checkpoint_id": instance.checkpoint_id,
            "step": instance.metadata.get("step"),
            "role": role,
            "content": instance.metadata.get("writes", {})
            .get("agent", {})
            .get("messages", [{}])[0]
            .get("kwargs", {})
            .get("content")
            if role == "agent"
            else instance.metadata.get("writes", {})
            .get("__start__", {})
            .get("messages", [{}])[0]
            .get("kwargs", {})
            .get("content"),
        }

    class Meta:
        model = DjangoCheckpoint
        fields = ["checkpoint_id", "role", "content"]


class TutorChatRequestSerializer(ChatRequestSerializer):
    """
    Serializer for requests sent to the tutor chatbot.
    """

    edx_module_id = serializers.CharField(required=True, allow_blank=False)
    block_siblings = serializers.ListField(
        child=serializers.CharField(), required=True, allow_empty=False
    )


class LLMModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = LLMModel
        fields = ["provider", "name", "litellm_id"]
