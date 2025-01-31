"""Serializers for the ai_chatbots app"""

from rest_framework import serializers


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


class SyllabusChatRequestSerializer(ChatRequestSerializer):
    """
    Serializer for requests sent to the syllabus chatbot.
    """

    course_id = serializers.CharField(required=True, allow_blank=False)
    collection_name = serializers.CharField(
        required=False, allow_blank=True, allow_null=True
    )
