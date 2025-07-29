"""DRF API views for chat sessions and messages."""

import requests
from bs4 import BeautifulSoup
from django.conf import settings
from django.db.models import QuerySet
from django_filters.rest_framework import DjangoFilterBackend
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, OpenApiResponse, extend_schema
from open_learning_ai_tutor.prompts import get_system_prompt
from rest_framework import mixins, viewsets
from rest_framework.exceptions import NotFound
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView as ApiView
from rest_framework.viewsets import GenericViewSet, ReadOnlyModelViewSet

from ai_chatbots.models import DjangoCheckpoint, LLMModel, UserChatSession
from ai_chatbots.permissions import IsThreadOwner
from ai_chatbots.prompts import CHATBOT_PROMPT_MAPPING
from ai_chatbots.serializers import (
    ChatMessageSerializer,
    LLMModelSerializer,
    SystemPromptSerializer,
    UserChatSessionSerializer,
)
from ai_chatbots.utils import get_django_cache
from main.constants import VALID_HTTP_METHODS
from main.views import DefaultPagination


@extend_schema(
    parameters=[
        OpenApiParameter(
            name="thread_id",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH,
            description="thread id of the chat session",
        )
    ]
)
class UserChatSessionsViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows user session chats to be viewed or edited.
    """

    http_method_names = VALID_HTTP_METHODS
    serializer_class = UserChatSessionSerializer
    pagination_class = DefaultPagination
    permission_classes = (IsAuthenticated,)
    lookup_field = "thread_id"

    def get_queryset(self) -> QuerySet:
        """
        Return chat sessions for a logged in user.
        """
        user = self.request.user if hasattr(self, "request") else None
        return UserChatSession.objects.filter(user=user).order_by("-created_on")


@extend_schema(
    parameters=[
        OpenApiParameter(
            name="thread_id",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH,
            description="thread id of the chat session",
        )
    ]
)
class ChatMessageViewSet(mixins.ListModelMixin, viewsets.GenericViewSet):
    """
    Read-only API endpoint for returning just human/agent chat messages in a thread.
    """

    http_method_names = ["get"]
    serializer_class = ChatMessageSerializer
    permission_classes = (
        IsAuthenticated,
        IsThreadOwner,
    )
    pagination_class = DefaultPagination

    message_filter = """
    (metadata->'writes'->'__start__'->'messages'->0->'kwargs'->>'type')::text IN
    ('human', 'system') OR (
        metadata->'writes'->'agent'->'messages'->0->'kwargs'->>'content' IS NOT NULL
        AND metadata->'writes'->'agent'->'messages'->0->'kwargs'->>'content' != ''
    )
    """

    def get_queryset(self):
        thread_id = self.kwargs["thread_id"]
        return (
            DjangoCheckpoint.objects.filter(thread_id=thread_id)
            .extra(where=[self.message_filter])  # noqa: S610 - just a hardcoded filter
            .order_by("metadata__step")
        )


class LLMModelViewSet(ReadOnlyModelViewSet):
    """
    API view to list available LLM models.
    """

    queryset = LLMModel.objects.filter(enabled=True)
    serializer_class = LLMModelSerializer
    permission_classes = (AllowAny,)
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ["provider"]
    ordering = ["provider", "name"]
    ordering_fields = ["provider", "name", "litellm_id"]


@extend_schema(
    parameters=[
        OpenApiParameter(
            name="run_readable_id",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH,
            description="run_readable_id of the course run",
        )
    ],
    responses={
        200: OpenApiResponse(description="List of problem sets"),
        500: OpenApiResponse(description="Error retrieving problem sets"),
    },
)
class ProblemSetList(ApiView):
    """
    API view to get a list of problem sets for a given course.
    """

    http_method_names = ["get"]
    permission_classes = (AllowAny,)

    def get(self, request, *args, **kwargs):  # noqa: ARG002
        run_readable_id = request.query_params.get("run_readable_id")
        if not run_readable_id:
            return Response(
                {"error": "run_readable_id parameter is required."}, status=400
            )

        url = f"{settings.PROBLEM_SET_URL}/{run_readable_id}"
        headers = {"Authorization": f"Bearer {settings.LEARN_ACCESS_TOKEN}"}
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return Response(response.json(), status=200)
        except requests.RequestException:
            return Response({"error": "Something went wrong"}, status=500)


@extend_schema(
    parameters=[
        OpenApiParameter(
            name="edx_module_id",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH,
            description="edx_module_id of the video content file",
        )
    ],
    responses={
        200: OpenApiResponse(description="Transcript block ID"),
        500: OpenApiResponse(description="Error retrieving transcript block ID"),
    },
)
class GetTranscriptBlockId(ApiView):
    """
    API view to get the transcript block ID from edx block for a cotentfile.
    """

    http_method_names = ["get"]
    permission_classes = (AllowAny,)

    def get(self, request, *args, **kwargs):  # noqa: ARG002
        edx_module_id = request.query_params.get("edx_module_id")
        if not edx_module_id:
            return Response(
                {"error": "edx_module_id parameter is required."}, status=400
            )

        try:
            url = settings.AI_MIT_CONTENTFILE_URL
            params = {
                "edx_module_id": edx_module_id,
            }
            response = requests.get(
                url,
                params=params,
                headers={"Authorization": f"Bearer {settings.LEARN_ACCESS_TOKEN}"},
                timeout=30,
            )
            response.raise_for_status()
            response = response.json()
            contentfile = (
                response.get("results")[0] if response.get("results") else None
            )

            transcript_block_id = get_transcript_block_id(contentfile)

            return Response(
                {"transcript_block_id": transcript_block_id},
                status=200,
            )

        except requests.RequestException:
            return Response(
                {"error": "Failed to retrieve contentfile"},
                status=500,
            )
        except ValueError as e:
            return Response(
                {"error": e.args[0]},
                status=500,
            )


def get_transcript_block_id(contentfile):
    """
    Given a video contentfile object with attributes return
    the transcript block ID.
    """
    video_block_id = contentfile.get("edx_module_id")
    xml_content = contentfile.get("content")

    if not xml_content:
        msg = "Contentfile has no content."
        raise ValueError(msg)

    soup = BeautifulSoup(xml_content, "html.parser")

    video_tag = soup.find("video")

    if video_tag is None:
        msg = "Contentfile has no video tag."
        raise ValueError(msg)

    transcripts = soup.find_all("transcript")
    if transcripts is None:
        msg = "Contentfile has no transcripts."
        raise ValueError(msg)

    english_transcript_id = None

    for transcript in transcripts:
        if transcript.get("language") == "en" and transcript.get("src"):
            english_transcript_id = transcript.get("src")
            break

    if not english_transcript_id:
        msg = "Contentfile has no English transcript."
        raise ValueError(msg)

    transcript_id_prefix = video_block_id.replace("block-v1:", "asset-v1:").replace(
        "video+block", "asset+block"
    )
    parts = transcript_id_prefix.split("@")
    transcript_id_prefix = "@".join(parts[:2])

    return f"{transcript_id_prefix}@{english_transcript_id}"


class SystemPromptViewSet(GenericViewSet):
    """
    API endpoint to retrieve chatbot system prompts.
    """

    serializer_class = SystemPromptSerializer
    permission_classes = (AllowAny,)
    lookup_field = "prompt_name"

    def get_queryset(self):
        """Return all available chatbot prompts."""
        return [
            {
                "prompt_name": name,
                "prompt_value": get_system_prompt(
                    name, CHATBOT_PROMPT_MAPPING, get_django_cache
                ),
            }
            for name in CHATBOT_PROMPT_MAPPING
        ]

    def get_object(self):
        """Return a specific system prompt."""
        prompt_name = self.kwargs.get(self.lookup_field)
        if prompt_name not in CHATBOT_PROMPT_MAPPING:
            raise NotFound

        return {
            "prompt_name": prompt_name,
            "prompt_value": get_system_prompt(
                prompt_name, CHATBOT_PROMPT_MAPPING, get_django_cache
            ),
        }

    def list(self, request, *args, **kwargs):  # noqa: ARG002
        """Return a list of system prompts."""
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    @extend_schema(
        parameters=[
            OpenApiParameter(
                name="prompt_name",
                type=OpenApiTypes.STR,
                location=OpenApiParameter.PATH,
                description="name of the system prompt",
            )
        ]
    )
    def retrieve(self, request, *args, **kwargs):  # noqa: ARG002
        """Return a specific system prompt."""
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        return Response(serializer.data)
