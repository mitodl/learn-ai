"""DRF API views for chat sessions and messages."""

from django.db.models import QuerySet
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import (
    OpenApiParameter,
    OpenApiResponse,
    extend_schema,
    inline_serializer,
)
from open_learning_ai_tutor.problems import get_pb_sol
from rest_framework import mixins, serializers, viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from ai_chatbots.models import DjangoCheckpoint, UserChatSession
from ai_chatbots.permissions import IsThreadOwner
from ai_chatbots.serializers import ChatMessageSerializer, UserChatSessionSerializer
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


@extend_schema(
    parameters=[
        OpenApiParameter(
            name="problem_code",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH,
            description="problem_code of the problem",
        )
    ],
    responses={
        200: inline_serializer(
            name="ProblemAndSolutionResponse",
            fields={
                "problem": serializers.CharField(),
                "solution": serializers.CharField(),
            },
        ),
        404: OpenApiResponse(description="Missing problem code"),
    },
)
class TutorProblemView(APIView):
    """
    endpoint that returns the content of a sample tutor problem
    """

    permission_classes = ()
    http_method_names = ["get"]

    def get(self, request):
        """
        Return a sample tutor problem and solution.
        """
        problem_code = request.GET.get("problem_code")
        try:
            problem, solution = get_pb_sol(problem_code)

            return Response(
                {
                    "problem": problem,
                    "solution": solution,
                }
            )
        except KeyError:
            return Response(
                {"error": f"Problem code {problem_code} not found."}, status=404
            )
