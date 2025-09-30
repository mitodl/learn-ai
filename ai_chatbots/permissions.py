"""Permission classes for ai_chatbots views"""

from rest_framework.permissions import BasePermission

from ai_chatbots.constants import AI_THREADS_ANONYMOUS_COOKIE_KEY
from ai_chatbots.models import UserChatSession
from main.utils import decode_value


class IsThreadOwner(BasePermission):
    """
    Permission to only allow owners of a thread to view its messages.
    Supports both authenticated users and anonymous users via session keys.
    """

    def has_permission(self, request, view):
        # Allow staff users full access
        if request.user.is_superuser:
            return True

        thread_id = view.kwargs.get("thread_id")
        chat_session = UserChatSession.objects.filter(thread_id=thread_id).first()
        if not chat_session:
            return False

        # For authenticated users, check if thread belongs to user
        if request.user.is_authenticated:
            return chat_session.user == request.user

        # For anonymous users, check if thread belongs to anon user via cookie
        cookie_value = request.COOKIES.get(
            f"{chat_session.agent}_{AI_THREADS_ANONYMOUS_COOKIE_KEY}"
        )
        if cookie_value:
            return thread_id == decode_value(
                request.COOKIES.get(
                    f"{chat_session.agent}_{AI_THREADS_ANONYMOUS_COOKIE_KEY}"
                )
            )
        return False
