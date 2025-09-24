"""Permission classes for ai_chatbots views"""

from rest_framework.permissions import BasePermission

from ai_chatbots.constants import AI_SESSION_COOKIE_KEY
from ai_chatbots.models import UserChatSession


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

        # For authenticated users, check if thread belongs to user
        if request.user.is_authenticated:
            return UserChatSession.objects.filter(
                thread_id=thread_id, user=request.user
            ).exists()

        # For anonymous users, check if thread belongs to session
        session_key = request.COOKIES.get(AI_SESSION_COOKIE_KEY)
        return UserChatSession.objects.filter(
            thread_id=thread_id, user=None, dj_session_key=session_key
        ).exists()

        return False
