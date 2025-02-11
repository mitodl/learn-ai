"""Permission classes for ai_chatbots views"""

from rest_framework.permissions import BasePermission

from ai_chatbots.models import UserChatSession


class IsThreadOwner(BasePermission):
    """
    Permission to only allow owners of a thread to view its messages.
    """

    def has_permission(self, request, view):
        # Allow staff users full access
        if request.user.is_superuser:
            return True

        # Check if thread belongs to user
        return UserChatSession.objects.filter(
            thread_id=view.kwargs.get("thread_id"), user=request.user
        ).exists()
