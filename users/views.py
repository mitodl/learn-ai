"""Views for user metadata."""

from django.http import JsonResponse


def current_user(request):
    """Return the current user."""
    return (
        JsonResponse(
            {
                "global_id": request.user.global_id,
                "username": request.user.username,
                "email": request.user.email,
                "is_staff": request.user.is_staff,
                "is_superuser": request.user.is_superuser,
            }
        )
        if not request.user.is_anonymous
        else JsonResponse({"anonymous": True})
    )
