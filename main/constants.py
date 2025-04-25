"""main constants"""

from rest_framework import status

PERMISSION_DENIED_ERROR_TYPE = "PermissionDenied"
NOT_AUTHENTICATED_ERROR_TYPE = "NotAuthenticated"
DJANGO_PERMISSION_ERROR_TYPES = (
    status.HTTP_401_UNAUTHORIZED,
    status.HTTP_403_FORBIDDEN,
)
VALID_HTTP_METHODS = ["get", "post", "patch", "delete"]


DURATION_MAPPING = {"minute": 60, "hour": 3600, "day": 86400, "week": 604800}

CONSUMER_THROTTLES_KEY = "consumer_throttles"
