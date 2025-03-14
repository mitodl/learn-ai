import logging

from channels.auth import AuthMiddlewareStack
from django.conf import settings
from starlette.middleware.cors import CORSMiddleware

from users.middleware import ApisixChannelAuthMiddleware

log = logging.getLogger(__name__)


def cors_middleware(app):
    """
    Return application with CORS middleware applied.
    """

    return CORSMiddleware(
        app,
        allow_origins=settings.CORS_ALLOW_ORIGINS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )


HTTP_MIDDLEWARE = [cors_middleware, AuthMiddlewareStack, ApisixChannelAuthMiddleware]

WS_MIDDLEWARE = [AuthMiddlewareStack]
