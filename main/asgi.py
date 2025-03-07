"""
ASGI Config file
"""

import os

import django
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from django.urls import re_path
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware

from main.middleware.configs import HTTP_MIDDLEWARE
from main.middleware.util import apply_middleware

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
# Ensure Django is set up before importing any app models or routing
django.setup()

import ai_chatbots.routing  # noqa: E402

django_asgi_app = get_asgi_application()

application = ProtocolTypeRouter(
    {
        "http": apply_middleware(
            HTTP_MIDDLEWARE,
            URLRouter(
                [
                    *ai_chatbots.routing.http_patterns,
                    re_path(r"", django_asgi_app),
                ]
            ),
        ),
    }
)

application = SentryAsgiMiddleware(application)
