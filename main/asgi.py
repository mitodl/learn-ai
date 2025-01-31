"""
ASGI Config file
"""

import os

from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from django.urls import re_path

from main.middleware.configs import HTTP_MIDDLEWARE
from main.middleware.util import apply_middleware

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")

import ai_chatbots.routing

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
