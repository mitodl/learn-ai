"""
ASGI Config file
"""

import os

from channels.routing import ProtocolTypeRouter, URLRouter

from main.middleware.configs import HTTP_MIDDLEWARE
from main.middleware.util import apply_middleware

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")

import ai_chatbots.routing  # noqa: I001


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
