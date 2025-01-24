"""
ASGI Config file
"""

import os

from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application

from main.middleware.configs import HTTP_MIDDLEWARE
from main.middleware.util import apply_middleware

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")

django_asgi_app = get_asgi_application()

import ai_chatbots.routing  # noqa: E402, I001


application = ProtocolTypeRouter(
    {
        "http": apply_middleware(
            HTTP_MIDDLEWARE, URLRouter(ai_chatbots.routing.http_patterns)
        ),
    }
)
