import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application

import ai_agents.routing

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")


application = ProtocolTypeRouter(
    {
        "http": get_asgi_application(),
        "websocket": AuthMiddlewareStack(
            URLRouter(ai_agents.routing.websocket_urlpatterns)
        ),
    }
)
