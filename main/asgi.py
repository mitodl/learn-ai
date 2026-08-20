"""
ASGI Config file
"""

import os

import django
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
from django.urls import Resolver404, re_path, resolve
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
from opentelemetry.util.http import sanitize_method
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware

from main.middleware.configs import HTTP_MIDDLEWARE
from main.middleware.util import apply_middleware

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "main.settings")
# Ensure Django is set up before importing any app models or routing
django.setup()

import ai_chatbots.routing  # noqa: E402

django_asgi_app = get_asgi_application()


def _otel_span_details(scope: dict) -> tuple[str, dict]:
    """Name server spans by matched route rather than by raw path.

    Channels' URLRouter never populates scope["route"], which the OTel default
    relies on, and raw paths carry UUID thread ids plus the /learn-api/<path>
    proxy tail -- naming spans by path would blow up span-metric cardinality.

    The method is sanitized for the same reason the path is. ASGI accepts
    extension methods, so an unsanitized scope["method"] lets a caller mint a
    new span name per request just by varying the verb -- reintroducing through
    the method exactly the cardinality blowup this function exists to prevent.
    sanitize_method collapses anything off the RFC list to "_OTHER", which the
    ASGI and Django instrumentations then report as "HTTP".
    """
    method = sanitize_method((scope.get("method") or "").strip())
    if method == "_OTHER":
        method = "HTTP"
    path = (scope.get("path") or "/").lstrip("/")
    try:
        return f"{method} {resolve(f'/{path}').route}", {}
    except Resolver404:
        # The chatbot consumers are routed by Channels, outside the Django
        # urlconf, and their paths are static. Anything else unmatched is a 404,
        # so leave it as the bare method rather than giving scanner noise a
        # series of its own -- which is what DjangoInstrumentor does too.
        return (f"{method} {path}" if path.startswith("http/") else method), {}


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

# Outermost so that every HTTP request gets a SERVER span and continues the
# traceparent APISIX sends. Django's own instrumentation cannot do this: nearly
# all traffic is routed to Channels consumers that never reach django_asgi_app.
application = OpenTelemetryMiddleware(
    application, default_span_details=_otel_span_details
)
