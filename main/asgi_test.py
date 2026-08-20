"""Tests for the ASGI entrypoint's OpenTelemetry wiring."""

import pytest
from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

from main.asgi import _otel_span_details, application


def test_application_is_otel_wrapped():
    """The outermost ASGI app must emit a SERVER span for every request.

    Channels routes nearly all traffic to consumers that never reach
    django_asgi_app, so DjangoInstrumentor alone leaves those requests untraced.
    """
    assert isinstance(application, OpenTelemetryMiddleware)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # Channels consumer route: static, outside the Django urlconf.
        ("/http/tutor_agent/", "GET http/tutor_agent/"),
        # Django routes collapse to their pattern, not the concrete id.
        (
            "/api/v0/chat_sessions/8f14e45f-ceea-467a-9f4c-3b1c2a1d0e77/messages/",
            "GET ^api/v0/chat_sessions/(?P<thread_id>[A-Za-z0-9_\\-]+)/messages/$",
        ),
        ("/learn-api/courses/12345/", "GET ^learn-api/(?P<path>.*)$"),
        ("/health/", "GET health/"),
        # Unmatched paths are 404s and get no series of their own.
        ("/nope/12345", "GET"),
    ],
)
def test_span_names_are_low_cardinality(path, expected):
    """Span names must never embed ids, or span metrics explode."""
    name, attributes = _otel_span_details({"method": "GET", "path": path})
    assert name == expected
    assert attributes == {}


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("GET", "GET health/"),
        ("get", "GET health/"),
        # ASGI accepts extension methods; an attacker-chosen verb must not
        # become its own span name.
        ("FOOBARBAZ", "HTTP health/"),
        ("", "HTTP health/"),
    ],
)
def test_method_is_sanitized(method, expected):
    """Cardinality can leak in through the method as easily as the path."""
    name, _ = _otel_span_details({"method": method, "path": "/health/"})
    assert name == expected
