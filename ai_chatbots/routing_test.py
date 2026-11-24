"""Tests for the ai_chatbots Channels routes."""

import pytest

from ai_chatbots.routing import http_patterns


def _route_name(path):
    """Return the name of the first route that matches the path, as URLRouter does."""
    return next(
        (route.name for route in http_patterns if route.pattern.match(path)), None
    )


@pytest.mark.parametrize(
    ("path", "route_name"),
    [
        ("http/canvas_tutor_agent/", "canvas_tutor_agent_sse"),
        ("http/canvas_syllabus_agent/", "canvas_syllabus_agent_sse"),
        ("http/support_agent/", "support_agent_sse"),
    ],
)
def test_routes_match_their_own_path(path, route_name):
    """The paths the gateway forwards still reach their consumers."""
    assert _route_name(path) == route_name


@pytest.mark.parametrize(
    "path",
    [
        "x/http/canvas_tutor_agent/",
        "ai/x/http/canvas_syllabus_agent/",
        "x/http/support_agent/",
    ],
)
def test_prefixed_paths_match_no_route(path):
    """
    The gateway only asks for a canvas_token on paths starting /http/canvas_,
    so a prefixed path must not reach a consumer through the open route.
    """
    assert _route_name(path) is None


@pytest.mark.parametrize(
    "path", ["http/canvas_tutor_agent/x", "http/canvas_syllabus_agent/x"]
)
def test_suffixed_canvas_paths_match_no_route(path):
    """
    The local gateway asks for a canvas_token only on the exact Canvas paths,
    so a suffixed one must not reach a Canvas consumer through the open route.
    """
    assert _route_name(path) is None
