"""Tests for main consumer classes"""

import pytest
from django.contrib.auth.models import AnonymousUser
from rest_framework import exceptions

from main.consumers import BaseThrottledAsyncConsumer
from main.utils import anonymize_ident


class MockThrottle(BaseThrottledAsyncConsumer):
    """
    Mock throttle class for testing.
    """

    wait_time = 10

    async def allow_request(self, consumer):  # noqa: ARG002
        return True

    async def wait(self):
        return self.wait_time


@pytest.mark.parametrize("allow_request", [True, False])
@pytest.mark.parametrize("wait_time", [5, 11])
async def test_throttle_consumer_check_throttles(mocker, allow_request, wait_time):
    """
    Check if request should be throttled.
    Raise an appropriate exception if the request is throttled.
    """
    mocker.patch(
        "main.consumer_throttles.UserScopedRateThrottle.allow_request",
        return_value=allow_request,
    )
    mocker.patch(
        "main.consumers_test.MockThrottle.allow_request", return_value=allow_request
    )
    mocker.patch(
        "main.consumer_throttles.UserScopedRateThrottle.wait", return_value=wait_time
    )
    mock_throttled = mocker.patch("main.consumers.BaseThrottledAsyncConsumer.throttled")
    consumer = BaseThrottledAsyncConsumer()
    consumer.throttle_classes = [
        "main.consumer_throttles.UserScopedRateThrottle",
        "main.consumers_test.MockThrottle",
    ]
    await consumer.check_throttles()
    if allow_request:
        mock_throttled.assert_not_called()
    else:
        mock_throttled.assert_called_once_with(
            wait_time if wait_time > MockThrottle.wait_time else MockThrottle.wait_time
        )


@pytest.mark.parametrize("wait_time", [5, 11])
async def test_throttle_consumer_throttled(wait_time):
    """
    Test the throttled method.
    """
    consumer = BaseThrottledAsyncConsumer()
    with pytest.raises(exceptions.Throttled) as excinfo:
        await consumer.throttled(wait_time)
    assert excinfo.value.wait == wait_time


def _consumer_for(mocker, *, user, session_key):
    """Build a bare consumer with a scope for identity tests."""
    consumer = BaseThrottledAsyncConsumer()
    mock_session = mocker.Mock()
    mock_session.session_key = session_key
    consumer.scope = {"user": user, "session": mock_session}
    return consumer


def test_get_trace_ident_hashes_anonymous_session_key(mocker):
    """Anonymous users' trace ident should be a hash, not the raw session key."""
    session_key = "abc123def456"
    consumer = _consumer_for(mocker, user=AnonymousUser(), session_key=session_key)

    trace_ident = consumer.get_trace_ident()

    assert trace_ident == anonymize_ident(consumer.get_ident())
    assert session_key not in trace_ident
    assert trace_ident.startswith("anon-")


@pytest.mark.django_db
def test_get_trace_ident_returns_global_id_for_authenticated_user(mocker, user):
    """Authenticated users' trace ident should still be their global_id."""
    consumer = _consumer_for(mocker, user=user, session_key="abc123def456")

    assert consumer.get_trace_ident() == user.global_id
    assert consumer.get_trace_ident() == consumer.get_ident()


def _anon_consumer(*, client_ip, forwarded_for=None):
    """Build a consumer for an anonymous request arriving through the gateway."""
    consumer = BaseThrottledAsyncConsumer()
    headers = [(b"x-forwarded-for", forwarded_for.encode())] if forwarded_for else []
    consumer.scope = {
        "user": AnonymousUser(),
        "client": (client_ip, 54321),
        "headers": headers,
    }
    return consumer


def test_get_throttle_ident_uses_rightmost_trusted_forwarded_address(settings):
    """Only the entry APISIX appended is trusted; the client controls the rest."""
    settings.REST_FRAMEWORK = {**settings.REST_FRAMEWORK, "NUM_PROXIES": 1}
    consumer = _anon_consumer(
        client_ip="10.0.0.2", forwarded_for="9.9.9.9, 203.0.113.5"
    )

    assert consumer.get_throttle_ident() == "203.0.113.5"


def test_get_throttle_ident_falls_back_to_peer_address(settings):
    """With no X-Forwarded-For, the peer address is the client."""
    settings.REST_FRAMEWORK = {**settings.REST_FRAMEWORK, "NUM_PROXIES": 1}
    consumer = _anon_consumer(client_ip="203.0.113.5")

    assert consumer.get_throttle_ident() == "203.0.113.5"


@pytest.mark.django_db
def test_get_throttle_ident_authenticated_uses_global_id(mocker, user):
    """Signed-in users are throttled per user, wherever they connect from."""
    consumer = _consumer_for(mocker, user=user, session_key="abc123def456")
    consumer.scope["client"] = ("203.0.113.5", 54321)

    assert consumer.get_throttle_ident() == user.global_id
