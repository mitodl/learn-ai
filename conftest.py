"""Project conftest"""

# pylint: disable=wildcard-import, unused-wildcard-import
from types import SimpleNamespace

import pytest

from fixtures.common import *  # noqa: F403
from fixtures.users import *  # noqa: F403
from main.exceptions import DoNotUseRequestException


@pytest.fixture(autouse=True)
def prevent_requests(mocker, request):  # noqa: PT004
    """Patch requests to error on request by default"""
    if "mocked_responses" in request.fixturenames:
        return
    mocker.patch(
        "requests.sessions.Session.request",
        autospec=True,
        side_effect=DoNotUseRequestException,
    )


@pytest.fixture(autouse=True)
def _use_dummy_redis_cache_backend(settings):
    new_cache_settings = settings.CACHES.copy()
    new_cache_settings["redis"] = {
        "BACKEND": "django.core.cache.backends.dummy.DummyCache",
    }
    settings.CACHES = new_cache_settings


@pytest.fixture
def mock_http_consumer_send(mocker):
    """Mock the AsyncHttpConsumer class send functions"""
    mock_send_body = mocker.patch(
        "channels.generic.http.AsyncHttpConsumer.send_body",
        new_callable=mocker.AsyncMock,
    )
    mock_send_headers = mocker.patch(
        "channels.generic.http.AsyncHttpConsumer.send_headers",
        new_callable=mocker.AsyncMock,
    )
    return SimpleNamespace(
        send_body=mock_send_body,
        send_headers=mock_send_headers,
    )
