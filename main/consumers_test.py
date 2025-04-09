"""Tests for main consumer classes"""

import pytest
from rest_framework import exceptions

from main.consumers import BaseThrottledAsyncConsumer


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
