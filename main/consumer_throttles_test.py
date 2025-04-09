"""Tests for consumer throttle classes."""

from time import time

import pytest
from asgiref.sync import sync_to_async
from django.contrib.auth.models import AnonymousUser
from django.core.cache import cache

from main.constants import DURATION_MAPPING
from main.consumer_throttles import CONSUMER_THROTTLES_KEY, UserScopedRateThrottle
from main.consumers import BaseThrottledAsyncConsumer
from main.factories import ConsumerThrottleLimitFactory


@pytest.mark.parametrize("is_authenticated", [True, False])
@pytest.mark.parametrize("interval", ["minute", "hour", "day", "week"])
async def test_scoped_throttle_parse_rate(user, is_authenticated, interval):
    """Test the parse_rate method of ScopedRateThrottle."""
    throttle_limit = await sync_to_async(ConsumerThrottleLimitFactory.create)(
        interval=interval
    )
    user = user if is_authenticated else AnonymousUser()
    rate = await UserScopedRateThrottle().parse_rate(throttle_limit.__dict__, user)
    assert rate == (
        throttle_limit.auth_limit if is_authenticated else throttle_limit.anon_limit,
        DURATION_MAPPING[interval],
    )


@pytest.mark.parametrize("has_consumer_scope", [True, False])
@pytest.mark.parametrize("over_limit", [True, False])
async def test_scoped_allow_request(user, has_consumer_scope, over_limit):
    """Test the allow_request method of ScopedRateThrottle."""
    throttle_limit = await sync_to_async(ConsumerThrottleLimitFactory.create)()
    cache.set(CONSUMER_THROTTLES_KEY, None)
    consumer = BaseThrottledAsyncConsumer()
    consumer.scope = {"user": user} if has_consumer_scope else {}
    consumer.throttle_scope = throttle_limit.throttle_key
    consumer.throttle_classes = [UserScopedRateThrottle]
    cache.set(
        f"throttle_{throttle_limit.throttle_key}_{user.global_id}",
        [time() for t in range(10 if over_limit else 1)],
    )
    rate = UserScopedRateThrottle()
    allowed = await rate.allow_request(consumer)
    if not has_consumer_scope:
        assert allowed is True
    else:
        assert allowed is not over_limit


@pytest.mark.parametrize(
    ("now", "history", "duration", "num_requests", "expected"),
    [
        (100, [50], 100, 10, 5.0),
        (80, [50], 200, 10, 17.0),
        (80, [50], 200, 1, 170.0),
        (75, [50], 200, 0, None),
        (100, [50], 25, 1, -25.0),
        (100, [50], 3600, 1, 3550.0),
    ],
)
async def test_wait(now, history, duration, num_requests, expected):
    rate = UserScopedRateThrottle()
    rate.now = now
    rate.history = history
    rate.duration = duration
    rate.num_requests = num_requests
    assert await rate.wait() == expected
