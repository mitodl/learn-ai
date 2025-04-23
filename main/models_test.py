"""main models tests"""

import pytest
from asgiref.sync import sync_to_async

from main.consumer_throttles import UserScopedRateThrottle
from main.factories import ConsumerThrottleLimitFactory


@pytest.mark.django_db
async def test_throttle_limit_save_reset_cache():
    """Saving changes to the throttle limit model should reset the cache"""
    auth_limits = (20, 10)
    anon_limits = (10, 5)
    intervals = ("minute", "hour")

    throttle_limit = await sync_to_async(ConsumerThrottleLimitFactory.create)(
        auth_limit=auth_limits[0],
        anon_limit=anon_limits[0],
        interval=intervals[0],
    )
    scoped_throttle = UserScopedRateThrottle()
    scoped_throttle.scope = throttle_limit.throttle_key
    assert await scoped_throttle.get_rate() == {
        "throttle_key": throttle_limit.throttle_key,
        "auth_limit": auth_limits[0],
        "anon_limit": anon_limits[0],
        "interval": intervals[0],
    }
    throttle_limit.auth_limit = auth_limits[1]
    throttle_limit.anon_limit = anon_limits[1]
    throttle_limit.interval = intervals[1]
    await throttle_limit.asave()
    assert await scoped_throttle.get_rate() == {
        "throttle_key": throttle_limit.throttle_key,
        "auth_limit": auth_limits[1],
        "anon_limit": anon_limits[1],
        "interval": intervals[1],
    }
