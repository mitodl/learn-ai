"""
Provides various throttling policies.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Literal

from asgiref.sync import sync_to_async
from django.contrib.auth import get_user_model
from django.core.cache import cache as default_cache
from django.core.exceptions import ImproperlyConfigured

from main.constants import DURATION_MAPPING
from main.models import ConsumerThrottleLimit

log = logging.getLogger(__name__)


CONSUMER_THROTTLES_KEY = "consumer_throttles"


class AsyncBaseThrottle(ABC):
    """
    Abstract class for throttling AsyncConsumer requests.
    Based on the DRF BaseThrottle class.
    """

    @abstractmethod
    async def allow_request(self) -> bool:
        """
        Return `True` if the request should be allowed, `False` otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def wait(self) -> int:
        """
        Return a recommended number of seconds to wait before
        the next request.
        """
        raise NotImplementedError


class AsyncSimpleRateThrottle(AsyncBaseThrottle):
    """
    A simple cache implementation, that only requires `.get_cache_key()`
    to be overridden.

    The rate (requests / seconds) is set by a `rate` attribute on the Throttle
    class.  The attribute is a string of the form 'number_of_requests/period'.

    Period should be one of:
    ('s', 'sec', 'm', 'min', 'h', 'hour', 'd', 'day', 'w', 'week')

    Previous request information used for throttling is stored in the cache.
    """

    cache = default_cache
    timer = time.time
    cache_format = "throttle_%(scope)s_%(ident)s"
    scope = None

    def __init__(self):
        if not getattr(self, "rate", None):
            self.rate = self.get_rate()
        self.num_requests, self.duration = self.parse_rate(self.rate)

    async def get_or_set_throttles(self) -> dict:
        """Get consumer throttles if defined or load them from database if empty"""
        throttles = default_cache.get(CONSUMER_THROTTLES_KEY, default={})
        if not throttles:
            # The correct way to use sync_to_async with a queryset method
            get_throttles = sync_to_async(
                lambda: {
                    item["throttle_key"]: item
                    for item in ConsumerThrottleLimit.objects.all().values()
                }
            )
            throttles = await get_throttles()
            default_cache.set(CONSUMER_THROTTLES_KEY, throttles)
        return throttles

    async def get_rate(self) -> dict:
        """
        Determine the string representation of the allowed request rate.
        """
        if not getattr(self, "scope", None):
            msg = "scope must be set for throttle"
            raise ImproperlyConfigured(msg)

        try:
            throttles = await self.get_or_set_throttles()
            return throttles[self.scope]
        except (KeyError, TypeError) as err:
            msg = f"No default throttle rate set:'{self.scope}' from {throttles}"
            raise ImproperlyConfigured(msg) from err

    async def get_cache_key(self, consumer) -> str:
        """
        Return a unique cache-key which can be used for throttling.
        Must be overridden.

        May return None if the request should not be throttled.
        """
        msg = "get_cache_key() must be overridden"
        raise NotImplementedError(msg)

    async def allow_request(self, consumer) -> bool:
        """
        Implement the check to see if the request should be throttled.

        On success calls `throttle_success`.
        On failure calls `throttle_failure`.
        """
        if not self.rate or not self.num_requests:
            # Rate is None or 0, so no throttling
            return True

        self.key = await self.get_cache_key(consumer)
        if self.key is None:
            return True

        self.history = self.cache.get(self.key, [])
        self.now = self.timer()
        # Drop any requests from the history which have now passed the
        # throttle duration
        while self.history and self.history[-1] <= self.now - self.duration:
            self.history.pop()
        if len(self.history) >= self.num_requests:
            return await self.throttle_failure()
        return await self.throttle_success()

    async def throttle_success(self) -> Literal[True]:
        """
        Insert the current request's timestamp along with the key
        into the cache.
        """
        self.history.insert(0, self.now)
        self.cache.set(self.key, self.history, self.duration)
        return True

    async def throttle_failure(self) -> Literal[False]:
        """
        Return False when a request to the API has failed due to throttling.
        """
        return False

    async def wait(self):
        """
        Return the recommended next request time in seconds.
        """
        if self.history:
            remaining_duration = self.duration - (self.now - self.history[-1])
        else:
            remaining_duration = self.duration

        available_requests = self.num_requests - len(self.history) + 1
        if available_requests <= 0:
            return None

        return remaining_duration / float(available_requests)


class UserScopedRateThrottle(AsyncSimpleRateThrottle):
    """
    Limits the rate of API calls by different amounts for various parts of
    the consumer API, per user.  Any consumer that has the `throttle_scope` property
    set will be throttled.  The unique cache key will be generated by concatenating the
    user id of the request, and the scope of the consumer being accessed.  Different
    limits can be set for authenticated and anonymous users.
    """

    scope_attr = "throttle_scope"

    def __init__(self):
        # Override the usual SimpleRateThrottle, because we can't determine
        # the rate until called by the consumer.
        pass

    async def parse_rate(self, rate, user) -> tuple[int, int]:
        """
        Return a tuple of:
        <allowed number of requests>, <period of time in seconds>
        for the given rate and user
        """
        authenticated = isinstance(user, get_user_model())
        if not rate or (authenticated and (user.is_staff or user.is_superuser)):
            # Staff and superusers are not throttled
            return (0, 0)

        user_limit = (
            rate.get("auth_limit", 0) if authenticated else rate.get("anon_limit", 0)
        )
        duration_seconds = DURATION_MAPPING[rate["interval"]]
        return (user_limit, duration_seconds)

    async def allow_request(self, consumer) -> bool:
        # We can only determine the scope once called by the consumer.
        self.scope = getattr(consumer, self.scope_attr, None)

        # If a view does not have a `throttle_scope` always allow the request
        if not self.scope:
            return True

        # Determine the allowed request rate
        self.rate = await self.get_rate()
        self.num_requests, self.duration = await self.parse_rate(
            self.rate, consumer.scope.get("user", None)
        )

        return await super().allow_request(consumer)

    async def get_cache_key(self, consumer) -> str:
        """
        If `consumer.throttle_scope` is not set, don't apply this throttle.

        Otherwise generate the unique cache key by concatenating the user id
        with the `.throttle_scope` property of the consumer.
        """
        return self.cache_format % {"scope": self.scope, "ident": consumer.get_ident()}
