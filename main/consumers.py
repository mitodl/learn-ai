import logging
from types import SimpleNamespace

from channels.consumer import AsyncConsumer
from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils.module_loading import import_string
from django.utils.text import slugify
from rest_framework.throttling import BaseThrottle

from main.exceptions import AsyncThrottled
from main.utils import anonymize_ident

log = logging.getLogger(__name__)


class BaseThrottledAsyncConsumer(AsyncConsumer):
    """
    Base class for throttled consumers.
    """

    throttle_classes = settings.CONSUMER_THROTTLE_CLASSES

    def get_session_key(self):
        """
        Get the session key for the current user.
        """
        session = self.scope.get("session")
        if session:
            if not session.session_key:
                session.save()
            return session.session_key
        return "Anonymous"

    def get_ident(self):
        """
        Get a unique identifier for the consumer user.
        """
        user = self.scope.get("user")
        if isinstance(user, get_user_model()):
            return user.global_id
        ident = self.get_session_key()
        return slugify(ident)

    def get_throttle_ident(self):
        """
        Get the identifier to throttle the consumer user by: the user for a
        signed-in user, the client IP address for an anonymous one.

        Not get_ident(): an anonymous session key changes every second on
        these routes, so it never fills a bucket (mitodl/hq#13968). And an IP
        must not become get_ident(), which also names anonymous chat threads.
        """
        user = self.scope.get("user")
        if isinstance(user, get_user_model()):
            return user.global_id
        request = SimpleNamespace(
            headers={
                key.decode("latin-1").lower(): value.decode("latin-1")
                for key, value in self.scope.get("headers") or []
            },
            META={"REMOTE_ADDR": (self.scope.get("client") or [None])[0]},
        )
        # DRF's lookup, so REST_FRAMEWORK["NUM_PROXIES"] decides which
        # X-Forwarded-For entry to trust, as in mit-learn (mitodl/hq#12775).
        return BaseThrottle().get_ident(request)

    def get_trace_ident(self):
        """
        Get an identifier for the consumer user that is safe to emit to
        external systems (trace/analytics providers such as LangSmith,
        Opik, and PostHog).

        Unlike get_ident(), which returns the raw session key for anonymous
        users, this hashes anonymous identifiers so a live session cookie
        value never leaves the process.
        """
        user = self.scope.get("user")
        if isinstance(user, get_user_model()):
            return user.global_id
        return anonymize_ident(self.get_ident())

    async def throttled(self, wait):
        """
        Determine what kind of exception to raise if throttled
        """
        raise AsyncThrottled(wait)

    async def get_throttles(self):
        """
        Instantiate and return the list of throttles that this view uses.
        """
        return [import_string(throttle)() for throttle in self.throttle_classes]

    async def check_throttles(self):
        """
        Check if request should be throttled.
        Raise an appropriate exception if the request is throttled.
        """

        throttle_durations = [
            await throttle.wait()
            for throttle in await self.get_throttles()
            if not await throttle.allow_request(self)
        ]

        if throttle_durations:
            durations = [
                duration for duration in throttle_durations if duration is not None
            ]

            duration = max(durations, default=None)
            await self.throttled(duration)
