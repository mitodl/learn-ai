import ipaddress
import logging
import threading
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

# Set after the first non-public throttle address is logged, so it logs once.
_non_public_address_reported = threading.Event()


class BaseThrottledAsyncConsumer(AsyncConsumer):
    """
    Base class for throttled consumers.
    """

    throttle_classes = settings.CONSUMER_THROTTLE_CLASSES
    # False for routes a server calls on behalf of many learners, where the
    # caller's address would put every learner in one bucket.
    throttle_anonymous_by_address = True

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
        Get the throttle key: the user if signed in, else the client IP.

        Not get_ident(): the anonymous session key changes every second
        (mitodl/hq#13968), and get_ident() also names chat threads, so it
        can't become an IP.
        """
        user = self.scope.get("user")
        if isinstance(user, get_user_model()):
            return user.global_id
        if not self.throttle_anonymous_by_address:
            return self.get_ident()
        # DRF's lookup, as in mit-learn (mitodl/hq#12775).
        request = self.client_request()
        address = BaseThrottle().get_ident(request)
        # Visitors connect from public addresses, so a private one means
        # NUM_PROXIES counted one of our proxies as the client.
        if not _non_public_address_reported.is_set() and not _is_public(address):
            _non_public_address_reported.set()
            log.error(
                "Anonymous throttle keyed on non-public address %s "
                "(X-Forwarded-For %r, peer %s); check NUM_PROXIES",
                address,
                request.headers.get("x-forwarded-for"),
                request.META["REMOTE_ADDR"],
            )
        return address

    def client_request(self):
        """Return the scope's headers and peer address shaped like a DRF request."""
        return SimpleNamespace(
            headers={
                key.decode("latin-1").lower(): value.decode("latin-1")
                for key, value in self.scope.get("headers") or []
            },
            META={"REMOTE_ADDR": (self.scope.get("client") or [None])[0]},
        )

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


def _is_public(address):
    """Return whether the address is a public IP."""
    try:
        return ipaddress.ip_address(address).is_global
    except ValueError:
        return False
