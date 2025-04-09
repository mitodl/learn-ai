import logging

from channels.consumer import AsyncConsumer
from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils.module_loading import import_string
from django.utils.text import slugify
from rest_framework import exceptions

log = logging.getLogger(__name__)


class BaseThrottledAsyncConsumer(AsyncConsumer):
    """
    Base class for throttled consumers.
    """

    throttle_classes = settings.CONSUMER_THROTTLE_CLASSES

    def get_ident(self):
        """
        Get a unique identifier for the consumer user.
        """
        user = self.scope.get("user")
        if isinstance(user, get_user_model()):
            return user.global_id
        session = self.scope.get("session")
        ident = session.session_key if session else None
        if session and not ident:
            session.save()
            ident = session.session_key
        if not ident:
            return "Anonymous"
        return slugify(ident)

    async def throttled(self, wait):
        """
        Determine what kind of exception to raise if throttled
        """
        raise exceptions.Throttled(wait)

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
