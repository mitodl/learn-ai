"""Middleware to handle users/auth."""

import base64
import json
import logging

from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model

log = logging.getLogger(__name__)


class ApisixChannelAuthMiddleware:
    """
    Middleware for authenticating users with APISIX via a consumer.

    You're still required to do "channels.auth.login" if you want to actually log
    in - or you can hide all the routes behind an APISIX route that passes auth,
    and this will just reauth the user every time.
    """

    def decode_scope_header(self, scope, decode_header):
        """
        Decode a header in the connection scope.

        (It's decode_x_header but for channels.)

        Args:
            scope (dict): the connection scope
            decode_header (str): the name of the header to decode
        Returns:
        dict of decoded values, or None if the header isn't found
        """

        headers = {}

        for header in scope.get("headers", []):
            log.debug("decode_scope_header: Header %s", header)
            headers[header[0].decode()] = header[1]

        if decode_header not in headers:
            log.error("decode_scope_header: Header %s not found", decode_header)
            return None

        decoded_x_header = base64.b64decode(headers[decode_header])
        return json.loads(decoded_x_header)

    @database_sync_to_async
    def get_user_from_apisix_headers(self, scope):
        """
        Get the user from the APISIX headers.

        Args:
            scope (dict): the connection scope
        Returns:
        User object, or AnonymousUser if the user is not found
        """

        x_userinfo = self.decode_scope_header(scope, "x-userinfo")

        if not x_userinfo:
            log.debug("get_user_from_apisix_headers: No x-userinfo header found")
            from django.contrib.auth.models import AnonymousUser

            return AnonymousUser

        log.debug("decoded x_userinfo: %s", x_userinfo)

        preferred_username = x_userinfo.get("preferred_username", "")
        email = x_userinfo.get("email", "")
        name = x_userinfo.get("name", "")
        sub = x_userinfo.get("sub", "")

        User = get_user_model()

        user, created = User.objects.filter(global_id=sub).get_or_create(
            defaults={
                "username": preferred_username,
                "email": email,
                "name": name,
                "global_id": sub,
                "is_active": True,
            }
        )

        if created:
            log.debug(
                "get_user_from_apisix_headers: User %s not found, created new",
                preferred_username,
            )
            user.set_unusable_password()
            user.save()
        else:
            log.debug(
                "get_user_from_apisix_headers: Found existing user for %s: %s",
                preferred_username,
                user,
            )
            user.save()

        return user

    def __init__(self, app):
        """Init the class."""

        self.app = app

    async def __call__(self, scope, receive, send):
        """
        Check the request for an authenticated user, or authenticate using the
        APISIX data if there isn't one.
        """

        user = await self.get_user_from_apisix_headers(scope)

        log.debug("ApisixChannelAuthMiddleware: Got user %s", user)

        if user.is_anonymous or not user.is_active:
            log.debug("ApisixChannelAuthMiddleware: User is not active or is anonymous")
            from django.contrib.auth.models import AnonymousUser

            scope["user"] = AnonymousUser
        else:
            scope["user"] = user

        return await self.app(scope, receive, send)
