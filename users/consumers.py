import json
import logging
from urllib.parse import parse_qs, urlsplit

from channels.generic.http import AsyncHttpConsumer
from django.conf import settings
from django.utils.encoding import iri_to_uri
from django.utils.http import url_has_allowed_host_and_scheme

log = logging.getLogger(__name__)


class UserMetaHttpConsumer(AsyncHttpConsumer):
    """
    Async HTTP consumer for user metadata.

    Returns the user's account info, or is_anonymous=True if they're not logged
    in. LoginHttpConsumer serves the same data on the route that requires auth.
    """

    async def handle(self, message: str):  # noqa: ARG002
        user = self.scope.get("user", None)

        if user and not user.is_anonymous:
            user = user.to_json()
        else:
            log.debug("Anon user, no session")
            user = json.dumps({"is_anonymous": True})

        await self.send_headers(
            headers=[
                (b"Cache-Control", b"no-cache"),
                (
                    b"Content-Type",
                    b"text/event-stream",
                ),
                (
                    b"Transfer-Encoding",
                    b"chunked",
                ),
                (b"Connection", b"keep-alive"),
            ]
        )
        # Headers are only sent after the first body event.
        # Set "more_body" to tell the interface server to not
        # finish the response yet:
        await self.send_chunk("")

        try:
            await self.send_chunk(user, more_body=True)
        except:  # noqa: E722
            log.exception("Error in UserMetaHttpConsumer")
        finally:
            await self.send_chunk("", more_body=False)
            await self.disconnect()

    async def send_chunk(self, chunk: str, *, more_body: bool = True):
        await self.send_body(body=chunk.encode("utf-8"), more_body=more_body)

    async def http_request(self, message):
        """
        Receives a request and holds the connection open
        until the client or server chooses to disconnect.
        """
        try:
            await self.handle(message)
        finally:
            pass


def allowed_login_redirect(url: str) -> str | None:
    """
    Return ``url`` percent-encoded if it is an absolute URL on an allowed host,
    and https unless LOGIN_REDIRECT_REQUIRE_HTTPS is off. Otherwise None.

    Hosts must match LOGIN_REDIRECT_ALLOWED_HOSTS exactly: no other port, user
    info or subdomain. The encoded URL is the one checked, so the redirect goes
    to exactly what passed.
    """
    location = iri_to_uri(url)
    if (
        url_has_allowed_host_and_scheme(
            location,
            allowed_hosts=set(settings.LOGIN_REDIRECT_ALLOWED_HOSTS),
            require_https=settings.LOGIN_REDIRECT_REQUIRE_HTTPS,
        )
        and urlsplit(location).netloc
    ):
        return location
    return None


class LoginHttpConsumer(UserMetaHttpConsumer):
    """
    The login route. APISIX signs the learner in before the request gets here.

    With a ``next`` URL on an allowed host, redirect there. Open edX's login
    passes through this so the learner is also signed in to MIT Learn. Without
    ``next``, return the user's account info, as ``user_meta`` does.
    """

    async def handle(self, message: str):
        """Redirect to an allowed ``next``, refuse any other, or return user info."""
        query = parse_qs(
            self.scope.get("query_string", b"").decode("utf-8", "replace"),
            keep_blank_values=True,
        )
        if "next" not in query:
            await super().handle(message)
            return

        next_url = query["next"][0]
        location = allowed_login_redirect(next_url)
        if location:
            await self.send_response(
                302,
                b"",
                headers=[
                    (b"Location", location.encode("ascii")),
                    (b"Cache-Control", b"no-store"),
                ],
            )
            return

        # Log the host only: the full URL may carry an OAuth state or user info
        try:
            host = urlsplit(next_url).hostname
        except ValueError:
            host = "<unparseable>"
        log.warning("Refused login redirect to host %r", host)
        await self.send_response(
            400,
            b"Invalid next URL",
            headers=[
                (b"Content-Type", b"text/plain"),
                (b"Cache-Control", b"no-store"),
            ],
        )
