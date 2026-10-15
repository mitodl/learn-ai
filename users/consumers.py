import json
import logging
import unicodedata
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


def is_allowed_login_redirect(url: str) -> bool:
    """
    Check that a login ``next`` URL is an absolute https URL on an allowed host.

    Hosts must match LOGIN_REDIRECT_ALLOWED_HOSTS exactly: no other port, user
    info or subdomain.
    """
    if any(unicodedata.category(char).startswith("C") for char in url):
        return False
    if url != url.strip():
        return False
    return url_has_allowed_host_and_scheme(
        url,
        allowed_hosts=set(settings.LOGIN_REDIRECT_ALLOWED_HOSTS),
        require_https=True,
    ) and bool(urlsplit(url).netloc)


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

        next_urls = query["next"]
        if len(next_urls) == 1 and is_allowed_login_redirect(next_urls[0]):
            await self.send_response(
                302,
                b"",
                headers=[
                    (b"Location", iri_to_uri(next_urls[0]).encode("ascii")),
                    (b"Cache-Control", b"no-store"),
                ],
            )
            return

        if len(next_urls) != 1:
            log.warning("Refused login redirect: %d next URLs", len(next_urls))
        else:
            # Log the host only: the full URL may carry an OAuth state
            try:
                host = urlsplit(next_urls[0]).netloc
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
