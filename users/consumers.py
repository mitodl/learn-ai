import json
import logging

from channels.generic.http import AsyncHttpConsumer

log = logging.getLogger(__name__)


class UserMetaHttpConsumer(AsyncHttpConsumer):
    """
    Async HTTP consumer for user metadata.

    Returns the user's account info, or is_anonymous=True if they're not logged
    in. This has two routes, one for just retrieving the data and one that should
    be set to require auth for a "login".
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
