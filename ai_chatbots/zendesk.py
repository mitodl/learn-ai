"""Zendesk ticket creation for the support intake bot.

Kept out of tools.py on purpose: nothing here is a LangGraph tool. SupportBot
files deterministically from get_completion rather than waiting for the model to
emit a tool call, so this returns a dict rather than tools.py's JSON string.
"""

import logging
from http import HTTPStatus
from uuid import uuid4

import httpx
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import validate_email

from ai_chatbots.constants import (
    ZENDESK_OAUTH_TOKEN_PATH,
    ZENDESK_SUPPORT_INTAKE_TAG,
    ZENDESK_TICKET_PATH,
    ZENDESK_UNVERIFIED_REQUESTER_TAG,
)
from ai_chatbots.utils import get_async_http_client, get_django_cache

log = logging.getLogger(__name__)

ZENDESK_TOKEN_CACHE_KEY = "zendesk_oauth_access_token"  # noqa: S105
# client_credentials needs the scopes it will exercise: write the ticket, and
# write the requester user Zendesk creates from the ticket's requester email.
ZENDESK_OAUTH_SCOPE = "tickets:write users:write"
# Re-mint this many seconds before the token's stated expiry, so an in-flight
# request never files with a token that lapses mid-call.
ZENDESK_TOKEN_EXPIRY_MARGIN = 60
# Timeouts that mean the request never left, so no ticket can have been made.
UNDELIVERED_TIMEOUTS = (httpx.ConnectTimeout, httpx.PoolTimeout)


class ZendeskTokenError(Exception):
    """Minting the access token failed, so no ticket was ever posted."""


def _outcome_is_unknown(exc: Exception) -> bool:
    """
    Say whether Zendesk may have created the ticket despite the failure.

    Callers must not report a definite failure here: a retry would duplicate
    the ticket, and telling the learner nothing was filed may be untrue.

    Only positive evidence counts, so anything unrecognised - a wrapped mint
    failure included - is read as nothing having been filed.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        # Zendesk can create the ticket and still fail the response.
        return status >= HTTPStatus.INTERNAL_SERVER_ERROR
    return isinstance(exc, httpx.TimeoutException) and not isinstance(
        exc, UNDELIVERED_TIMEOUTS
    )


async def _get_access_token(client, *, force_refresh: bool = False) -> str:
    """
    Return a Zendesk access token, minting and caching one when needed.

    client_credentials tokens are short-lived and carry no refresh token, so the
    token is cached until just before it expires and re-minted on demand.
    """
    cache = get_django_cache()
    if not force_refresh:
        token = await cache.aget(ZENDESK_TOKEN_CACHE_KEY)
        if token:
            return token

    try:
        response = await client.post(
            f"{settings.AI_ZENDESK_API_URL.rstrip('/')}{ZENDESK_OAUTH_TOKEN_PATH}",
            json={
                "grant_type": "client_credentials",
                "client_id": settings.AI_ZENDESK_OAUTH_CLIENT_ID,
                "client_secret": settings.AI_ZENDESK_OAUTH_CLIENT_SECRET,
                "scope": ZENDESK_OAUTH_SCOPE,
            },
            timeout=settings.REQUESTS_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        # Wrapped so the 5xx or timeout from here cannot be mistaken for one
        # from the ticket POST, which may have filed in spite of it.
        raise ZendeskTokenError from exc
    ttl = max(data["expires_in"] - ZENDESK_TOKEN_EXPIRY_MARGIN, 1)
    await cache.aset(ZENDESK_TOKEN_CACHE_KEY, data["access_token"], ttl)
    return data["access_token"]


async def _post_ticket(
    client, ticket_data: dict, *, idempotency_key: str = "", force_refresh: bool = False
):
    """POST the ticket with a (cached or freshly minted) access token."""
    token = await _get_access_token(client, force_refresh=force_refresh)
    headers = {"Authorization": f"Bearer {token}"}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    return await client.post(
        f"{settings.AI_ZENDESK_API_URL.rstrip('/')}{ZENDESK_TICKET_PATH}",
        json={"ticket": ticket_data},
        headers=headers,
        timeout=settings.REQUESTS_TIMEOUT,
    )


def _custom_fields(page_url: str, name: str, username: str) -> list[dict]:
    """
    Pair each value with the field id configured for this Zendesk instance.

    Field ids are minted per instance, so an environment pointed at a different
    Zendesk has different ones and a fresh one has none at all. An unconfigured
    id is skipped rather than guessed: Zendesk rejects the whole ticket for an
    id it does not recognise.
    """
    return [
        {"id": field_id, "value": value}
        for field_id, value in (
            (settings.AI_ZENDESK_URL_FIELD_ID, page_url),
            (settings.AI_ZENDESK_NAME_FIELD_ID, name),
            (settings.AI_ZENDESK_USERNAME_FIELD_ID, username),
        )
        if field_id and value
    ]


async def file_support_ticket(  # noqa: PLR0913
    *,
    subject: str,
    description: str,
    email: str,
    page_url: str = "",
    name: str = "",
    username: str = "",
    email_verified: bool = False,
    thread_id: str = "",
) -> dict:
    """
    Create a Zendesk ticket.

    Returns {"reference": ...} on success. On failure, {"error": ...,
    "uncertain": ...}, where uncertain means Zendesk may have created the
    ticket regardless, so the caller must neither retry nor claim it failed.
    """
    # Validated ahead of the stub return: a stub that accepts what Zendesk
    # rejects hides the bad address until the first deployed environment.
    try:
        validate_email((email or "").strip())
    except ValidationError:
        return {
            "error": (
                "That does not look like a valid email address. Ask the "
                "learner for one before filing."
            ),
            "uncertain": False,
        }

    if settings.AI_ZENDESK_STUB_MODE:
        # Loud on purpose: the learner is told their request was filed, and
        # nothing was. A quiet stub left on in production looks like success.
        log.warning(
            "AI_ZENDESK_STUB_MODE is on - no Zendesk ticket was created, "
            "but the learner was told one was."
        )
        return {"reference": f"STUB-{uuid4().hex[:8].upper()}", "stub": True}

    body = f"{description}\n\nPage: {page_url}" if page_url else description
    tags = [ZENDESK_SUPPORT_INTAKE_TAG]
    if not email_verified:
        tags.append(ZENDESK_UNVERIFIED_REQUESTER_TAG)

    custom_fields = _custom_fields(page_url, name, username)

    try:
        client = get_async_http_client()
        ticket_data = {
            "subject": subject,
            "comment": {"body": body},
            "requester": {"email": email.strip(), "name": name or email.strip()},
            "tags": tags,
        }
        if custom_fields:
            ticket_data["custom_fields"] = custom_fields
        if thread_id:
            # Zendesk de-duplicates on the header; external_id is what lets a
            # human find the pair on an instance that does not honour it.
            ticket_data["external_id"] = thread_id
        response = await _post_ticket(client, ticket_data, idempotency_key=thread_id)
        if response.status_code == HTTPStatus.UNAUTHORIZED:
            # The cached token can be revoked before its TTL; re-mint once.
            response = await _post_ticket(
                client, ticket_data, idempotency_key=thread_id, force_refresh=True
            )
        response.raise_for_status()
        return {"reference": response.json()["ticket"]["id"], "stub": False}
    except Exception as exc:  # noqa: BLE001
        uncertain = _outcome_is_unknown(exc)
        # No exc_info: the traceback's frame locals hold the learner's email and
        # their problem description, and Sentry ships frame locals by default.
        log.error(  # noqa: TRY400
            "Failed to create Zendesk support ticket (%s, status %s). May have "
            "been filed anyway: %s. Search Zendesk for external_id %s.",
            type(exc).__name__,
            getattr(getattr(exc, "response", None), "status_code", "n/a"),
            uncertain,
            thread_id or "n/a",
        )
        return {
            "error": "Could not file the support request. Please try again later.",
            "uncertain": uncertain,
        }
