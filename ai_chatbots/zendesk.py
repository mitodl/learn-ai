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
    ZENDESK_UNVERIFIED_SUBJECT,
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
# Asked for on the mint request, and used as the TTL when the response leaves
# expires_in out - Zendesk documents that field as not mandatory. Zendesk takes
# 300 to 172800 seconds here and rejects anything outside it.
ZENDESK_TOKEN_LIFETIME = 1800
# Failures that need a connection already up and the request already going out,
# so Zendesk may have taken the ticket before the call broke. A connect or pool
# failure is not here: nothing was ever sent, so nothing can have been filed.
SENT_BUT_UNANSWERED = (
    httpx.WriteTimeout,
    httpx.WriteError,
    httpx.ReadTimeout,
    httpx.ReadError,
    httpx.RemoteProtocolError,
)
# Mint a token, POST the ticket, re-mint after a 401, POST again.
WORST_PATH_REQUESTS = 4
# A scalar httpx timeout is not a budget for the request: it is given to each of
# connect, write, read and pool separately, so one request can spend four of them.
HTTPX_TIMEOUT_PHASES = 4

# Why the filing failed, for a caller deciding whether a retry is worth offering.
# Zendesk turned the ticket down and will turn the same one down again.
TICKET_REJECTED = "rejected"
# The ticket never landed for a reason that may not hold next time.
TICKET_UNSENT = "unsent"
# The address would not reach the learner, so a different one is needed.
TICKET_BAD_EMAIL = "bad_email"
# What Zendesk answers when the ticket, not the caller or the URL, is the
# problem: malformed, too large, or rejected by its own validation.
REJECTION_STATUSES = frozenset(
    {
        HTTPStatus.BAD_REQUEST,
        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        HTTPStatus.UNPROCESSABLE_ENTITY,
    }
)


class ZendeskTokenError(Exception):
    """Minting the access token failed, so no ticket was ever posted."""


def max_filing_seconds() -> int:
    """
    Longest a file_support_ticket call can run before every request has timed
    out. Read at call time so it tracks a changed REQUESTS_TIMEOUT.
    """
    return settings.REQUESTS_TIMEOUT * HTTPX_TIMEOUT_PHASES * WORST_PATH_REQUESTS


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
    return isinstance(exc, SENT_BUT_UNANSWERED)


def _is_rejection(exc: Exception) -> bool:
    """
    Say whether Zendesk turned the ticket down on its own merits, so sending
    the same one again gets the same answer.

    Listed rather than taken as the whole 4xx range: a 401, 403 or 404 is our
    credentials or our API URL, and a thread blocked on one would still refuse
    to file after someone had fixed it.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in REJECTION_STATUSES


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
                "expires_in": ZENDESK_TOKEN_LIFETIME,
            },
            timeout=settings.REQUESTS_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        # Zendesk does not have to send expires_in back. Falling back to the
        # lifetime we asked for keeps a filing from failing on a missing key.
        lifetime = data.get("expires_in") or ZENDESK_TOKEN_LIFETIME
        ttl = max(lifetime - ZENDESK_TOKEN_EXPIRY_MARGIN, 1)
        token = data["access_token"]
    except Exception as exc:
        # Wrapped so the 5xx or timeout from here cannot be mistaken for one
        # from the ticket POST, which may have filed in spite of it.
        raise ZendeskTokenError from exc
    await cache.aset(ZENDESK_TOKEN_CACHE_KEY, token, ttl)
    return token


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
    "uncertain": ..., "reason": ...}, where uncertain means Zendesk may have
    created the ticket regardless, so the caller must neither retry nor claim
    it failed, and reason says whether retrying could ever work.
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
            "reason": TICKET_BAD_EMAIL,
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
            # Neither the subject nor the body is the requester's to put in a
            # notification email until something has proved the address is
            # theirs. Agents still read both on the ticket.
            "subject": subject if email_verified else ZENDESK_UNVERIFIED_SUBJECT,
            "comment": {"body": body, "public": email_verified},
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
            "reason": TICKET_REJECTED if _is_rejection(exc) else TICKET_UNSENT,
        }

    try:
        reference = response.json()["ticket"]["id"]
    except (ValueError, KeyError, TypeError):
        # raise_for_status already passed, so the ticket exists and only its id
        # is lost. Reported as uncertain because a retry would file a second one.
        # No exc_info, for the same reason as above: frame locals hold the
        # learner's address and their problem description.
        log.error(  # noqa: TRY400
            "Zendesk accepted the ticket (status %s) but the response carried no "
            "ticket id. Search Zendesk for external_id %s.",
            response.status_code,
            thread_id or "n/a",
        )
        return {
            "error": "Could not file the support request. Please try again later.",
            "uncertain": True,
            "reason": TICKET_UNSENT,
        }
    return {"reference": reference, "stub": False}
