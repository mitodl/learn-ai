"""Zendesk ticket creation for the support intake bot.

Kept out of tools.py on purpose: nothing here is a LangGraph tool. SupportBot
files deterministically from get_completion rather than waiting for the model to
emit a tool call, so this returns a dict rather than tools.py's JSON string.
"""

import logging
from uuid import uuid4

from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import validate_email

from ai_chatbots.constants import (
    ZENDESK_SUPPORT_INTAKE_TAG,
    ZENDESK_TICKET_PATH,
    ZENDESK_UNVERIFIED_REQUESTER_TAG,
)
from ai_chatbots.utils import get_async_http_client

log = logging.getLogger(__name__)


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
) -> dict:
    """Create a Zendesk ticket, returning {"reference": ...} or {"error": ...}."""
    if settings.AI_ZENDESK_STUB_MODE:
        # Loud on purpose: the learner is told their request was filed, and
        # nothing was. A quiet stub left on in production looks like success.
        log.warning(
            "AI_ZENDESK_STUB_MODE is on - no Zendesk ticket was created, "
            "but the learner was told one was."
        )
        return {"reference": f"STUB-{uuid4().hex[:8].upper()}", "stub": True}

    try:
        validate_email((email or "").strip())
    except ValidationError:
        return {
            "error": (
                "That does not look like a valid email address. Ask the "
                "learner for one before filing."
            )
        }

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
        response = await client.post(
            f"{settings.AI_ZENDESK_API_URL.rstrip('/')}{ZENDESK_TICKET_PATH}",
            json={"ticket": ticket_data},
            headers={"Authorization": f"Bearer {settings.AI_ZENDESK_OAUTH_TOKEN}"},
            timeout=settings.REQUESTS_TIMEOUT,
        )
        response.raise_for_status()
        return {"reference": response.json()["ticket"]["id"], "stub": False}
    except Exception as exc:  # noqa: BLE001
        # No exc_info: the traceback's frame locals hold the learner's email and
        # their problem description, and Sentry ships frame locals by default.
        log.error(  # noqa: TRY400
            "Failed to create Zendesk support ticket (%s, status %s)",
            type(exc).__name__,
            getattr(getattr(exc, "response", None), "status_code", "n/a"),
        )
        return {"error": "Could not file the support request. Please try again later."}
