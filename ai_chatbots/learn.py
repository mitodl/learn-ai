"""
Client for the MIT Learn API.

One function per Learn endpoint. Each owns its URL setting, params, limit and
timeout, so callers never build a Learn request by hand. Every function raises
LearnError on transport failure, non-2xx after retries, or a non-JSON body.
"""

import logging
from typing import NamedTuple

from django.conf import settings
from httpx import HTTPError

from ai_chatbots.utils import async_request

log = logging.getLogger(__name__)


class LearnError(Exception):
    """A request to the Learn API failed after retries, or returned a non-JSON body."""


class SearchResult(NamedTuple):
    """A search response plus exactly what was sent, for tool metadata."""

    results: list[dict]
    url: str
    params: dict


async def _get(url: str, params: dict, *, timeout: int | None = None) -> dict:
    try:
        response = await async_request(
            url,
            params,
            timeout=timeout or settings.REQUESTS_TIMEOUT,
            include_learn_token=bool(settings.LEARN_ACCESS_TOKEN),
        )
        response.raise_for_status()
        return response.json()
    except (HTTPError, ValueError) as error:
        log.exception("Learn API request failed: %s", url)
        raise LearnError(str(error)) from error


async def learning_resources(readable_id: str) -> list[dict]:
    """Return the learning resources with this readable id (one per platform)."""
    body = await _get(
        settings.AI_MIT_LEARNING_RESOURCES_URL,
        {"readable_id": readable_id, "limit": 5},
        # A cheap metadata lookup, not a search, and callers wait on it, so it
        # gets a tighter timeout than the default.
        timeout=settings.AI_COURSE_PLATFORM_LOOKUP_TIMEOUT,
    )
    return body.get("results") or []
