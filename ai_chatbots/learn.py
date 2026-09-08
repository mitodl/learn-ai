"""
Client for the MIT Learn API.

One function per Learn endpoint. Each owns its URL setting, params, limit and
timeout, so callers never build a Learn request by hand. Every function raises
LearnError on transport failure, non-2xx after retries, or a non-JSON body.
"""

import logging
from typing import NamedTuple

from asgiref.sync import sync_to_async
from django.conf import settings
from httpx import HTTPError

from ai_chatbots.constants import HYBRID_SEARCH_FEATURE_FLAG
from ai_chatbots.utils import async_request
from main.features import is_enabled

log = logging.getLogger(__name__)


class LearnError(Exception):
    """A request to the Learn API failed after retries, or returned a non-JSON body."""

    def __init__(self, message: str, url: str, params: dict):
        super().__init__(message)
        self.url = url
        self.params = params


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
        raise LearnError(str(error), url, params) from error


async def _hybrid_params() -> dict:
    enabled = await sync_to_async(is_enabled)(HYBRID_SEARCH_FEATURE_FLAG)
    return {"hybrid_search": True} if enabled else {}


async def _search(url: str, params: dict) -> SearchResult:
    params = {**params, **await _hybrid_params()}
    body = await _get(url, params)
    return SearchResult(body.get("results") or [], url, params)


SEARCH_BACKENDS = {
    "vector": lambda: settings.AI_MIT_SEARCH_VECTOR_URL,
    "elastic": lambda: settings.AI_MIT_SEARCH_ELASTIC_URL,
}


async def search_learning_resources(
    q: str, filters: dict, *, backend: str = "vector"
) -> SearchResult:
    """
    Search learning resources on the vector or elastic endpoint. filters are the
    optional search params (resource_type, free, offered_by, certification)
    with None values dropped.
    """
    if backend not in SEARCH_BACKENDS:
        msg = f"Unknown search backend {backend!r}, expected {sorted(SEARCH_BACKENDS)}"
        raise ValueError(msg)
    result = await _search(
        SEARCH_BACKENDS[backend](),
        {"q": q, "limit": settings.AI_MIT_SEARCH_LIMIT, **filters},
    )
    # The vector endpoint can ignore `limit` and return more rows than asked
    return result._replace(results=result.results[: settings.AI_MIT_SEARCH_LIMIT])


async def search_content_files(
    q: str,
    resource_readable_id: str | list[str],
    *,
    collection_name: str | None = None,
    platform: str | None = None,
) -> SearchResult:
    """Search content file chunks of one course, or of several related courses."""
    params = {
        "q": q,
        "resource_readable_id": resource_readable_id,
        "limit": settings.AI_MIT_CONTENT_SEARCH_LIMIT,
    }
    if collection_name:
        params["collection_name"] = collection_name
    if platform:
        params["platform"] = [platform]
    return await _search(settings.AI_MIT_SYLLABUS_URL, params)


async def search_transcript(q: str, edx_module_id: str) -> SearchResult:
    """Search the transcript chunks of one video."""
    return await _search(
        settings.AI_MIT_VIDEO_TRANSCRIPT_URL,
        {
            "q": q,
            "edx_module_id": edx_module_id,
            "limit": settings.AI_MIT_TRANSCRIPT_SEARCH_LIMIT,
        },
    )


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


async def contentfiles(
    edx_module_ids: list[str], *, timeout: int | None = None
) -> list[dict]:
    """Return the contentfiles for these edX module ids."""
    body = await _get(
        settings.AI_MIT_CONTENTFILE_URL,
        {"edx_module_id": edx_module_ids},
        timeout=timeout,
    )
    return body.get("results") or []


async def problem_set(
    run_readable_id: str, title: str | None = None, *, timeout: int | None = None
) -> dict:
    """
    Return the tutor problem sets of a run, or one problem set by title.

    Learn redirects any other form of this URL to the trailing-slash one, and
    the transport does not follow redirects, so the URL is built canonically.
    """
    path = "/".join(part for part in (run_readable_id, title) if part)
    url = f"{settings.PROBLEM_SET_URL.rstrip('/')}/{path}/"
    return await _get(url, {}, timeout=timeout)
