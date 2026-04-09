"""Utility functions for ai chat agents"""

import asyncio
import json
import logging
from enum import Enum

import httpx
from channels.db import database_sync_to_async
from django.conf import settings
from django.core.cache import BaseCache, caches
from langchain_core.messages import HumanMessage
from named_enum import ExtendedEnum

from ai_chatbots.models import DjangoCheckpoint

log = logging.getLogger(__name__)

# Message type identifiers for LangChain serialized messages
TOOL_MESSAGE_ID = ["langchain", "schema", "messages", "ToolMessage"]
AI_MESSAGE_ID = ["langchain", "schema", "messages", "AIMessage"]

# HTTP statuses that indicate transient upstream failures worth retrying.
# Other 4xx codes are deterministic and should not be retried.
RETRYABLE_STATUS_CODES = frozenset({408, 429, 502, 503, 504})

# httpx exceptions that represent transient network/transport failures.
RETRYABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)

# Max retry attempts for transient HTTP failures (1 original + 2 retries).
_RETRY_MAX_ATTEMPTS = 3
_RETRY_BASE_DELAY = 0.5
_RETRY_MAX_DELAY = 2.0


def _compute_retry_delay(attempt: int) -> float:
    """Compute a capped exponential backoff delay in seconds."""
    return min(_RETRY_BASE_DELAY * (2**attempt), _RETRY_MAX_DELAY)


async def _sleep_before_retry(attempt: int) -> None:
    """Sleep before retrying a transient upstream failure."""
    await asyncio.sleep(_compute_retry_delay(attempt))


class HTTPClientManager:
    """Manager for shared HTTP client instances with connection pooling."""

    def __init__(self):
        """Initialize empty shared sync and async HTTP clients."""
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    @property
    def sync_client(self) -> httpx.Client:
        """Get or create the shared synchronous HTTP client."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                timeout=httpx.Timeout(settings.REQUESTS_TIMEOUT),
                limits=httpx.Limits(
                    keepalive_expiry=settings.REQUESTS_KEEPALIVE_EXPIRY,
                ),
            )
        return self._sync_client

    @property
    def async_client(self) -> httpx.AsyncClient:
        """Get or create the shared asynchronous HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.REQUESTS_TIMEOUT),
                limits=httpx.Limits(
                    keepalive_expiry=settings.REQUESTS_KEEPALIVE_EXPIRY,
                ),
            )
        return self._async_client

    def reset(self):
        """Reset client instances (primarily for testing)."""
        if self._sync_client:
            self._sync_client.close()
        self._sync_client = None
        if self._async_client:
            asyncio.run(self._async_client.aclose())
        self._async_client = None


# Module-level singleton instance
http_client_manager = HTTPClientManager()


def enum_zip(label: str, enum: ExtendedEnum) -> type[Enum]:
    """
    Create a new Enum with both name and value equal to
    the names of the given ExtendedEnum.

    Args:
        label: The label for the new Enum
        enum: The Enum to use as a basis for the new Enum

    Returns:
        A new Enum with the names of the given Enum as both name and value

    """
    return Enum(label, dict(zip(enum.names(), enum.names())))


def get_django_cache() -> BaseCache:
    """
    Get the Django redis cache if enabled, default cache otherwise.

    Returns:
        The Django cache.

    """
    if settings.CELERY_BROKER_URL:
        return caches["redis"]
    return caches["default"]


def get_sync_http_client() -> httpx.Client:
    """
    Get or create a shared synchronous HTTP client with connection pooling.

    This client is reused across requests to benefit from connection pooling,
    reducing latency and server load.

    Returns:
        Shared httpx.Client instance with connection pooling configured.
    """
    return http_client_manager.sync_client


def get_async_http_client() -> httpx.AsyncClient:
    """
    Get or create a shared asynchronous HTTP client with connection pooling.

    This client is reused across async requests to benefit from connection pooling,
    reducing latency and server load. Safe to use with multiple concurrent async tasks.

    Returns:
        Shared httpx.AsyncClient instance with connection pooling configured.
    """
    return http_client_manager.async_client


def request_with_token(url, params, follow_redirects, timeout: int = 30):
    """
    Make a synchronous HTTP GET request with bearer token authentication.

    Uses a shared httpx client with connection pooling for better performance
    across multiple requests.

    Args:
        url: The URL to request
        params: Query parameters
        timeout: Request timeout in seconds

    Returns:
        httpx.Response object with requests-compatible interface
    """
    client = get_sync_http_client()
    return client.get(
        url,
        params=params,
        headers={"Authorization": f"Bearer {settings.LEARN_ACCESS_TOKEN}"},
        timeout=timeout,
        follow_redirects=follow_redirects,
    )


async def async_request_with_token(url, params, timeout: int = 30):
    """
    Make an asynchronous HTTP GET request with bearer token authentication.

    This function should be used in async contexts (like LangGraph tools) to avoid
    blocking the event loop with synchronous HTTP requests. Uses a shared client
    with connection pooling for optimal performance with concurrent requests.

    Transient upstream failures are retried up to ``_RETRY_MAX_ATTEMPTS`` times
    with a capped exponential backoff. The following are considered transient
    and trigger a retry:

    - Network/transport exceptions in ``RETRYABLE_EXCEPTIONS`` (connect errors,
      read/write/pool timeouts, remote protocol errors).
    - HTTP responses with a status code in ``RETRYABLE_STATUS_CODES``
      (408, 429, 502, 503, 504).

    Non-retryable responses (including 2xx success and 4xx client errors other
    than 408/429) are returned to the caller as-is, preserving the existing
    contract that callers decide what to do via their own ``raise_for_status``.
    If retries are exhausted for a retryable HTTP status, the final response is
    returned to the caller. If retries are exhausted for a transport error, the
    original ``httpx.RequestError`` is reraised.

    Args:
        url: The URL to request
        params: Query parameters
        timeout: Request timeout in seconds

    Returns:
        httpx.Response object with compatible interface to requests.Response
    """
    client = get_async_http_client()
    headers = {"Authorization": f"Bearer {settings.LEARN_ACCESS_TOKEN}"}

    for attempt in range(_RETRY_MAX_ATTEMPTS):
        try:
            response = await client.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
            )
        except RETRYABLE_EXCEPTIONS:
            if attempt == _RETRY_MAX_ATTEMPTS - 1:
                raise
            delay = _compute_retry_delay(attempt)
            log.warning(
                (
                    "Retrying HTTP request to %s after transient transport "
                    "error; attempt %d/%d in %.1fs"
                ),
                url,
                attempt + 2,
                _RETRY_MAX_ATTEMPTS,
                delay,
                exc_info=True,
            )
            await _sleep_before_retry(attempt)
            continue

        if response.status_code not in RETRYABLE_STATUS_CODES:
            return response

        if attempt == _RETRY_MAX_ATTEMPTS - 1:
            return response

        delay = _compute_retry_delay(attempt)
        log.warning(
            (
                "Retrying HTTP request to %s after transient upstream status "
                "%d; attempt %d/%d in %.1fs"
            ),
            url,
            response.status_code,
            attempt + 2,
            _RETRY_MAX_ATTEMPTS,
            delay,
        )
        await _sleep_before_retry(attempt)

    return response


def truncate_to_latest_human_message(messages: list) -> list:
    """
    Truncate messages to keep only the latest HumanMessage and everything after it.

    Returns a cleaned copy of the messages list.
    """
    if not messages:
        return messages

    # Find the last HumanMessage
    last_human_index = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_index = i
            break

    if last_human_index >= 0:
        return messages[last_human_index:]
    else:
        # No HumanMessage found, return empty list
        return []


async def save_truncated_checkpoint(thread_id: str, keep_ids: set[str]) -> None:
    """
    Truncate the latest checkpoint for a thread to only keep specified message IDs.

    Args:
        thread_id: The thread ID to clean up
        keep_ids: Set of message IDs to keep
    """

    @database_sync_to_async
    def update_checkpoint():
        latest_checkpoint = (
            DjangoCheckpoint.objects.filter(thread_id=thread_id).order_by("-id").first()
        )
        if not latest_checkpoint:
            return

        checkpoint_data = latest_checkpoint.checkpoint
        is_string = isinstance(checkpoint_data, str)
        if is_string:
            checkpoint_data = json.loads(checkpoint_data)

        messages = checkpoint_data.get("channel_values", {}).get("messages", [])

        filtered_messages = [
            msg_dict
            for msg_dict in messages
            if isinstance(msg_dict, dict)
            and msg_dict.get("kwargs", {}).get("id") in keep_ids
        ]

        checkpoint_data["channel_values"]["messages"] = filtered_messages
        num_messages = len(filtered_messages)

        latest_checkpoint.checkpoint = (
            json.dumps(checkpoint_data) if is_string else checkpoint_data
        )
        latest_checkpoint.save(update_fields=["checkpoint"])
        log.info(
            "Truncated checkpoint messages to %d messages in checkpoint %s",
            num_messages,
            latest_checkpoint.id,
        )

    await update_checkpoint()
