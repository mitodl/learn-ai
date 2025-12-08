"""Utility functions for ai chat agents"""

import asyncio
import logging
from enum import Enum

import httpx
from django.conf import settings
from django.core.cache import BaseCache, caches
from named_enum import ExtendedEnum

log = logging.getLogger(__name__)

# Message type identifiers for LangChain serialized messages
TOOL_MESSAGE_ID = ["langchain", "schema", "messages", "ToolMessage"]
AI_MESSAGE_ID = ["langchain", "schema", "messages", "AIMessage"]


class HTTPClientManager:
    """Manager for shared HTTP client instances with connection pooling."""

    def __init__(self):
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


def request_with_token(url, params, timeout: int = 30):
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
    )


async def async_request_with_token(url, params, timeout: int = 30):
    """
    Make an asynchronous HTTP GET request with bearer token authentication.

    This function should be used in async contexts (like LangGraph tools) to avoid
    blocking the event loop with synchronous HTTP requests. Uses a shared client
    with connection pooling for optimal performance with concurrent requests.

    Args:
        url: The URL to request
        params: Query parameters
        timeout: Request timeout in seconds

    Returns:
        httpx.Response object with compatible interface to requests.Response
    """
    client = get_async_http_client()
    return await client.get(
        url,
        params=params,
        headers={"Authorization": f"Bearer {settings.LEARN_ACCESS_TOKEN}"},
        timeout=timeout,
    )


def collect_answered_tool_call_ids(messages: list[dict]) -> set[str]:
    """Collect tool_call_ids from ToolMessages in serialized checkpoint data."""
    answered_tool_calls = set()
    for msg_dict in messages:
        if not isinstance(msg_dict, dict) or msg_dict.get("id") != TOOL_MESSAGE_ID:
            continue
        tool_call_id = msg_dict.get("kwargs", {}).get("tool_call_id")
        if tool_call_id:
            answered_tool_calls.add(tool_call_id)
    return answered_tool_calls


def filter_orphaned_tool_calls(
    messages: list[dict], answered_tool_calls: set[str]
) -> bool:
    """
    Filter orphaned tool calls from AIMessages in-place.

    Returns True if any modifications were made.
    """
    modified = False
    for msg_dict in messages:
        if not isinstance(msg_dict, dict) or msg_dict.get("id") != AI_MESSAGE_ID:
            continue
        if filter_tool_calls_from_message(msg_dict, answered_tool_calls):
            modified = True
    return modified


def filter_tool_calls_from_message(
    msg_dict: dict, answered_tool_calls: set[str]
) -> bool:
    """
    Filter tool calls from a single AIMessage dict in-place.

    Returns True if modifications were made.
    """
    kwargs = msg_dict.get("kwargs", {})
    tool_calls = kwargs.get("tool_calls", [])
    if not tool_calls:
        return False

    valid_tool_calls = [tc for tc in tool_calls if tc.get("id") in answered_tool_calls]

    if len(valid_tool_calls) == len(tool_calls):
        return False

    # Update kwargs
    if valid_tool_calls:
        kwargs["tool_calls"] = valid_tool_calls
    else:
        kwargs.pop("tool_calls", None)

    # Update additional_kwargs if present
    additional = kwargs.get("additional_kwargs", {})
    if "tool_calls" in additional:
        if valid_tool_calls:
            additional["tool_calls"] = valid_tool_calls
        else:
            additional.pop("tool_calls", None)

    return True


def remove_orphaned_tool_calls_from_checkpoint(
    checkpoint_data: dict,
) -> bool:
    """
    Remove orphaned tool calls from a checkpoint's serialized messages in-place.

    Args:
        checkpoint_data: The checkpoint data dict (already parsed from JSON if needed)

    Returns:
        True if modifications were made, False otherwise
    """
    messages = checkpoint_data.get("channel_values", {}).get("messages", [])
    answered_tool_calls = collect_answered_tool_call_ids(messages)
    return filter_orphaned_tool_calls(messages, answered_tool_calls)


def truncate_checkpoint_to_message_ids(
    checkpoint_data: dict, keep_ids: set[str]
) -> int:
    """
    Truncate a checkpoint's messages to only keep those with IDs in keep_ids.

    Modifies checkpoint_data in-place.

    Args:
        checkpoint_data: The checkpoint data dict (already parsed from JSON if needed)
        keep_ids: Set of message IDs to keep

    Returns:
        The number of messages after truncation
    """
    messages = checkpoint_data.get("channel_values", {}).get("messages", [])

    filtered_messages = [
        msg_dict
        for msg_dict in messages
        if isinstance(msg_dict, dict)
        and msg_dict.get("kwargs", {}).get("id") in keep_ids
    ]

    checkpoint_data["channel_values"]["messages"] = filtered_messages
    return len(filtered_messages)


async def remove_orphaned_tool_calls_from_db(thread_id: str) -> None:
    """
    Remove orphaned tool calls from the latest checkpoint for a thread.

    Args:
        thread_id: The thread ID to clean up
    """
    import json

    from channels.db import database_sync_to_async

    from ai_chatbots.api import DjangoCheckpoint

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

        modified = remove_orphaned_tool_calls_from_checkpoint(checkpoint_data)

        if modified:
            latest_checkpoint.checkpoint = (
                json.dumps(checkpoint_data) if is_string else checkpoint_data
            )
            latest_checkpoint.save(update_fields=["checkpoint"])
            log.info(
                "Removed orphaned tool calls from checkpoint %s",
                latest_checkpoint.id,
            )

    await update_checkpoint()


async def truncate_checkpoint_messages_in_db(
    thread_id: str, keep_ids: set[str]
) -> None:
    """
    Truncate the latest checkpoint for a thread to only keep specified message IDs.

    Args:
        thread_id: The thread ID to clean up
        keep_ids: Set of message IDs to keep
    """
    import json

    from channels.db import database_sync_to_async

    from ai_chatbots.api import DjangoCheckpoint

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

        num_messages = truncate_checkpoint_to_message_ids(checkpoint_data, keep_ids)

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
