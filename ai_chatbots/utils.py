"""Utility functions for ai chat agents"""

import logging
from enum import Enum

import requests
from django.conf import settings
from django.core.cache import BaseCache, caches
from named_enum import ExtendedEnum
from requests_toolbelt.multipart.encoder import uuid4

log = logging.getLogger(__name__)


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


def request_with_token(url, params, timeout: int = 30):
    return requests.get(
        url,
        params=params,
        headers={"Authorization": f"Bearer {settings.LEARN_ACCESS_TOKEN}"},
        timeout=timeout,
    )


def add_message_ids(messages):
    """Add unique IDs to messages that don't have one"""
    for message in messages:
        # Handle both dict and LangChain message objects
        if isinstance(message, dict):
            # Dictionary format (e.g., from JSON)
            if not message.get("id"):
                message["id"] = str(uuid4())
        # LangChain message object
        elif not hasattr(message, "id") or not message.id:
            message.id = str(uuid4())
    return messages
