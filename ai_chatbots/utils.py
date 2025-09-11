"""Utility functions for ai chat agents"""

import logging
from enum import Enum
from uuid import UUID, uuid5

import requests
from django.conf import settings
from django.core.cache import BaseCache, caches
from named_enum import ExtendedEnum

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


def generate_message_id(thread_id: str, output_id: int, message: dict) -> str:
    """Generate a unique ID for a message based on its content and thread ID"""
    return str(
        uuid5(UUID(thread_id), f'{output_id}_{message["type"]}_{message["content"]}')
    )


def add_message_ids(thread_id, output_id: int, messages: list[dict]) -> list[dict]:
    """Add unique IDs to messages that don't have one"""
    for message in messages:
        # Handle both dict and LangChain message objects
        if isinstance(message, dict):
            if not message.get("id"):
                message["id"] = generate_message_id(thread_id, output_id, message)
        # # LangChain message object
        elif not hasattr(message, "id") or not message.id:
            message.id = generate_message_id(thread_id, output_id, message.__dict__)
    return messages
