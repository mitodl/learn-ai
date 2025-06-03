"""Utility functions for ai chat agents"""

import logging
from enum import Enum

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
