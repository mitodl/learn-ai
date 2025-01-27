"""main utilities"""

import datetime
import logging
from enum import Flag, auto
from functools import wraps
from itertools import islice

from django.views.decorators.cache import cache_page

log = logging.getLogger(__name__)

# This is the Django ImageField max path size
IMAGE_PATH_MAX_LENGTH = 100


def cache_page_for_anonymous_users(*cache_args, **cache_kwargs):
    def inner_decorator(func):
        @wraps(func)
        def inner_function(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return cache_page(*cache_args, **cache_kwargs)(func)(
                    request, *args, **kwargs
                )
            return func(request, *args, **kwargs)

        return inner_function

    return inner_decorator


def cache_page_for_all_users(*cache_args, **cache_kwargs):
    def inner_decorator(func):
        @wraps(func)
        def inner_function(request, *args, **kwargs):
            return cache_page(*cache_args, **cache_kwargs)(func)(
                request, *args, **kwargs
            )

        return inner_function

    return inner_decorator


class FeatureFlag(Flag):
    """
    FeatureFlag enum

    Members should have values of increasing powers of 2 (1, 2, 4, 8, ...)

    """

    EXAMPLE_FEATURE = auto()


def is_near_now(time):
    """
    Returns true if time is within five seconds or so of now
    Args:
        time (datetime.datetime):
            The time to test
    Returns:
        bool:
            True if near now, false otherwise
    """  # noqa: D401
    now = datetime.datetime.now(tz=datetime.UTC)
    five_seconds = datetime.timedelta(0, 5)
    return now - five_seconds < time < now + five_seconds


def now_in_utc():
    """
    Get the current time in UTC
    Returns:
        datetime.datetime: A datetime object for the current time
    """
    return datetime.datetime.now(tz=datetime.UTC)


def chunks(iterable, *, chunk_size=20):
    """
    Yields chunks of an iterable as sub lists each of max size chunk_size.

    Args:
        iterable (iterable): iterable of elements to chunk
        chunk_size (int): Max size of each sublist

    Yields:
        list: List containing a slice of list_to_chunk
    """  # noqa: D401
    chunk_size = max(1, chunk_size)
    iterable = iter(iterable)
    chunk = list(islice(iterable, chunk_size))

    while len(chunk) > 0:
        yield chunk
        chunk = list(islice(iterable, chunk_size))


class Singleton(type):
    """
    Ensure only instance of singleton class is created
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
