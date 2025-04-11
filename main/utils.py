"""main utilities"""

import base64
import datetime
import json
import logging
from enum import Flag, auto
from functools import wraps
from itertools import islice

from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils.encoding import force_str
from django.utils.http import urlsafe_base64_decode
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


def decode_x_header(request, header):
    """
    Decode an 'X-' header.

    For things that put some JSON-encoded data in a HTTP header, this will both
    base64 decode it and then JSON decode it, and return the resulting dict.
    (This is used for the APISIX code - it puts user data in X-User-Info in
    this format.)

    Args:
        request (HttpRequest): the HTTP request
        header (str): the name of the header to decode
    Returns:
    dict of decoded values, or None if the header isn't found
    """
    x_userinfo = request.META.get(header, False)

    if not x_userinfo:
        return None

    decoded_x_userinfo = base64.b64decode(x_userinfo)
    return json.loads(decoded_x_userinfo)


def decode_apisix_headers(request, model="users_user"):
    """
    Decode the APISIX-specific headers.

    APISIX delivers user information via the X-User-Info header that it
    attaches to the request. This data can contain an arbitrary amount of
    information, so this returns just the data that we care about, normalized
    into a structure we expect (or rather ones that match Django objects).

    This mapping can be adjusted by changing the APISIX_USERDATA_MAP setting.
    This is a nested dict: the top level is the model that the mapping belongs
    to, and it is set to a dict of the mappings of model field names to APISIX
    field names. Model names are in app_model form (like the table name).

    Args:
    - request (Request): the current HTTP request object
    - model (string): the model data to retrieve (defaults to "auth_user")

    Returns: dict of applicable data or None if no data
    """

    if model not in settings.APISIX_USERDATA_MAP:
        error = "Model %s is invalid"
        raise ValueError(error, model)

    data_mapping = settings.APISIX_USERDATA_MAP[model]

    try:
        apisix_result = decode_x_header(request, "HTTP_X_USERINFO")
        if not apisix_result:
            log.debug(
                "decode_apisix_headers: No APISIX-specific header found",
            )
            return None
    except json.JSONDecodeError:
        log.debug(
            "decode_apisix_headers: Got bad APISIX-specific header: %s",
            request.META.get("HTTP_X_USERINFO", ""),
        )

        return None

    log.info("decode_apisix_headers: Got %s", apisix_result)

    return {
        modelKey: apisix_result[data_mapping[modelKey]]
        for modelKey in data_mapping
        if data_mapping[modelKey] in apisix_result
    }


def get_user_from_apisix_headers(request):
    """Get a user based on the APISIX headers."""

    decoded_headers = decode_apisix_headers(request)
    User = get_user_model()

    if not decoded_headers:
        return None

    global_id = decoded_headers.get("global_id", None)

    log.info("get_user_from_apisix_headers: Authenticating %s", global_id)

    user, created = User.objects.filter(global_id=global_id).get_or_create(
        defaults={
            "global_id": global_id,
            "username": decoded_headers.get("username", ""),
            "email": decoded_headers.get("email", ""),
            "name": decoded_headers.get("name", ""),
        }
    )

    if created:
        log.info(
            "get_user_from_apisix_headers: User %s not found, created new",
            global_id,
        )
        user.set_unusable_password()
        user.save()
    else:
        log.info(
            "get_user_from_apisix_headers: Found existing user for %s: %s",
            global_id,
            user,
        )

        user.name = decoded_headers.get("name", "")
        user.save()

    profile_data = decode_apisix_headers(request, "users_userprofile")

    if profile_data:
        log.info(
            "get_user_from_apisix_headers: Setting up additional profile for %s",
            global_id,
        )

        _, profile = user.profile.filter(user=user).get_or_create(defaults=profile_data)
        profile.save()
        user.refresh_from_db()

    return user


def decode_value(encoded_value: str) -> str:
    """Decode a base64-encoded value with proper padding."""
    PAD_AMT = 4
    if not encoded_value:
        return ""
    padding = PAD_AMT - (len(encoded_value) % PAD_AMT)
    if padding != PAD_AMT:
        encoded_value += "=" * padding
    return force_str(urlsafe_base64_decode(encoded_value))


def format_seconds(seconds):
    """Format seconds into the most appropriate time unit"""
    td = datetime.timedelta(seconds=seconds)

    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if days > 0:
        return f"{days}d{f' {hours}h' if hours else ''}"
    elif hours > 0:
        return f"{hours}h{f' {minutes}m' if minutes else ''}"
    elif minutes > 0:
        return f"{minutes}m{f' {seconds}s' if seconds else ''}"
    else:
        return f"{seconds}s"
