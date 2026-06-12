"""Utils tests"""

import datetime
import json
from base64 import b64encode
from math import ceil
from uuid import uuid4

import pytest
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser

from main.utils import (
    chunks,
    get_user_from_apisix_headers,
    is_near_now,
    now_in_utc,
)

User = get_user_model()


def _apisix_request(mocker, user_info, request_user):
    """Build a mock request carrying base64 APISIX user info and a request.user."""
    return mocker.Mock(
        META={"HTTP_X_USERINFO": b64encode(json.dumps(user_info).encode())},
        user=request_user,
    )


@pytest.fixture
def apisix_user_info():
    """Sample APISIX X-User-Info payload."""
    return {
        "sub": str(uuid4()),
        "email": "apisix_user@example.edu",
        "preferred_username": "apisix_user",
        "name": "APISIX User",
    }


def test_now_in_utc():
    """now_in_utc() should return the current time set to the UTC time zone"""
    now = now_in_utc()
    assert is_near_now(now)
    assert now.tzinfo == datetime.UTC


def test_is_near_now():
    """
    Test is_near_now for now
    """
    now = datetime.datetime.now(tz=datetime.UTC)
    assert is_near_now(now) is True
    later = now + datetime.timedelta(0, 6)
    assert is_near_now(later) is False
    earlier = now - datetime.timedelta(0, 6)
    assert is_near_now(earlier) is False


def test_chunks():
    """
    Test for chunks
    """
    input_list = list(range(113))
    output_list = []
    for nums in chunks(input_list):
        output_list += nums
    assert output_list == input_list

    output_list = []
    for nums in chunks(input_list, chunk_size=1):
        output_list += nums
    assert output_list == input_list

    output_list = []
    for nums in chunks(input_list, chunk_size=124):
        output_list += nums
    assert output_list == input_list


def test_chunks_iterable():
    """
    Test that chunks works on non-list iterables too
    """
    count = 113
    input_range = range(count)
    chunk_output = []
    for chunk in chunks(input_range, chunk_size=10):
        chunk_output.append(chunk)  # noqa: PERF402
    assert len(chunk_output) == ceil(113 / 10)

    range_list = []
    for chunk in chunk_output:
        range_list += chunk
    assert range_list == list(range(count))


@pytest.mark.django_db
def test_apisix_headers_create_user(mocker, apisix_user_info):
    """A first-time APISIX request creates the user from header data."""
    request = _apisix_request(mocker, apisix_user_info, AnonymousUser())

    user = get_user_from_apisix_headers(request)

    assert user is not None
    assert user.global_id == apisix_user_info["sub"]
    assert user.email == apisix_user_info["email"]
    assert user.username == apisix_user_info["preferred_username"]
    assert user.name == apisix_user_info["name"]


@pytest.mark.django_db
def test_apisix_headers_unchanged_skips_save(mocker, apisix_user_info):
    """A repeat request with unchanged identity issues no User save."""
    get_user_from_apisix_headers(
        _apisix_request(mocker, apisix_user_info, AnonymousUser())
    )

    mock_save = mocker.patch.object(User, "save")
    user = get_user_from_apisix_headers(
        _apisix_request(mocker, apisix_user_info, AnonymousUser())
    )

    assert user.global_id == apisix_user_info["sub"]
    mock_save.assert_not_called()


@pytest.mark.django_db
def test_apisix_headers_changed_field_updates_user(mocker, apisix_user_info):
    """A changed header field updates and persists the User."""
    get_user_from_apisix_headers(
        _apisix_request(mocker, apisix_user_info, AnonymousUser())
    )

    changed = {**apisix_user_info, "name": "Changed Name"}
    user = get_user_from_apisix_headers(
        _apisix_request(mocker, changed, AnonymousUser())
    )

    assert user.name == "Changed Name"
    user.refresh_from_db()
    assert user.name == "Changed Name"


@pytest.mark.django_db
def test_apisix_headers_change_bumps_updated_on(mocker, apisix_user_info):
    """A user-field change advances updated_on."""
    user = get_user_from_apisix_headers(
        _apisix_request(mocker, apisix_user_info, AnonymousUser())
    )
    User.objects.filter(pk=user.pk).update(
        updated_on=user.updated_on - datetime.timedelta(days=1)
    )
    original_updated_on = User.objects.get(pk=user.pk).updated_on

    changed = {**apisix_user_info, "name": "Changed Name"}
    get_user_from_apisix_headers(_apisix_request(mocker, changed, AnonymousUser()))

    assert User.objects.get(pk=user.pk).updated_on > original_updated_on


@pytest.mark.django_db
def test_apisix_headers_ambiguous_identity_fails_closed(mocker, apisix_user_info):
    """An ambiguous global_id match returns None rather than guessing a user."""
    from main.factories import UserFactory

    UserFactory.create(global_id=apisix_user_info["sub"])
    UserFactory.create(global_id=apisix_user_info["sub"])

    request = _apisix_request(mocker, apisix_user_info, AnonymousUser())

    assert get_user_from_apisix_headers(request) is None
