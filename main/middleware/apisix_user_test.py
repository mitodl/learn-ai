"""Tests for ApisixUserMiddleware."""

import json
from base64 import b64encode
from uuid import uuid4

import pytest
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser

from main.factories import UserFactory
from main.middleware.apisix_user import ApisixUserMiddleware

User = get_user_model()


@pytest.fixture
def apisix_user_info():
    """Sample APISIX X-User-Info payload."""
    return {
        "sub": str(uuid4()),
        "email": "apisix_user@example.edu",
        "preferred_username": "apisix_user",
        "name": "APISIX User",
    }


def _request(mocker, apisix_user_info, request_user):
    """Build a mock request carrying base64 APISIX user info and a request.user."""
    return mocker.Mock(
        META={"HTTP_X_USERINFO": b64encode(json.dumps(apisix_user_info).encode())},
        user=request_user,
    )


@pytest.mark.django_db
def test_anonymous_request_logs_in(mocker, apisix_user_info):
    """An anonymous request with valid headers logs the APISIX user in."""
    mock_login = mocker.patch("main.middleware.apisix_user.login")
    request = _request(mocker, apisix_user_info, AnonymousUser())

    ApisixUserMiddleware(mocker.Mock()).process_request(request)

    assert request.user.global_id == apisix_user_info["sub"]
    mock_login.assert_called_once()


@pytest.mark.django_db
def test_same_user_skips_login(mocker, apisix_user_info):
    """A repeat request as the already-authenticated user does not call login()."""
    existing = UserFactory.create(
        global_id=apisix_user_info["sub"],
        email=apisix_user_info["email"],
        username=apisix_user_info["preferred_username"],
        name=apisix_user_info["name"],
    )
    mock_login = mocker.patch("main.middleware.apisix_user.login")
    request = _request(mocker, apisix_user_info, existing)

    ApisixUserMiddleware(mocker.Mock()).process_request(request)

    assert request.user == existing
    mock_login.assert_not_called()


@pytest.mark.django_db
def test_different_user_logs_out_then_in(mocker, apisix_user_info):
    """A request whose session user differs from APISIX logs out then in."""
    other_user = UserFactory.create()
    mock_login = mocker.patch("main.middleware.apisix_user.login")
    mock_logout = mocker.patch("main.middleware.apisix_user.logout")
    request = _request(mocker, apisix_user_info, other_user)

    ApisixUserMiddleware(mocker.Mock()).process_request(request)

    mock_logout.assert_called_once()
    mock_login.assert_called_once()
    assert request.user.global_id == apisix_user_info["sub"]


@pytest.mark.django_db
def test_missing_headers_logs_out_authenticated_user(mocker):
    """An authenticated request with no APISIX headers logs the Django user out.

    APISIX is the source of truth: once it stops injecting the user header (e.g.
    after an APISIX/Keycloak logout), the lingering Django session must not keep
    the user authenticated.
    """
    mock_logout = mocker.patch("main.middleware.apisix_user.logout")
    request = mocker.Mock(META={}, user=UserFactory.create())

    ApisixUserMiddleware(mocker.Mock()).process_request(request)

    mock_logout.assert_called_once_with(request)


@pytest.mark.django_db
def test_missing_headers_leaves_anonymous_user_alone(mocker):
    """An anonymous request with no APISIX headers does not call logout()."""
    mock_logout = mocker.patch("main.middleware.apisix_user.logout")
    request = mocker.Mock(META={}, user=AnonymousUser())

    ApisixUserMiddleware(mocker.Mock()).process_request(request)

    mock_logout.assert_not_called()


@pytest.mark.django_db
def test_api_gateway_userdata_always_set(mocker, apisix_user_info):
    """api_gateway_userdata is populated even when login is skipped."""
    existing = UserFactory.create(
        global_id=apisix_user_info["sub"],
        email=apisix_user_info["email"],
        username=apisix_user_info["preferred_username"],
        name=apisix_user_info["name"],
    )
    mocker.patch("main.middleware.apisix_user.login")
    request = _request(mocker, apisix_user_info, existing)

    ApisixUserMiddleware(mocker.Mock()).process_request(request)

    assert request.api_gateway_userdata["global_id"] == apisix_user_info["sub"]
