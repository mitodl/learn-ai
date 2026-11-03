"""Tests for users middleware"""

import base64
import json

import pytest
from asgiref.sync import sync_to_async
from django.contrib.auth.models import AnonymousUser

from main.factories import UserFactory
from users.middleware import ApisixChannelAuthMiddleware

# The middleware queries from a sync_to_async worker thread with its own
# connection, so the test-case transaction wouldn't isolate these tests.
pytestmark = pytest.mark.django_db(transaction=True)


async def _scope_user(headers):
    """Run the middleware over headers and return the resulting scope user"""
    captured = {}

    async def app(scope, receive, send):
        captured["user"] = scope["user"]

    await ApisixChannelAuthMiddleware(app)({"headers": headers}, None, None)
    return captured["user"]


def _userinfo_header(userinfo):
    """Encode an x-userinfo header the way APISIX sends it"""
    return (b"x-userinfo", base64.b64encode(json.dumps(userinfo).encode()))


async def test_no_userinfo_header_is_anonymous():
    """A request without x-userinfo gets an AnonymousUser instance"""
    user = await _scope_user([])
    assert isinstance(user, AnonymousUser)


@pytest.mark.parametrize("userinfo", [{}, {"sub": ""}])
@pytest.mark.parametrize("local_user_count", [1, 2])
async def test_userinfo_without_sub_is_anonymous(userinfo, local_user_count):
    """
    A header without a sub is anonymous. It must not match local users, whose
    global_id defaults to an empty string.
    """
    await sync_to_async(UserFactory.create_batch)(local_user_count, global_id="")
    user = await _scope_user(
        [_userinfo_header({**userinfo, "preferred_username": "someone"})]
    )
    assert isinstance(user, AnonymousUser)


async def test_userinfo_with_sub_resolves_user():
    """A header with a sub resolves to a user with that global_id"""
    user = await _scope_user(
        [
            _userinfo_header(
                {
                    "sub": "abc-123",
                    "preferred_username": "someone",
                    "email": "someone@example.com",
                    "name": "Some One",
                }
            )
        ]
    )
    assert user.is_authenticated
    assert user.global_id == "abc-123"
