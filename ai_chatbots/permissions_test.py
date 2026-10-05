"""Tests for ai_chatbots permissions"""

from types import SimpleNamespace

import pytest
from django.contrib.auth.models import AnonymousUser
from django.test import RequestFactory
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode

from ai_chatbots.constants import AI_THREADS_ANONYMOUS_COOKIE_KEY
from ai_chatbots.factories import UserChatSessionFactory
from ai_chatbots.permissions import IsThreadOwner

pytestmark = pytest.mark.django_db


def _anon_request(cookies):
    """Build an anonymous request with the given cookies"""
    request = RequestFactory().get("/")
    request.user = AnonymousUser()
    request.COOKIES = cookies
    return request


def _has_permission(request, thread_id):
    """Run IsThreadOwner against a view for thread_id"""
    view = SimpleNamespace(kwargs={"thread_id": thread_id})
    return IsThreadOwner().has_permission(request, view)


def test_is_thread_owner_anon_valid_cookie():
    """An anonymous user with a cookie matching the thread is allowed"""
    session = UserChatSessionFactory.create(user=None)
    cookie = urlsafe_base64_encode(force_bytes(f"{session.thread_id}|1000"))
    request = _anon_request(
        {f"{session.agent}_{AI_THREADS_ANONYMOUS_COOKIE_KEY}": cookie}
    )
    assert _has_permission(request, session.thread_id) is True


def test_is_thread_owner_anon_missing_cookie():
    """An anonymous user without a cookie is denied"""
    session = UserChatSessionFactory.create(user=None)
    assert _has_permission(_anon_request({}), session.thread_id) is False


def test_is_thread_owner_anon_malformed_cookie(caplog):
    """An anonymous user with an undecodable cookie is denied instead of erroring"""
    session = UserChatSessionFactory.create(user=None)
    request = _anon_request(
        {f"{session.agent}_{AI_THREADS_ANONYMOUS_COOKIE_KEY}": "!!not-base64!!"}
    )
    assert _has_permission(request, session.thread_id) is False
    assert f"Invalid cookie value for anon user and {session.thread_id}" in caplog.text
