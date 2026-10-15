"""Tests for the learner API client"""

import asyncio
import copy
import pickle

import httpx
import pytest
from django.contrib.auth.models import AnonymousUser

from ai_chatbots import learner_api
from ai_chatbots.learner_api import (
    LearnerAPIError,
    LearnerSession,
    forwarded_learner_session,
    get_course_dates,
    get_enrollments,
    get_program_certificates,
    get_program_enrollments,
    has_learner_session,
)

LEARN_COOKIE = "learn_session"
EDX_COOKIE = "edx-lms-sessionid"
COURSE_KEY = "course-v1:MITxT+18.01x+3T2026"


@pytest.fixture(autouse=True)
def learner_api_settings(settings):
    """Point the client at fixed hosts and cookie names."""
    settings.LEARNER_API_LEARN_BASE_URL = "https://api.learn.test/"
    settings.LEARNER_API_LEARN_COOKIE_NAME = LEARN_COOKIE
    settings.LEARNER_API_OPENEDX_BASE_URL = "https://courses.learn.test/"
    settings.LEARNER_API_OPENEDX_COOKIE_NAME = EDX_COOKIE


class _Sent(list):
    """Requests sent, plus responses to give by path."""

    responses: dict


@pytest.fixture
def requests_sent(mocker):
    """Answer every request with a JSON 200 (or a set response) and record it."""
    sent = _Sent()
    sent.responses = {}
    real_client = httpx.AsyncClient

    def handler(request):
        sent.append(request)
        return sent.responses.get(
            request.url.path,
            httpx.Response(200, json=[], headers={"Set-Cookie": "csrf=abc"}),
        )

    mocker.patch.object(
        learner_api.httpx,
        "AsyncClient",
        side_effect=lambda **kwargs: real_client(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    return sent


@pytest.fixture
def signed_in_scope(mocker):
    """Build a request scope for a signed-in learner with both session cookies."""
    return {
        "user": mocker.Mock(is_authenticated=True),
        "cookies": {
            LEARN_COOKIE: "learn-1",
            f"{LEARN_COOKIE}2": "learn-2",
            EDX_COOKIE: "edx-1",
            "unrelated": "other",
        },
    }


@pytest.mark.parametrize(
    "operation",
    [get_enrollments, get_program_enrollments, get_program_certificates],
)
async def test_learn_calls_send_only_the_learn_session(
    requests_sent, signed_in_scope, operation
):
    """MIT Learn calls carry the Learn session and its chunks, nothing else."""
    with forwarded_learner_session(signed_in_scope):
        assert await operation() == []

    (request,) = requests_sent
    assert request.url.host == "api.learn.test"
    assert (
        request.headers["Cookie"] == f"{LEARN_COOKIE}=learn-1; {LEARN_COOKIE}2=learn-2"
    )


async def test_course_dates_send_only_the_edx_session(requests_sent, signed_in_scope):
    """Open edX calls carry only the edX session, to the course host."""
    with forwarded_learner_session(signed_in_scope):
        await get_course_dates(COURSE_KEY)

    (request,) = requests_sent
    assert request.url.host == "courses.learn.test"
    assert request.url.path == f"/api/course_home/v1/dates/{COURSE_KEY}"
    assert request.headers["Cookie"] == f"{EDX_COOKIE}=edx-1"


async def test_returned_cookies_are_not_sent_back(requests_sent, signed_in_scope):
    """Cookies an API sets are not added to the learner's next call."""
    with forwarded_learner_session(signed_in_scope):
        await get_enrollments()
        await get_program_enrollments()

    assert "csrf" not in requests_sent[1].headers["Cookie"]


@pytest.mark.parametrize(
    ("user", "cookies"),
    [
        (AnonymousUser(), {LEARN_COOKIE: "learn-1", EDX_COOKIE: "edx-1"}),
        (None, {LEARN_COOKIE: "learn-1"}),
        ("signed_in", {}),
        ("signed_in", {f"{LEARN_COOKIE}2": "chunk-without-first-part"}),
        ("signed_in", {LEARN_COOKIE: ""}),
        ("signed_in", {LEARN_COOKIE: "caf\xe9"}),
        ("signed_in", {LEARN_COOKIE: "learn-1", f"{LEARN_COOKIE}2": "has space"}),
    ],
)
async def test_no_session_makes_no_request(mocker, requests_sent, user, cookies):
    """Anonymous users, and requests without a usable session cookie, get no data."""
    if user == "signed_in":
        user = mocker.Mock(is_authenticated=True)
    with forwarded_learner_session({"user": user, "cookies": cookies}):
        assert has_learner_session() is False
        with pytest.raises(LearnerAPIError):
            await get_enrollments()

    assert requests_sent == []


async def test_edx_session_alone_does_not_reach_learn(
    mocker, requests_sent, signed_in_scope
):
    """A learner with only an edX session can't call MIT Learn APIs."""
    signed_in_scope["cookies"] = {EDX_COOKIE: "edx-1"}
    with forwarded_learner_session(signed_in_scope):
        assert has_learner_session() is False
        with pytest.raises(LearnerAPIError):
            await get_enrollments()
        await get_course_dates(COURSE_KEY)

    assert [request.url.host for request in requests_sent] == ["courses.learn.test"]


async def test_session_ends_with_the_request(requests_sent, signed_in_scope):
    """Once the request is done, its session is gone."""
    with forwarded_learner_session(signed_in_scope):
        assert has_learner_session() is True
    assert has_learner_session() is False

    with pytest.raises(LearnerAPIError):
        await get_enrollments()
    assert requests_sent == []


@pytest.mark.parametrize("status", [302, 401, 403, 404, 500])
async def test_error_statuses_raise(requests_sent, signed_in_scope, status):
    """Non-2xx responses, redirects included, raise and are not followed."""
    requests_sent.responses["/mitxonline/api/v3/enrollments/"] = httpx.Response(
        status, headers={"Location": "https://elsewhere.test/"}
    )
    with (
        forwarded_learner_session(signed_in_scope),
        pytest.raises(LearnerAPIError) as excinfo,
    ):
        await get_enrollments()

    assert excinfo.value.status_code == status
    assert len(requests_sent) == 1


async def test_transport_error_is_not_chained(mocker, signed_in_scope):
    """Errors from httpx, whose request holds the cookies, aren't chained on."""
    mocker.patch.object(
        learner_api.httpx.AsyncClient, "get", side_effect=httpx.ConnectError("down")
    )
    with (
        forwarded_learner_session(signed_in_scope),
        pytest.raises(LearnerAPIError) as excinfo,
    ):
        await get_enrollments()

    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__ is True


async def test_invalid_json_raises(requests_sent, signed_in_scope):
    """A 200 that isn't JSON raises."""
    requests_sent.responses["/mitxonline/api/v3/enrollments/"] = httpx.Response(
        200, text="<html>login</html>"
    )
    with (
        forwarded_learner_session(signed_in_scope),
        pytest.raises(LearnerAPIError) as excinfo,
    ):
        await get_enrollments()

    assert excinfo.value.__suppress_context__ is True


@pytest.mark.parametrize(
    "course_key",
    [
        "../../api/user/v1/accounts",
        f"{COURSE_KEY}/../../admin",
        f"{COURSE_KEY}?x=1",
        "course-v1:MITxT+18.01x",
        "https://evil.test/",
    ],
)
async def test_invalid_course_key_makes_no_request(
    requests_sent, signed_in_scope, course_key
):
    """Only well-formed course keys reach the course host."""
    with forwarded_learner_session(signed_in_scope), pytest.raises(LearnerAPIError):
        await get_course_dates(course_key)

    assert requests_sent == []


@pytest.mark.parametrize(
    "base_url",
    ["", "http://api.learn.test/", "https://api.learn.test:443x/", "/relative/"],
)
async def test_unusable_base_url_makes_no_request(
    settings, requests_sent, signed_in_scope, base_url
):
    """A misconfigured or non-https base URL never sends the session anywhere."""
    settings.LEARNER_API_LEARN_BASE_URL = base_url
    with forwarded_learner_session(signed_in_scope), pytest.raises(LearnerAPIError):
        await get_enrollments()

    assert requests_sent == []


async def test_concurrent_requests_keep_their_own_session(mocker, requests_sent):
    """Two requests handled at once each send only their own learner's cookie."""

    async def request_as(value):
        scope = {
            "user": mocker.Mock(is_authenticated=True),
            "cookies": {LEARN_COOKIE: value},
        }
        with forwarded_learner_session(scope):
            await asyncio.sleep(0)
            await get_enrollments()
            await asyncio.sleep(0)
            await get_program_enrollments()

    await asyncio.gather(request_as("learner-a"), request_as("learner-b"))

    assert sorted(request.headers["Cookie"] for request in requests_sent) == [
        f"{LEARN_COOKIE}=learner-a",
        f"{LEARN_COOKIE}=learner-a",
        f"{LEARN_COOKIE}=learner-b",
        f"{LEARN_COOKIE}=learner-b",
    ]


def test_learner_session_hides_cookie_values():
    """The session never shows or saves its cookie values."""
    session = LearnerSession({"learn": {"name": "secret-value"}, "openedx": {}})

    assert "secret-value" not in repr(session)
    assert "secret-value" not in str(session)
    assert not hasattr(session, "__dict__")
    for save in (pickle.dumps, copy.copy, copy.deepcopy):
        with pytest.raises(TypeError):
            save(session)
