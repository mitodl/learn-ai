"""Tests for the user consumers"""

import json
from urllib.parse import quote

import pytest
from asgiref.testing import ApplicationCommunicator

from users.consumers import LoginHttpConsumer

AUTHORIZE_URL = (
    "https://mitxonline.mit.edu/oauth2/authorize/?client_id=edx"
    "&redirect_uri=https%3A%2F%2Fcourses.learn.mit.edu%2Fauth%2Fcomplete%2Fol-oauth2%2F"
    "&state=abc123&response_type=code&code_challenge=xyz&code_challenge_method=S256"
)


@pytest.fixture(autouse=True)
def allowed_hosts(settings):
    """Allow the production hosts plus a local frontend, https only."""
    settings.LOGIN_REDIRECT_ALLOWED_HOSTS = [
        "mitxonline.mit.edu",
        "courses.learn.mit.edu",
        "learn.mit.edu",
        "ai.open.odl.local:8003",
    ]
    settings.LOGIN_REDIRECT_REQUIRE_HTTPS = True


@pytest.fixture(autouse=True)
def no_channel_layer(mocker):
    """
    Run the consumer without a channel layer. Other tests leave a mocked one
    cached, which the consumer would otherwise wait on.
    """
    mocker.patch("channels.consumer.get_channel_layer", return_value=None)


async def get_login(query: str = "") -> dict:
    """GET the login route with a query string; return status, headers, body."""
    communicator = ApplicationCommunicator(
        LoginHttpConsumer.as_asgi(),
        {
            "type": "http",
            "method": "GET",
            "path": "/http/login/",
            "query_string": query.encode(),
            "headers": [],
        },
    )
    await communicator.send_input({"type": "http.request", "body": b""})
    start = await communicator.receive_output(1)
    body = b""
    while True:
        message = await communicator.receive_output(1)
        body += message.get("body", b"")
        if not message.get("more_body"):
            break
    await communicator.send_input({"type": "http.disconnect"})
    await communicator.wait(1)
    return {
        "status": start["status"],
        "headers": start.get("headers", []),
        "body": body,
    }


@pytest.mark.parametrize(
    ("next_url", "location"),
    [
        ("https://learn.mit.edu/dashboard", "https://learn.mit.edu/dashboard"),
        (
            "https://courses.learn.mit.edu/learning/course/course-v1:MITxT+18.01x+3T2026/home",
            "https://courses.learn.mit.edu/learning/course/course-v1:MITxT+18.01x+3T2026/home",
        ),
        (AUTHORIZE_URL, AUTHORIZE_URL),
        ("https://learn.mit.edu/caf\u00e9", "https://learn.mit.edu/caf%C3%A9"),
        (
            "https://learn.mit.edu/\r\nSet-Cookie: x=1",
            "https://learn.mit.edu/%0D%0ASet-Cookie:%20x=1",
        ),
    ],
)
async def test_login_redirects_to_allowed_next(next_url, location):
    """An allowed next URL is redirected to, percent-encoded where needed."""
    response = await get_login(f"next={quote(next_url, safe='')}")

    assert response["status"] == 302
    headers = dict(response["headers"])
    assert headers[b"Location"].decode() == location
    assert headers[b"Cache-Control"] == b"no-store"


@pytest.mark.parametrize(
    "next_url",
    [
        "https://learn.mit.edu.evil.test/",
        "https://evil.test/?https://learn.mit.edu/",
        "https://evil.test\\@learn.mit.edu/",
        "https://learn.mit.edu@evil.test/",
        "https://user@learn.mit.edu/",
        "https://learn.mit.edu:8443/",
        "https://learn.mit.edu:443/",
        "https://learn.mit.edu./",
        "https%3A%2F%2Flearn.mit.edu%2F",
        "https://learn.mit.edu/" + "a" * 2100,
        "https://sub.learn.mit.edu/",
        "http://learn.mit.edu/",
        "//learn.mit.edu/",
        "/dashboard",
        "javascript:alert(1)//learn.mit.edu",
        " https://learn.mit.edu/",
        "https://learn.mit\t.edu/",
        "",
    ],
)
async def test_login_refuses_other_next(next_url):
    """Anything but an https URL on an exactly allowed host is refused."""
    response = await get_login(f"next={quote(next_url, safe='')}")

    assert response["status"] == 400
    assert b"Location" not in dict(response["headers"])


async def test_login_refused_log_omits_user_info(caplog):
    """The refusal log names the host, not credentials in the URL."""
    next_url = "https://learner:secret-token@learn.mit.edu/"
    await get_login(f"next={quote(next_url, safe='')}")

    assert [record.args for record in caplog.records] == [("learn.mit.edu",)]
    assert "secret-token" not in caplog.text


@pytest.mark.parametrize(
    ("require_https", "status"),
    [(True, 400), (False, 302)],
)
async def test_login_http_next_only_without_https_requirement(
    settings, require_https, status
):
    """A local http frontend is allowed only when https isn't required."""
    settings.LOGIN_REDIRECT_REQUIRE_HTTPS = require_https
    url = quote("http://ai.open.odl.local:8003/syllabus", safe="")
    response = await get_login(f"next={url}")

    assert response["status"] == status


async def test_login_without_next_returns_user_info():
    """Without next, the login route still returns the user's info."""
    response = await get_login()

    assert response["status"] == 200
    assert json.loads(response["body"]) == {"is_anonymous": True}
