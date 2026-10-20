"""Tests for Zendesk support ticket creation."""

import logging
import re

import pytest
from httpx import (
    ConnectError,
    ConnectTimeout,
    HTTPStatusError,
    PoolTimeout,
    ReadTimeout,
    RequestError,
)

from ai_chatbots.constants import (
    ZENDESK_OAUTH_TOKEN_PATH,
    ZENDESK_SUPPORT_INTAKE_TAG,
    ZENDESK_UNVERIFIED_REQUESTER_TAG,
)
from ai_chatbots.utils import get_django_cache
from ai_chatbots.zendesk import (
    ZENDESK_TOKEN_CACHE_KEY,
    file_support_ticket,
    max_filing_seconds,
)

pytestmark = pytest.mark.django_db

# Stand-ins for the per-instance field ids an environment configures.
URL_FIELD_ID = 51720932683537
NAME_FIELD_ID = 51721582949009
USERNAME_FIELD_ID = 51741744445201

# A token already in the cache, so most tests file without hitting the mint
# endpoint - the common case once an environment has warmed up.
CACHED_TOKEN = "cached-zd-token"  # noqa: S105
# What the mint endpoint hands back when the cache is cold.
MINTED_TOKEN = "minted-zd-token"  # noqa: S105


@pytest.fixture
def token_cache(settings):
    """
    Point the default cache at locmem so the token actually persists.

    The test default cache is a no-op DummyCache, which would make the
    mint-and-cache behaviour untestable.
    """
    settings.CELERY_BROKER_URL = ""
    settings.CACHES = {
        **settings.CACHES,
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "zendesk-token-test",
        },
    }
    cache = get_django_cache()
    cache.clear()
    return cache


@pytest.fixture
def zendesk_ticket_settings(settings, token_cache):
    """Stub mode off, OAuth client configured, a token already cached."""
    settings.AI_ZENDESK_STUB_MODE = False
    settings.AI_ZENDESK_API_URL = "https://mitlearn.zendesk.com"
    settings.AI_ZENDESK_OAUTH_CLIENT_ID = "mit_odl_support"
    settings.AI_ZENDESK_OAUTH_CLIENT_SECRET = "zdsecret"  # noqa: S105
    settings.AI_ZENDESK_URL_FIELD_ID = URL_FIELD_ID
    settings.AI_ZENDESK_NAME_FIELD_ID = NAME_FIELD_ID
    settings.AI_ZENDESK_USERNAME_FIELD_ID = USERNAME_FIELD_ID
    token_cache.set(ZENDESK_TOKEN_CACHE_KEY, CACHED_TOKEN, 300)
    return settings


def _response(mocker, json_body, status_code=200):
    """Build a stand-in httpx response for the async client mock."""
    response = mocker.Mock()
    response.json.return_value = json_body
    response.status_code = status_code
    response.raise_for_status = mocker.Mock()
    return response


def _failing_response(mocker, status_code):
    """Build a response that raises from raise_for_status, as httpx's does."""
    response = _response(mocker, {}, status_code=status_code)
    response.raise_for_status.side_effect = HTTPStatusError(
        f"server said {status_code}", request=mocker.Mock(), response=response
    )
    return response


def _mock_zendesk_client(mocker, *, ticket_responses, expires_in=1799):
    """
    Patch the async client so mint POSTs and ticket POSTs get separate replies.

    ``ticket_responses`` is consumed in order, so a test can hand back a 401
    first and a success second to exercise the re-mint path.
    """
    mint_response = _response(
        mocker,
        {
            "access_token": MINTED_TOKEN,
            "token_type": "bearer",
            "scope": "tickets:write users:write",
            "expires_in": expires_in,
        },
    )
    ticket_iter = iter(ticket_responses)

    async def post(url, *_args, **_kwargs):
        if url.endswith(ZENDESK_OAUTH_TOKEN_PATH):
            return mint_response
        return next(ticket_iter)

    client = mocker.Mock()
    client.post = mocker.AsyncMock(side_effect=post)
    mocker.patch("ai_chatbots.zendesk.get_async_http_client", return_value=client)
    return client


def _ticket_args(**overrides):
    args = {
        "subject": "Video will not play",
        "description": "The week 3 video just spins.",
        "email": "learner@example.com",
        "page_url": "https://learn.mit.edu/week-3",
    }
    args.update(overrides)
    return args


async def test_file_support_ticket_stub_reference_is_visibly_fake(settings, mocker):
    """Stub mode must not look like a real ticket number, and must not call out."""
    settings.AI_ZENDESK_STUB_MODE = True
    client = mocker.patch("ai_chatbots.zendesk.get_async_http_client")

    result = await file_support_ticket(**_ticket_args())

    assert re.fullmatch(r"STUB-[0-9A-F]{8}", result["reference"]), result["reference"]
    assert result["stub"] is True
    client.assert_not_called()


async def test_file_support_ticket_stub_warns(settings, caplog):
    """
    Nothing downstream reads the "stub" flag, so a stub left on in production is
    invisible: the learner is told a ticket exists and no one is looking at one.
    """
    settings.AI_ZENDESK_STUB_MODE = True

    with caplog.at_level(logging.WARNING, logger="ai_chatbots.zendesk"):
        await file_support_ticket(**_ticket_args())

    assert "AI_ZENDESK_STUB_MODE" in caplog.text


async def test_file_support_ticket_stub_references_differ(settings):
    """A constant reference reads as a real ticket id the second time you see it."""
    settings.AI_ZENDESK_STUB_MODE = True

    first = await file_support_ticket(**_ticket_args())
    second = await file_support_ticket(**_ticket_args())

    assert first["reference"] != second["reference"]


async def test_file_support_ticket_stub_still_rejects_a_bad_email(settings):
    """
    Stub mode is the local path, so an address it accepts and the real path
    rejects is a bug that only shows up once deployed.
    """
    settings.AI_ZENDESK_STUB_MODE = True

    result = await file_support_ticket(**_ticket_args(email="not-an-address"))

    assert "error" in result
    assert "reference" not in result


async def test_file_support_ticket_posts_to_zendesk(
    zendesk_ticket_settings, mock_httpx_async_client
):
    """A real ticket returns the Zendesk id, tags itself, and carries the page URL."""
    mock_patch = mock_httpx_async_client(
        {"ticket": {"id": 4821}},
        patch_path="ai_chatbots.zendesk.get_async_http_client",
    )

    result = await file_support_ticket(**_ticket_args(), email_verified=True)

    assert result == {"reference": 4821, "stub": False}

    client = mock_patch.return_value
    client.post.assert_called_once()
    args, kwargs = client.post.call_args
    assert args[0] == "https://mitlearn.zendesk.com/api/v2/tickets.json"
    ticket = kwargs["json"]["ticket"]
    assert ticket["subject"] == "Video will not play"
    assert ticket["requester"]["email"] == "learner@example.com"
    assert "tim_support_intake" in ticket["tags"]
    assert "https://learn.mit.edu/week-3" in ticket["comment"]["body"]
    # OAuth bearer, not basic auth: Zendesk removes API tokens on 2027-04-30.
    assert kwargs["headers"]["Authorization"] == f"Bearer {CACHED_TOKEN}"
    assert "auth" not in kwargs


@pytest.mark.parametrize("bad_email", ["", "   ", "not-an-email", "nobody@"])
async def test_file_support_ticket_refuses_bad_email(
    zendesk_ticket_settings, mock_httpx_async_client, bad_email
):
    """Filing a ticket nobody can reply to is worse than not filing one."""
    mock_patch = mock_httpx_async_client(
        {"ticket": {"id": 1}},
        patch_path="ai_chatbots.zendesk.get_async_http_client",
    )

    result = await file_support_ticket(**_ticket_args(email=bad_email))

    assert "error" in result
    assert "reference" not in result
    mock_patch.return_value.post.assert_not_called()


async def test_file_support_ticket_returns_error_on_zendesk_failure(
    zendesk_ticket_settings, mock_httpx_async_client
):
    """A Zendesk outage must not produce an invented reference number."""
    mock_patch = mock_httpx_async_client(
        {},
        patch_path="ai_chatbots.zendesk.get_async_http_client",
    )
    mock_patch.return_value.post.side_effect = RequestError("zendesk is down")

    result = await file_support_ticket(**_ticket_args())

    assert "error" in result
    assert "reference" not in result


async def test_file_support_ticket_failure_log_carries_no_learner_pii(
    zendesk_ticket_settings, mock_httpx_async_client, caplog
):
    """
    Sentry ships frame locals by default, so log.exception here would attach the
    learner's address and their problem description to the event.
    """
    mock_patch = mock_httpx_async_client(
        {},
        patch_path="ai_chatbots.zendesk.get_async_http_client",
    )
    mock_patch.return_value.post.side_effect = RequestError("zendesk is down")

    with caplog.at_level(logging.ERROR, logger="ai_chatbots.zendesk"):
        await file_support_ticket(**_ticket_args())

    assert "RequestError" in caplog.text
    assert "learner@example.com" not in caplog.text
    assert not any(record.exc_info for record in caplog.records)


@pytest.mark.parametrize("page_url", ["", None])
async def test_file_support_ticket_without_page_url(
    zendesk_ticket_settings, mock_httpx_async_client, page_url
):
    """A surface that sends no page URL must still be able to file."""
    mock_patch = mock_httpx_async_client(
        {"ticket": {"id": 77}},
        patch_path="ai_chatbots.zendesk.get_async_http_client",
    )

    result = await file_support_ticket(**_ticket_args(page_url=page_url))

    assert result == {"reference": 77, "stub": False}
    _, kwargs = mock_patch.return_value.post.call_args
    assert kwargs["json"]["ticket"]["comment"]["body"] == "The week 3 video just spins."


async def test_file_support_ticket_fills_the_zendesk_custom_fields(
    zendesk_ticket_settings, mock_httpx_async_client
):
    """
    Support filters and views read the url and name ticket fields, which Zendesk
    only populates from the custom_fields array - top level keys are dropped.
    """
    mock_patch = mock_httpx_async_client(
        {"ticket": {"id": 4821}},
        patch_path="ai_chatbots.zendesk.get_async_http_client",
    )

    await file_support_ticket(**_ticket_args(name="Ada Lovelace"))

    _, kwargs = mock_patch.return_value.post.call_args
    ticket = kwargs["json"]["ticket"]
    assert ticket["custom_fields"] == [
        {"id": URL_FIELD_ID, "value": "https://learn.mit.edu/week-3"},
        {"id": NAME_FIELD_ID, "value": "Ada Lovelace"},
    ]
    assert ticket["requester"] == {
        "email": "learner@example.com",
        "name": "Ada Lovelace",
    }


async def test_file_support_ticket_fills_the_zendesk_username_custom_field(
    zendesk_ticket_settings, mock_httpx_async_client
):
    """
    The username is the account handle, kept as its own custom field so it
    survives even when the requester name above is a display name instead.
    """
    mock_patch = mock_httpx_async_client(
        {"ticket": {"id": 4821}},
        patch_path="ai_chatbots.zendesk.get_async_http_client",
    )

    await file_support_ticket(**_ticket_args(name="Ada Lovelace", username="ada"))

    _, kwargs = mock_patch.return_value.post.call_args
    ticket = kwargs["json"]["ticket"]
    assert ticket["custom_fields"] == [
        {"id": URL_FIELD_ID, "value": "https://learn.mit.edu/week-3"},
        {"id": NAME_FIELD_ID, "value": "Ada Lovelace"},
        {"id": USERNAME_FIELD_ID, "value": "ada"},
    ]


async def test_file_support_ticket_names_the_requester_by_email_when_unknown(
    zendesk_ticket_settings, mock_httpx_async_client
):
    """An anonymous learner gives us no name, so the address is all support gets."""
    mock_patch = mock_httpx_async_client(
        {"ticket": {"id": 4821}},
        patch_path="ai_chatbots.zendesk.get_async_http_client",
    )

    await file_support_ticket(**_ticket_args())

    _, kwargs = mock_patch.return_value.post.call_args
    ticket = kwargs["json"]["ticket"]
    assert ticket["requester"]["name"] == "learner@example.com"
    assert ticket["custom_fields"] == [
        {"id": URL_FIELD_ID, "value": "https://learn.mit.edu/week-3"}
    ]


async def test_file_support_ticket_omits_empty_custom_fields(
    zendesk_ticket_settings, mock_httpx_async_client
):
    """Zendesk rejects a custom field sent with an empty value."""
    mock_patch = mock_httpx_async_client(
        {"ticket": {"id": 4821}},
        patch_path="ai_chatbots.zendesk.get_async_http_client",
    )

    await file_support_ticket(**_ticket_args(page_url="", name=""))

    _, kwargs = mock_patch.return_value.post.call_args
    assert "custom_fields" not in kwargs["json"]["ticket"]


async def test_file_support_ticket_omits_unconfigured_custom_fields(
    zendesk_ticket_settings, mock_httpx_async_client
):
    """
    Field ids are minted per Zendesk instance. An environment pointed at one
    that has no such field must still file: Zendesk rejects the whole ticket
    for an id it does not recognise.
    """
    zendesk_ticket_settings.AI_ZENDESK_URL_FIELD_ID = 0
    zendesk_ticket_settings.AI_ZENDESK_NAME_FIELD_ID = 0
    zendesk_ticket_settings.AI_ZENDESK_USERNAME_FIELD_ID = 0
    mock_patch = mock_httpx_async_client(
        {"ticket": {"id": 4821}},
        patch_path="ai_chatbots.zendesk.get_async_http_client",
    )

    result = await file_support_ticket(
        **_ticket_args(name="Ada Lovelace", username="ada")
    )

    assert result == {"reference": 4821, "stub": False}
    _, kwargs = mock_patch.return_value.post.call_args
    assert "custom_fields" not in kwargs["json"]["ticket"]


async def test_file_support_ticket_keys_the_ticket_to_the_thread(
    zendesk_ticket_settings, mock_httpx_async_client
):
    """
    Nothing local can tell a lost response from a refused one, so a retry after
    a timeout would file twice. Zendesk has to be the one to say it already has
    this ticket, and external_id is what lets a human find the pair if it does
    not.
    """
    mock_patch = mock_httpx_async_client(
        {"ticket": {"id": 4821}},
        patch_path="ai_chatbots.zendesk.get_async_http_client",
    )

    await file_support_ticket(**_ticket_args(thread_id="thread-abc"))

    _, kwargs = mock_patch.return_value.post.call_args
    assert kwargs["headers"]["Idempotency-Key"] == "thread-abc"
    assert kwargs["json"]["ticket"]["external_id"] == "thread-abc"


async def test_file_support_ticket_without_a_thread_id(
    zendesk_ticket_settings, mock_httpx_async_client
):
    """An empty key is not a key, and Zendesk rejects a ticket over a bad one."""
    mock_patch = mock_httpx_async_client(
        {"ticket": {"id": 4821}},
        patch_path="ai_chatbots.zendesk.get_async_http_client",
    )

    await file_support_ticket(**_ticket_args())

    _, kwargs = mock_patch.return_value.post.call_args
    assert "Idempotency-Key" not in kwargs["headers"]
    assert "external_id" not in kwargs["json"]["ticket"]


async def test_file_support_ticket_reuses_the_key_on_the_401_retry(
    zendesk_ticket_settings, mocker
):
    """
    The re-mint posts the same ticket a second time. A fresh key there would be
    a second ticket on the one occasion we know the first POST was not applied.
    """
    client = _mock_zendesk_client(
        mocker,
        ticket_responses=[
            _response(mocker, {}, status_code=401),
            _response(mocker, {"ticket": {"id": 4821}}),
        ],
    )

    await file_support_ticket(**_ticket_args(thread_id="thread-abc"))

    keys = [
        call.kwargs["headers"]["Idempotency-Key"]
        for call in client.post.call_args_list
        if call.args[0].endswith("/api/v2/tickets.json")
    ]
    assert keys == ["thread-abc", "thread-abc"]


@pytest.mark.parametrize(
    ("status_code", "uncertain"),
    [(500, True), (502, True), (503, True), (400, False), (422, False)],
)
async def test_file_support_ticket_says_whether_zendesk_may_have_filed_it(
    zendesk_ticket_settings, mocker, status_code, uncertain
):
    """
    Zendesk can create the ticket and still fail the response, so a 5xx leaves
    the outcome unknown. A rejection created nothing and is a plain failure.
    """
    _mock_zendesk_client(
        mocker, ticket_responses=[_failing_response(mocker, status_code)]
    )

    result = await file_support_ticket(**_ticket_args())

    assert "reference" not in result
    assert result["uncertain"] is uncertain


@pytest.mark.parametrize(
    ("error", "uncertain"),
    [
        (ReadTimeout("zendesk took too long"), True),
        (ConnectTimeout("never reached zendesk"), False),
        (PoolTimeout("no connection free"), False),
        (ConnectError("dns failure"), False),
        (RequestError("zendesk is down"), False),
    ],
)
async def test_file_support_ticket_says_a_lost_reply_is_not_a_refusal(
    zendesk_ticket_settings, mock_httpx_async_client, error, uncertain
):
    """
    A read timeout means the request went out and we stopped waiting, so a
    ticket may exist. A connect failure never put it on the wire.
    """
    mock_patch = mock_httpx_async_client(
        {},
        patch_path="ai_chatbots.zendesk.get_async_http_client",
    )
    mock_patch.return_value.post.side_effect = error

    result = await file_support_ticket(**_ticket_args())

    assert result["uncertain"] is uncertain


async def test_file_support_ticket_is_sure_it_failed_when_the_mint_times_out(
    zendesk_ticket_settings, token_cache, mocker
):
    """
    A timeout minting the token happens before any ticket is posted, so there
    is nothing on Zendesk's side to be unsure about.
    """
    token_cache.delete(ZENDESK_TOKEN_CACHE_KEY)
    client = mocker.Mock()
    client.post = mocker.AsyncMock(side_effect=ReadTimeout("mint took too long"))
    mocker.patch("ai_chatbots.zendesk.get_async_http_client", return_value=client)

    result = await file_support_ticket(**_ticket_args())

    assert result["uncertain"] is False


@pytest.mark.parametrize(
    ("email_verified", "expect_tag"), [(True, False), (False, True)]
)
async def test_file_support_ticket_tags_an_unverified_requester(
    zendesk_ticket_settings, mock_httpx_async_client, email_verified, expect_tag
):
    """
    A surface learn-ai holds no session for just asserts who the learner is, and
    so does anyone typing an address into the chat. Support sees a name on the
    ticket either way, so the ticket has to say which one it is.
    """
    mock_patch = mock_httpx_async_client(
        {"ticket": {"id": 4821}},
        patch_path="ai_chatbots.zendesk.get_async_http_client",
    )

    await file_support_ticket(**_ticket_args(), email_verified=email_verified)

    _, kwargs = mock_patch.return_value.post.call_args
    tags = kwargs["json"]["ticket"]["tags"]
    assert (ZENDESK_UNVERIFIED_REQUESTER_TAG in tags) is expect_tag
    assert ZENDESK_SUPPORT_INTAKE_TAG in tags


@pytest.mark.parametrize("email_verified", [True, False])
async def test_file_support_ticket_withholds_unverified_text_from_the_requester(
    zendesk_ticket_settings, mock_httpx_async_client, email_verified
):
    """
    Nothing proves an unverified requester owns the address they named, and the
    intake route takes anonymous requests. A public first comment hands whoever
    files a way to send chosen text to a chosen address over MIT's domain, via
    whichever received-request trigger the instance has turned on. A private
    comment and a fixed subject leave the notification with nothing of theirs
    to carry, whatever those triggers are set to.
    """
    mock_patch = mock_httpx_async_client(
        {"ticket": {"id": 4821}},
        patch_path="ai_chatbots.zendesk.get_async_http_client",
    )

    await file_support_ticket(
        **_ticket_args(subject="Click http://evil.invalid to keep your account"),
        email_verified=email_verified,
    )

    _, kwargs = mock_patch.return_value.post.call_args
    ticket = kwargs["json"]["ticket"]
    assert ticket["comment"]["public"] is email_verified
    assert ("evil.invalid" in ticket["subject"]) is email_verified
    # Either way the agent reads the learner's own words in the ticket body.
    assert "The week 3 video just spins." in ticket["comment"]["body"]


async def test_file_support_ticket_mints_and_caches_token_when_cache_is_cold(
    zendesk_ticket_settings, token_cache, mocker
):
    """A cold cache mints a client_credentials token, uses it, and caches it."""
    token_cache.delete(ZENDESK_TOKEN_CACHE_KEY)
    client = _mock_zendesk_client(
        mocker, ticket_responses=[_response(mocker, {"ticket": {"id": 9}})]
    )

    result = await file_support_ticket(**_ticket_args())

    assert result == {"reference": 9, "stub": False}
    mint_call, ticket_call = client.post.call_args_list
    # The mint request carries the client_credentials grant, not the ticket.
    assert (
        mint_call.args[0] == f"https://mitlearn.zendesk.com{ZENDESK_OAUTH_TOKEN_PATH}"
    )
    assert mint_call.kwargs["json"] == {
        "grant_type": "client_credentials",
        "client_id": "mit_odl_support",
        "client_secret": "zdsecret",
        "scope": "tickets:write users:write",
    }
    # The ticket is filed with the freshly minted token, which is now cached.
    assert ticket_call.kwargs["headers"]["Authorization"] == f"Bearer {MINTED_TOKEN}"
    assert token_cache.get(ZENDESK_TOKEN_CACHE_KEY) == MINTED_TOKEN


async def test_file_support_ticket_uses_cached_token_without_minting(
    zendesk_ticket_settings, mocker
):
    """A warm cache must file without a round trip to the mint endpoint."""
    client = _mock_zendesk_client(
        mocker, ticket_responses=[_response(mocker, {"ticket": {"id": 9}})]
    )

    await file_support_ticket(**_ticket_args())

    posted_urls = [call.args[0] for call in client.post.call_args_list]
    assert posted_urls == ["https://mitlearn.zendesk.com/api/v2/tickets.json"]


async def test_file_support_ticket_remints_on_401(zendesk_ticket_settings, mocker):
    """A cached token revoked before its TTL gets re-minted and the file retried."""
    client = _mock_zendesk_client(
        mocker,
        ticket_responses=[
            _response(mocker, {}, status_code=401),
            _response(mocker, {"ticket": {"id": 9}}),
        ],
    )

    result = await file_support_ticket(**_ticket_args())

    assert result == {"reference": 9, "stub": False}
    ticket_calls = [
        call
        for call in client.post.call_args_list
        if call.args[0].endswith("/api/v2/tickets.json")
    ]
    assert len(ticket_calls) == 2
    # The retry uses the re-minted token, not the stale cached one.
    assert (
        ticket_calls[0].kwargs["headers"]["Authorization"] == f"Bearer {CACHED_TOKEN}"
    )
    assert (
        ticket_calls[1].kwargs["headers"]["Authorization"] == f"Bearer {MINTED_TOKEN}"
    )


async def test_max_filing_seconds_covers_every_request_on_the_worst_path(
    zendesk_ticket_settings, mocker, token_cache
):
    """
    Callers hold a lock for the length of a filing call, so the figure they
    size it from has to cover the slowest one: a cold token cache, a 401 that
    re-mints, and both ticket POSTs, each free to burn the full timeout.
    """
    token_cache.delete(ZENDESK_TOKEN_CACHE_KEY)
    client = _mock_zendesk_client(
        mocker,
        ticket_responses=[
            _response(mocker, {}, status_code=401),
            _response(mocker, {"ticket": {"id": 9}}),
        ],
    )

    await file_support_ticket(**_ticket_args())

    spent = sum(call.kwargs["timeout"] for call in client.post.call_args_list)
    assert spent == zendesk_ticket_settings.REQUESTS_TIMEOUT * 4
    assert max_filing_seconds() >= spent
