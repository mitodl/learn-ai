"""Tests for Zendesk support ticket creation."""

import logging
import re

import pytest
from httpx import RequestError

from ai_chatbots.constants import (
    ZENDESK_SUPPORT_INTAKE_TAG,
    ZENDESK_UNVERIFIED_REQUESTER_TAG,
)
from ai_chatbots.zendesk import file_support_ticket

pytestmark = pytest.mark.django_db

# Stand-ins for the per-instance field ids an environment configures.
URL_FIELD_ID = 51720932683537
NAME_FIELD_ID = 51721582949009
USERNAME_FIELD_ID = 51741744445201


@pytest.fixture
def zendesk_ticket_settings(settings):
    """Stub mode off, credentials present, custom fields configured."""
    settings.AI_ZENDESK_STUB_MODE = False
    settings.AI_ZENDESK_API_URL = "https://mitlearn.zendesk.com"
    settings.AI_ZENDESK_OAUTH_TOKEN = "zdtoken"  # noqa: S105
    settings.AI_ZENDESK_URL_FIELD_ID = URL_FIELD_ID
    settings.AI_ZENDESK_NAME_FIELD_ID = NAME_FIELD_ID
    settings.AI_ZENDESK_USERNAME_FIELD_ID = USERNAME_FIELD_ID
    return settings


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


async def test_file_support_ticket_posts_to_zendesk(
    zendesk_ticket_settings, mock_httpx_async_client
):
    """A real ticket returns the Zendesk id, tags itself, and carries the page URL."""
    mock_patch = mock_httpx_async_client(
        {"ticket": {"id": 4821}},
        patch_path="ai_chatbots.zendesk.get_async_http_client",
    )

    result = await file_support_ticket(**_ticket_args())

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
    assert kwargs["headers"]["Authorization"] == "Bearer zdtoken"
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
