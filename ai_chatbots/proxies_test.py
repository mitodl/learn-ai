"""Tests for ai_chatbots proxies functionality"""

from unittest.mock import ANY
from urllib.parse import urljoin

import pytest

from ai_chatbots.constants import AI_ANONYMOUS_USER
from ai_chatbots.proxies import LiteLLMProxy


@pytest.fixture(autouse=True)
def proxy_settings(settings):
    """Enable proxy use via settings"""
    settings.AI_PROXY_URL = "http://localhost:8000"
    settings.AI_PROXY_AUTH_TOKEN = "fake"  # noqa: S105
    return settings


@pytest.mark.parametrize(
    ("user_id", "multiplier", "endpoint"),
    [
        ("user1", 5, "new"),
        (AI_ANONYMOUS_USER, 10, "update"),
    ],
)
def test_litellm_create_user(settings, mocker, user_id, multiplier, endpoint):
    """Test that correct api calls are made to create a LitelLM proxy user"""
    mock_request_post = mocker.patch("ai_chatbots.proxies.requests.post")
    LiteLLMProxy().create_proxy_user(user_id, endpoint)

    expected_multiplier = multiplier if user_id == "anonymous" else 1
    expected_body = {
        "user_id": user_id,
        "max_parallel_requests": settings.AI_MAX_PARALLEL_REQUESTS
        * expected_multiplier,
        "tpm_limit": settings.AI_TPM_LIMIT * expected_multiplier,
        "rpm_limit": settings.AI_RPM_LIMIT * expected_multiplier,
        "max_budget": settings.AI_MAX_BUDGET * expected_multiplier,
        "budget_duration": settings.AI_BUDGET_DURATION,
    }

    mock_request_post.assert_called_once_with(
        urljoin(settings.AI_PROXY_URL, f"/customer/{endpoint}"),
        json=expected_body,
        timeout=settings.REQUESTS_TIMEOUT,
        headers={"Authorization": f"Bearer {settings.AI_PROXY_AUTH_TOKEN}"},
    )


def test_litellm_create_user_exists(settings, mocker):
    """If user already exists, update the user instead"""
    mock_request_post = mocker.patch(
        "ai_chatbots.proxies.requests.post",
        side_effect=[
            Exception("Error, duplicate key value violates unique constraint"),
            mocker.Mock(),
        ],
    )

    LiteLLMProxy().create_proxy_user("user1")

    assert mock_request_post.call_count == 2
    for endpoint in ("new", "update"):
        mock_request_post.assert_any_call(
            urljoin(settings.AI_PROXY_URL, f"/customer/{endpoint}"),
            json=ANY,
            timeout=settings.REQUESTS_TIMEOUT,
            headers={"Authorization": f"Bearer {settings.AI_PROXY_AUTH_TOKEN}"},
        )


@pytest.mark.parametrize(
    ("url_key", "api_key"),
    [
        ("base_url", "openai_api_key"),
        ("proxy_url", "proxy_key"),
    ],
)
def test_get_api_kwargs(settings, url_key, api_key):
    """Test that the correct api kwargs are returned for the proxy"""
    assert LiteLLMProxy().get_api_kwargs(base_url_key=url_key, api_key_key=api_key) == {
        f"{url_key}": settings.AI_PROXY_URL,
        f"{api_key}": settings.AI_PROXY_AUTH_TOKEN,
    }


def test_get_additional_kwargs(mocker):
    """Test that the correct additional kwargs are returned for the proxy"""
    mock_agent = mocker.Mock(user_id="user1", JOB_ID="job1", TASK_NAME="task1")
    assert LiteLLMProxy().get_additional_kwargs(mock_agent) == {
        "user": mock_agent.user_id,
        "store": True,
        "extra_body": {
            "drop_params": True,
            "metadata": {
                "tags": [
                    f"jobID:{mock_agent.JOB_ID}",
                    f"taskName:{mock_agent.TASK_NAME}",
                ]
            }
        },
    }
