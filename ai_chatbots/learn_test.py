"""Tests for the Learn API client"""

import httpx
import pytest

from ai_chatbots import learn

RESOURCES_URL = "https://api.learn.mit.edu/api/v1/learning_resources/"


@pytest.fixture(autouse=True)
def _learn_settings(settings):
    settings.AI_MIT_LEARNING_RESOURCES_URL = RESOURCES_URL
    settings.AI_COURSE_PLATFORM_LOOKUP_TIMEOUT = 5
    settings.LEARN_ACCESS_TOKEN = "learn-token"  # noqa: S105


async def test_learning_resources_sends_bearer_token(
    mock_async_get_client, httpx_response
):
    """The configured Learn token is sent as a bearer header."""
    client = mock_async_get_client(
        return_value=httpx_response(200, json_body={"results": [{"id": 1}]})
    )

    result = await learn.learning_resources("MITx+6.00.1x")

    assert result == [{"id": 1}]
    client.get.assert_called_once_with(
        RESOURCES_URL,
        params={"readable_id": "MITx+6.00.1x", "limit": 5},
        headers={"Authorization": "Bearer learn-token"},
        timeout=5,
    )


async def test_no_authorization_header_without_token(
    settings, mock_async_get_client, httpx_response
):
    """No bearer header is sent when no token is configured; httpx rejects empty ones."""
    settings.LEARN_ACCESS_TOKEN = ""
    client = mock_async_get_client(
        return_value=httpx_response(200, json_body={"results": []})
    )

    assert await learn.learning_resources("MITx+6.00.1x") == []
    assert client.get.call_args.kwargs["headers"] == {}


@pytest.mark.usefixtures("_no_retry_sleep")
async def test_http_error_raises_learn_error(mock_async_get_client, httpx_response):
    """A persistent upstream error becomes LearnError after the retry budget."""
    client = mock_async_get_client(return_value=httpx_response(502))

    with pytest.raises(learn.LearnError):
        await learn.learning_resources("MITx+6.00.1x")
    assert client.get.call_count == 3


@pytest.mark.usefixtures("_no_retry_sleep")
async def test_transport_error_raises_learn_error(mock_async_get_client):
    """A connection failure becomes LearnError."""
    mock_async_get_client(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(learn.LearnError):
        await learn.learning_resources("MITx+6.00.1x")


async def test_non_json_body_raises_learn_error(mock_async_get_client):
    """A non-JSON body is a LearnError, never a bare ValueError."""
    mock_async_get_client(
        return_value=httpx.Response(
            200, text="<html>", request=httpx.Request("GET", RESOURCES_URL)
        )
    )

    with pytest.raises(learn.LearnError) as excinfo:
        await learn.learning_resources("MITx+6.00.1x")
    assert not isinstance(excinfo.value, ValueError)


TRANSCRIPT_URL = "https://api.learn.mit.edu/api/v0/vector_content_files_search/"


@pytest.fixture
def mock_hybrid_flag(mocker):
    """Control the hybrid search feature flag as seen by the client."""
    return mocker.patch("ai_chatbots.learn.is_enabled", return_value=False)


@pytest.mark.parametrize("hybrid", [True, False])
async def test_search_transcript(
    settings, mock_async_get_client, httpx_response, mock_hybrid_flag, hybrid
):
    """Transcript search owns its URL, limit and the hybrid flag."""
    settings.AI_MIT_VIDEO_TRANSCRIPT_URL = TRANSCRIPT_URL
    settings.AI_MIT_TRANSCRIPT_SEARCH_LIMIT = 4
    settings.REQUESTS_TIMEOUT = 30
    mock_hybrid_flag.return_value = hybrid
    chunks = [{"chunk_content": "a"}, {"chunk_content": "b"}]
    client = mock_async_get_client(
        return_value=httpx_response(200, json_body={"results": chunks})
    )

    result = await learn.search_transcript("eigenvalues", "asset-1")

    expected_params = {"q": "eigenvalues", "edx_module_id": "asset-1", "limit": 4}
    if hybrid:
        expected_params["hybrid_search"] = True
    client.get.assert_called_once_with(
        TRANSCRIPT_URL,
        params=expected_params,
        headers={"Authorization": "Bearer learn-token"},
        timeout=30,
    )
    assert result == learn.SearchResult(chunks, TRANSCRIPT_URL, expected_params)
