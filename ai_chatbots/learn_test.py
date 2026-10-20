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

    with pytest.raises(learn.LearnError) as excinfo:
        await learn.learning_resources("MITx+6.00.1x")
    assert client.get.call_count == 3
    # tools report what was tried, so a failure isn't mistaken for no results
    assert excinfo.value.url == RESOURCES_URL
    assert excinfo.value.params == {"readable_id": "MITx+6.00.1x", "limit": 5}


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


SYLLABUS_URL = "https://api.learn.mit.edu/api/v0/vector_content_files_search/"


@pytest.mark.parametrize(
    ("readable_id", "collection_name", "hybrid"),
    [
        ("MITx+6.00.1x", "vector512", False),
        ("MITx+6.00.1x", None, True),
        (["course-v1:UAI+12", "course-v1:UAI+13"], None, False),
    ],
)
async def test_search_content_files(  # noqa: PLR0913
    settings,
    mock_async_get_client,
    httpx_response,
    mock_hybrid_flag,
    readable_id,
    collection_name,
    hybrid,
):
    """Content file search takes one course or a list, optional collection."""
    settings.AI_MIT_SYLLABUS_URL = SYLLABUS_URL
    settings.AI_MIT_CONTENT_SEARCH_LIMIT = 6
    mock_hybrid_flag.return_value = hybrid
    chunks = [{"resource_point_id": "p1"}]
    client = mock_async_get_client(
        return_value=httpx_response(200, json_body={"results": chunks})
    )

    result = await learn.search_content_files(
        "main topics", readable_id, collection_name=collection_name
    )

    expected_params = {
        "q": "main topics",
        "resource_readable_id": readable_id,
        "limit": 6,
    }
    if collection_name:
        expected_params["collection_name"] = collection_name
    if hybrid:
        expected_params["hybrid_search"] = True
    assert client.get.call_args.args == (SYLLABUS_URL,)
    assert client.get.call_args.kwargs["params"] == expected_params
    assert result == learn.SearchResult(chunks, SYLLABUS_URL, expected_params)


VECTOR_SEARCH_URL = "https://api.learn.mit.edu/api/v0/vector_learning_resources_search/"
ELASTIC_SEARCH_URL = "https://api.learn.mit.edu/api/v1/learning_resources_search/"


@pytest.mark.parametrize(
    ("backend", "expected_url"),
    [
        (None, VECTOR_SEARCH_URL),
        ("vector", VECTOR_SEARCH_URL),
        ("elastic", ELASTIC_SEARCH_URL),
    ],
)
@pytest.mark.parametrize("hybrid", [True, False])
async def test_search_learning_resources(  # noqa: PLR0913
    settings,
    mock_async_get_client,
    httpx_response,
    mock_hybrid_flag,
    backend,
    expected_url,
    hybrid,
):
    """Resource search picks the vector or elastic endpoint and trims to the limit."""
    settings.AI_MIT_SEARCH_VECTOR_URL = VECTOR_SEARCH_URL
    settings.AI_MIT_SEARCH_ELASTIC_URL = ELASTIC_SEARCH_URL
    settings.AI_MIT_SEARCH_LIMIT = 2
    mock_hybrid_flag.return_value = hybrid
    rows = [{"id": 1}, {"id": 2}, {"id": 3}]
    client = mock_async_get_client(
        return_value=httpx_response(200, json_body={"results": rows})
    )
    filters = {"resource_type": ["course"], "free": True}
    kwargs = {} if backend is None else {"backend": backend}

    result = await learn.search_learning_resources("physics", filters, **kwargs)

    expected_params = {"q": "physics", "limit": 2, **filters}
    if hybrid:
        expected_params["hybrid_search"] = True
    assert client.get.call_args.args == (expected_url,)
    assert client.get.call_args.kwargs["params"] == expected_params
    # The vector endpoint can ignore `limit`; results are capped regardless.
    assert result == learn.SearchResult(rows[:2], expected_url, expected_params)


async def test_search_learning_resources_rejects_unknown_backend():
    """Only the two configured endpoints can be searched."""
    with pytest.raises(ValueError, match="backend"):
        await learn.search_learning_resources("physics", {}, backend="https://x/")


CONTENTFILE_URL = "https://api.learn.mit.edu/api/v1/contentfiles/"
PROBLEM_SET_URL = "https://api.learn.mit.edu/api/v0/tutor/problems/"


async def test_contentfiles(settings, mock_async_get_client, httpx_response):
    """Contentfiles are fetched by edx module id, with an optional timeout."""
    settings.AI_MIT_CONTENTFILE_URL = CONTENTFILE_URL
    rows = [{"edx_module_id": "block1", "content": "<problem/>"}]
    client = mock_async_get_client(
        return_value=httpx_response(200, json_body={"results": rows})
    )

    result = await learn.contentfiles(["block1", "block2"], timeout=10)

    assert result == rows
    client.get.assert_called_once_with(
        CONTENTFILE_URL,
        params={"edx_module_id": ["block1", "block2"]},
        headers={"Authorization": "Bearer learn-token"},
        timeout=10,
    )


async def test_contentfiles_default_timeout(
    settings, mock_async_get_client, httpx_response
):
    """Without an explicit timeout the shared REQUESTS_TIMEOUT applies."""
    settings.AI_MIT_CONTENTFILE_URL = CONTENTFILE_URL
    settings.REQUESTS_TIMEOUT = 17
    client = mock_async_get_client(
        return_value=httpx_response(200, json_body={"results": []})
    )

    assert await learn.contentfiles(["block1"]) == []
    assert client.get.call_args.kwargs["timeout"] == 17


@pytest.mark.parametrize("base_url", [PROBLEM_SET_URL, PROBLEM_SET_URL.rstrip("/")])
@pytest.mark.parametrize(
    ("title", "expected_path"),
    [
        (None, "run-1/"),
        ("Problem Set 1", "run-1/Problem%20Set%201/"),
        # each part stays one path segment, so it can't reach other endpoints
        ("../../v1/learning_resources", "run-1/..%2F..%2Fv1%2Flearning_resources/"),
    ],
)
async def test_problem_set(  # noqa: PLR0913
    settings, mock_async_get_client, httpx_response, base_url, title, expected_path
):
    """Problem set URLs are canonical: single slashes, trailing slash (Learn 301s otherwise)."""
    settings.PROBLEM_SET_URL = base_url
    body = {"problem_set_files": [], "solution_set_files": []}
    client = mock_async_get_client(return_value=httpx_response(200, json_body=body))

    result = await learn.problem_set("run-1", title, timeout=10)

    assert result == body
    assert client.get.call_args.args == (f"{PROBLEM_SET_URL}{expected_path}",)
    assert client.get.call_args.kwargs["params"] == {}


@pytest.mark.parametrize(("run_readable_id", "title"), [("..", None), ("run-1", ".")])
async def test_problem_set_rejects_dot_segments(
    mock_async_get_client, run_readable_id, title
):
    """Dot segments would be resolved by httpx, so they never reach the transport."""
    client = mock_async_get_client()

    with pytest.raises(learn.LearnError):
        await learn.problem_set(run_readable_id, title)
    client.get.assert_not_called()
