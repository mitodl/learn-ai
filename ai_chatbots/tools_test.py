"""AI agent tools and schemas"""

import json

import pytest
from httpx import RequestError
from pydantic_core._pydantic_core import ValidationError

from ai_chatbots.tools import (
    search_content_files,
    search_courses,
    search_related_course_content_files,
    search_support_articles,
)

ZENDESK_PORTAL_URLS = {
    "mitxonline": "https://mitxonline.zendesk.com",
    "ocw": "https://mitocw.zendesk.com",
    "mitlearn": "https://support.learn.mit.edu",
}


@pytest.fixture(autouse=True)
def _mock_feature_flag(mocker):
    """Default feature flags to disabled so tests don't touch the durable cache."""
    return mocker.patch("ai_chatbots.tools.feature_is_enabled", return_value=False)


@pytest.fixture
def mock_get_resources(mock_httpx_async_client, search_results):
    """Mock httpx async client for resource search tests."""
    return mock_httpx_async_client(
        search_results, patch_path="ai_chatbots.utils.get_async_http_client"
    )


@pytest.fixture
def mock_get_content_files(mock_httpx_async_client, content_chunk_results):
    """Mock httpx async requests for content file tests."""
    return mock_httpx_async_client(
        content_chunk_results, patch_path="ai_chatbots.utils.get_async_http_client"
    )


@pytest.mark.parametrize(
    "params",
    [
        {"q": "physics"},
        {"q": "biology", "resource_type": ["course", "video", "document"]},
        {"q": "chemistry", "resource_type": ["course"], "free": True},
        {
            "q": "astronomy",
            "resource_type": ["course"],
            "free": False,
            "certification": True,
        },
        {
            "q": "ecology",
            "resource_type": ["course"],
            "certification": True,
            "offered_by": ["xpro"],
        },
    ],
)
@pytest.mark.parametrize(
    ("search_url", "limit"),
    [
        ("https://mit.edu/search", 5),
        ("https://mit.edu/vector", 10),
        ("https://mit.edu/vector", 20),
    ],
)
async def test_search_courses(  # noqa: PLR0913
    settings, params, mock_get_resources, search_results, search_url, limit
):
    """Test that the search_courses tool returns expected results w/expected params."""
    settings.AI_MIT_SEARCH_URL = search_url
    settings.AI_MIT_SEARCH_LIMIT = limit
    settings.LEARN_ACCESS_TOKEN = "test_token"  # noqa: S105
    params["state"] = {"search_url": [search_url]}
    results = json.loads(await search_courses.ainvoke(params))
    params.pop("state")
    expected_params = {"limit": limit, **params}
    # The mock client's get method should be called
    mock_get_resources.return_value.get.assert_called_once_with(
        search_url,
        params=expected_params,
        headers={"Authorization": f"Bearer {settings.LEARN_ACCESS_TOKEN}"},
        timeout=30,
    )
    # The vector endpoint can ignore `limit` and return more rows than asked;
    # the tool must cap results at AI_MIT_SEARCH_LIMIT regardless.
    assert len(results["results"]) == min(limit, len(search_results["results"]))


@pytest.mark.parametrize(
    "search_url",
    ["https://mit.edu/search", "https://mit.edu/vector"],
)
async def test_search_courses_override_url(settings, mock_get_resources, search_url):
    """Test that the search_courses tool returns expected results w/expected params."""
    settings.AI_MIT_SEARCH_URL = "http://default_url.edu"
    settings.LEARN_ACCESS_TOKEN = "test_token"  # noqa: S105
    params = {
        "q": "physics",
        "limit": 10,
        "resource_type": ["course"],
        "state": {"search_url": [search_url]},
    }
    await search_courses.ainvoke(params)
    params.pop("state")
    # The mock client's get method should be called
    mock_get_resources.return_value.get.assert_called_once_with(
        search_url,
        params=params,
        headers={"Authorization": f"Bearer {settings.LEARN_ACCESS_TOKEN}"},
        timeout=30,
    )


@pytest.mark.parametrize(
    "params",
    [
        {"foo": "bar"},
        {"resource_type": ["course"]},
        {"q": "biology", "resource_type": ["movie"]},
        {"q": "biology", "resource_type": "course"},
        {"q": "biology", "free": "maybe"},
        {"q": "biology", "certification": "probably"},
        {"q": "biology", "offered_by": ["MIT", "edx"]},
    ],
)
def test_invalid_params(params):
    """Test that invalid parameters raise a validation error."""
    with pytest.raises(ValidationError):
        search_courses.invoke(params)


async def test_httpx_exception(mocker):
    """Test that a request exception returns a JSON error msg"""
    mock_client = mocker.Mock()
    mock_client.get = mocker.AsyncMock(side_effect=RequestError("Connection error"))

    mocker.patch("ai_chatbots.utils.get_async_http_client", return_value=mock_client)

    result = await search_courses.ainvoke(
        {"q": "physics", "state": {"search_url": ["https://test.edu/search"]}}
    )
    assert result == '{"error": "An error occurred while searching"}'


@pytest.mark.usefixtures("_no_retry_sleep")
async def test_search_courses_handles_http_status_error(
    mock_async_get_client, httpx_response
):
    """Persistent 502s should become the tool's JSON error payload."""
    mock_client = mock_async_get_client(
        return_value=httpx_response(502, url="https://test.edu/search")
    )

    result = await search_courses.ainvoke(
        {"q": "physics", "state": {"search_url": ["https://test.edu/search"]}}
    )

    assert result == '{"error": "An error occurred while searching"}'
    # Confirm the retry budget was actually exercised before giving up.
    assert mock_client.get.call_count == 3


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("search_url", "limit"),
    [("https://mit.edu/search", 5), ("https://mit.edu/vector", 10)],
)
@pytest.mark.parametrize("no_collection_name", [True, False])
async def test_search_content_files(  # noqa: PLR0913
    settings,
    mock_get_content_files,
    syllabus_agent_state,
    content_chunk_results,
    search_url,
    limit,
    no_collection_name,
):
    """Test that the search_content_files tool returns expected results w/expected params."""
    settings.AI_MIT_SYLLABUS_URL = search_url
    settings.AI_MIT_CONTENT_SEARCH_LIMIT = limit
    settings.LEARN_ACCESS_TOKEN = "test_token"  # noqa: S105
    expected_params = {
        "q": "main topics",
        "limit": limit,
        "resource_readable_id": syllabus_agent_state["course_id"][-1],
        "collection_name": syllabus_agent_state["collection_name"][-1],
    }
    if no_collection_name:
        expected_params.pop("collection_name")
        syllabus_agent_state.pop("collection_name")

    results = json.loads(
        await search_content_files.ainvoke(
            {"q": "main topics", "state": syllabus_agent_state}
        )
    )
    mock_get_content_files.return_value.get.assert_called_once_with(
        search_url,
        params=expected_params,
        headers={"Authorization": f"Bearer {settings.LEARN_ACCESS_TOKEN}"},
        timeout=30,
    )
    assert len(results["results"]) == len(content_chunk_results["results"])
    assert len(results["citation_sources"]) == len(
        {
            result["resource_point_id"]
            for result in content_chunk_results["results"]
            if result["url"]
        }
    )
    for idx, result in enumerate(content_chunk_results["results"]):
        if content_chunk_results["results"][idx]["url"]:
            assert results["citation_sources"][
                content_chunk_results["results"][idx]["resource_point_id"]
            ] == {
                "citation_url": result.get("url"),
                "citation_title": (result.get("title") or result["content_title"]),
            }


@pytest.mark.django_db
@pytest.mark.parametrize("exclude_canvas", [True, False])
async def test_search_canvas_content_files(
    settings,
    mock_httpx_async_client,
    syllabus_agent_state,
    content_chunk_results,
    exclude_canvas,
):
    """Test that search_content_files returns canvas results only if exclude_canvas is False."""
    settings.LEARN_ACCESS_TOKEN = "test_token"  # noqa: S105

    syllabus_agent_state["exclude_canvas"] = [str(exclude_canvas)]
    for result in content_chunk_results["results"]:
        result["platform"]["code"] = "canvas"

    # Mock httpx async client
    mock_httpx_async_client(
        content_chunk_results, patch_path="ai_chatbots.utils.get_async_http_client"
    )

    results = json.loads(
        await search_content_files.ainvoke(
            {"q": "main topics", "state": syllabus_agent_state}
        )
    )

    assert len(results["results"]) == (
        len(content_chunk_results["results"]) if not exclude_canvas else 0
    )
    assert len(results["citation_sources"]) == (
        len(
            {
                result["resource_point_id"]
                for result in content_chunk_results["results"]
                if result["url"]
            }
        )
        if not exclude_canvas
        else 0
    )


@pytest.mark.parametrize(
    ("resource_type", "offered_by"),
    [
        (None, None),
        (None, ["xpro"]),
        (["course"], None),
    ],
)
async def test_search_courses_handles_none_kwargs(
    settings, mock_get_resources, resource_type, offered_by
):
    """Test that search_courses handles None values for resource_type and offered_by."""
    settings.AI_MIT_SEARCH_URL = "https://mit.edu/search"
    settings.AI_MIT_SEARCH_LIMIT = 10
    settings.LEARN_ACCESS_TOKEN = "test_token"  # noqa: S105

    params = {
        "q": "physics",
        "state": {"search_url": ["https://mit.edu/search"]},
        "resource_type": resource_type,
        "offered_by": offered_by,
    }

    await search_courses.ainvoke(params)

    # Build expected params - None values should be excluded
    expected_params = {"q": "physics", "limit": 10}
    if resource_type is not None:
        expected_params["resource_type"] = resource_type
    if offered_by is not None:
        expected_params["offered_by"] = offered_by

    mock_get_resources.return_value.get.assert_called_once_with(
        "https://mit.edu/search",
        params=expected_params,
        headers={"Authorization": f"Bearer {settings.LEARN_ACCESS_TOKEN}"},
        timeout=30,
    )


@pytest.mark.django_db
@pytest.mark.parametrize("no_collection_name", [True, False])
async def test_search_related_course_content_files(
    settings,
    mock_get_content_files,
    syllabus_agent_state,
    content_chunk_results,
    no_collection_name,
):
    """Test that search_related_course_content_files searches across related courses."""
    settings.AI_MIT_SYLLABUS_URL = "https://mit.edu/search"
    settings.AI_MIT_CONTENT_SEARCH_LIMIT = 5
    settings.LEARN_ACCESS_TOKEN = "test_token"  # noqa: S105

    related = ["course-v1:UAI+12", "course-v1:UAI+13"]
    syllabus_agent_state["related_courses"] = related

    expected_params = {
        "q": "main topics",
        "limit": 5,
        "resource_readable_id": related,
        "collection_name": syllabus_agent_state["collection_name"][-1],
    }
    if no_collection_name:
        expected_params.pop("collection_name")
        syllabus_agent_state.pop("collection_name")

    results = json.loads(
        await search_related_course_content_files.ainvoke(
            {"q": "main topics", "state": syllabus_agent_state}
        )
    )
    mock_get_content_files.return_value.get.assert_called_once_with(
        "https://mit.edu/search",
        params=expected_params,
        headers={"Authorization": f"Bearer {settings.LEARN_ACCESS_TOKEN}"},
        timeout=30,
    )
    assert len(results["results"]) == len(content_chunk_results["results"])


@pytest.mark.parametrize("is_hybrid_enabled", [True, False])
async def test_content_file_search_hybrid_flag(
    settings,
    mock_get_content_files,
    syllabus_agent_state,
    is_hybrid_enabled,
    mocker,
):
    """Test that the hybrid_search parameter is added when the feature flag is enabled."""
    settings.AI_MIT_SYLLABUS_URL = "https://mit.edu/search"
    settings.AI_MIT_CONTENT_SEARCH_LIMIT = 5
    settings.LEARN_ACCESS_TOKEN = "test_token"  # noqa: S105

    mocker.patch(
        "ai_chatbots.tools.feature_is_enabled",
        return_value=is_hybrid_enabled,
    )

    expected_params = {
        "q": "main topics",
        "limit": 5,
        "resource_readable_id": syllabus_agent_state["course_id"][-1],
        "collection_name": syllabus_agent_state["collection_name"][-1],
    }
    if is_hybrid_enabled:
        expected_params["hybrid_search"] = True

    await search_content_files.ainvoke(
        {"q": "main topics", "state": syllabus_agent_state}
    )

    mock_get_content_files.return_value.get.assert_called_once_with(
        "https://mit.edu/search",
        params=expected_params,
        headers={"Authorization": f"Bearer {settings.LEARN_ACCESS_TOKEN}"},
        timeout=30,
    )


@pytest.mark.parametrize("is_hybrid_enabled", [True, False])
async def test_search_courses_hybrid_flag(
    settings,
    mock_get_resources,
    is_hybrid_enabled,
    mocker,
):
    """Test that search_courses adds hybrid_search when the feature flag is enabled."""
    search_url = "https://mit.edu/search"
    settings.AI_MIT_SEARCH_URL = search_url
    settings.AI_MIT_SEARCH_LIMIT = 10
    settings.LEARN_ACCESS_TOKEN = "test_token"  # noqa: S105

    mocker.patch(
        "ai_chatbots.tools.feature_is_enabled",
        return_value=is_hybrid_enabled,
    )

    expected_params = {"q": "physics", "limit": 10}
    if is_hybrid_enabled:
        expected_params["hybrid_search"] = True

    await search_courses.ainvoke(
        {"q": "physics", "state": {"search_url": [search_url]}}
    )

    mock_get_resources.return_value.get.assert_called_once_with(
        search_url,
        params=expected_params,
        headers={"Authorization": f"Bearer {settings.LEARN_ACCESS_TOKEN}"},
        timeout=30,
    )


@pytest.fixture
def mock_get_zendesk_articles(mock_httpx_async_client, zendesk_article_results):
    """Mock httpx async requests for Zendesk support article tests."""
    return mock_httpx_async_client(
        zendesk_article_results, patch_path="ai_chatbots.utils.get_async_http_client"
    )


@pytest.fixture(autouse=True)
def _zendesk_settings(settings):
    """Assign default Zendesk portal settings."""
    settings.AI_ZENDESK_PORTAL_URLS = dict(ZENDESK_PORTAL_URLS)
    settings.AI_ZENDESK_SEARCH_LIMIT = 5
    settings.AI_ZENDESK_ARTICLE_MAX_CHARS = 1500


@pytest.mark.parametrize("platform", ["mitxonline", "ocw", "mitlearn"])
async def test_search_support_articles(
    settings, mock_get_zendesk_articles, zendesk_article_results, platform
):
    """The tool should query only the portal matching the requested platform."""
    settings.AI_ZENDESK_SEARCH_LIMIT = 3
    results = json.loads(
        await search_support_articles.ainvoke(
            {"q": "certificate", "platform": [platform]}
        )
    )

    # No authorization is sent to the public help centers
    mock_get_zendesk_articles.return_value.get.assert_called_once_with(
        f"{ZENDESK_PORTAL_URLS[platform]}/api/v2/help_center/articles/search.json",
        params={"query": "certificate", "per_page": 3},
        headers={},
        timeout=30,
    )
    expected_articles = zendesk_article_results["results"]
    assert len(results["results"]) == len(expected_articles)
    first_result = results["results"][0]
    first_article = expected_articles[0]
    assert first_result["id"] == f"{platform}-{first_article['id']}"
    assert first_result["title"] == first_article["title"]
    assert first_result["url"] == first_article["html_url"]
    assert first_result["platform"] == platform
    # The html body should be converted to plain text
    assert "<" not in first_result["content"]
    assert first_result["content"].startswith(
        "What Are The Two Course Enrollment Tracks?"
    )
    assert results["citation_sources"][first_result["id"]] == {
        "citation_url": first_article["html_url"],
        "citation_title": first_article["title"],
    }
    assert results["metadata"]["parameters"]["platform"] == [platform]


async def test_search_support_articles_all_portals(mock_get_zendesk_articles):
    """All portals should be searched, and interleaved results capped at the limit."""
    results = json.loads(await search_support_articles.ainvoke({"q": "refund"}))

    mock_client = mock_get_zendesk_articles.return_value
    assert mock_client.get.call_count == len(ZENDESK_PORTAL_URLS)
    assert [call.args[0] for call in mock_client.get.call_args_list] == [
        f"{url}/api/v2/help_center/articles/search.json"
        for url in ZENDESK_PORTAL_URLS.values()
    ]
    # 3 articles per portal, interleaved by relevance rank and capped at 5
    assert [result["platform"] for result in results["results"]] == [
        "mitxonline",
        "ocw",
        "mitlearn",
        "mitxonline",
        "ocw",
    ]
    assert results["metadata"]["parameters"]["platform"] == list(ZENDESK_PORTAL_URLS)


async def test_search_support_articles_truncates_content(
    settings, mock_get_zendesk_articles
):
    """Article text should be truncated to the configured max length."""
    settings.AI_ZENDESK_ARTICLE_MAX_CHARS = 20
    results = json.loads(
        await search_support_articles.ainvoke(
            {"q": "certificate", "platform": ["mitxonline"]}
        )
    )
    assert [len(result["content"]) for result in results["results"]] == [20, 20, 20]


async def test_search_support_articles_skips_unconfigured_portals(
    settings, mock_get_zendesk_articles
):
    """Portals without a configured url should not be searched."""
    settings.AI_ZENDESK_PORTAL_URLS = {**ZENDESK_PORTAL_URLS, "ocw": ""}
    results = json.loads(await search_support_articles.ainvoke({"q": "refund"}))

    assert {result["platform"] for result in results["results"]} == {
        "mitxonline",
        "mitlearn",
    }
    assert mock_get_zendesk_articles.return_value.get.call_count == 2


async def test_search_support_articles_no_portals(settings, mock_get_zendesk_articles):
    """An empty result is returned if no portal is configured for the platform."""
    settings.AI_ZENDESK_PORTAL_URLS = {**ZENDESK_PORTAL_URLS, "ocw": ""}
    result = json.loads(
        await search_support_articles.ainvoke({"q": "credit", "platform": ["ocw"]})
    )
    assert result == {"results": []}
    mock_get_zendesk_articles.return_value.get.assert_not_called()


@pytest.mark.usefixtures("_no_retry_sleep")
async def test_search_support_articles_portal_error(
    mocker, mock_httpx_async_client, zendesk_article_results
):
    """A failing portal should not prevent results from the other portals."""
    mock_response = mocker.Mock()
    mock_response.json.return_value = zendesk_article_results
    mock_response.status_code = 200
    mock_response.raise_for_status = mocker.Mock()

    connection_error = RequestError("Connection error")

    def _get(url, **kwargs):
        """Fail every request to the MITx Online portal."""
        if url.startswith(ZENDESK_PORTAL_URLS["mitxonline"]):
            raise connection_error
        return mock_response

    mock_client = mock_httpx_async_client(zendesk_article_results)
    mock_client.get = mocker.AsyncMock(side_effect=_get)
    mocker.patch("ai_chatbots.utils.get_async_http_client", return_value=mock_client)

    results = json.loads(await search_support_articles.ainvoke({"q": "refund"}))

    # The failing portal exhausts its retries and is dropped from the results
    assert {result["platform"] for result in results["results"]} == {"ocw", "mitlearn"}


@pytest.mark.parametrize(
    "params",
    [
        {"platform": ["mitxonline"]},
        {"q": "certificate", "platform": ["mitx"]},
        {"q": "certificate", "platform": "mitxonline"},
    ],
)
def test_invalid_support_article_params(params):
    """Test that invalid support search parameters raise a validation error."""
    with pytest.raises(ValidationError):
        search_support_articles.invoke(params)
