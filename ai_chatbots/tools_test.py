"""AI agent tools and schemas"""

import json

import pytest
from django.core.cache import caches
from httpx import RequestError
from pydantic_core._pydantic_core import ValidationError

from ai_chatbots.constants import ZENDESK_PLATFORM_CATEGORY_IDS
from ai_chatbots.tools import (
    search_content_files,
    search_courses,
    search_related_course_content_files,
    search_support_articles,
)

ZENDESK_URL = "https://support.learn.mit.edu"
ZENDESK_SEARCH_URL = f"{ZENDESK_URL}/api/v2/help_center/articles/search.json"
LEARNING_RESOURCES_URL = "https://api.learn.mit.edu/api/v1/learning_resources/"
OCW_CATEGORY = ZENDESK_PLATFORM_CATEGORY_IDS["ocw"]
COURSE_ID = "8.20+january-iap_2021"


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
def mock_support_requests(mocker, zendesk_article_results):
    """
    Mock the two requests a support search makes: the MIT Learn lookup of the
    course platform, and the Zendesk help center article search.
    """

    def _mock_requests(platform="ocw"):
        def _response(json_value):
            response = mocker.Mock()
            response.json.return_value = json_value
            response.status_code = 200
            response.raise_for_status = mocker.Mock()
            return response

        resources = {"results": [{"platform": {"code": platform}}] if platform else []}

        async def _get(url, **kwargs):
            """Answer based on which of the two APIs is being called."""
            if url == LEARNING_RESOURCES_URL:
                return _response(resources)
            return _response(zendesk_article_results)

        mock_client = mocker.Mock()
        mock_client.get = mocker.AsyncMock(side_effect=_get)
        mocker.patch(
            "ai_chatbots.utils.get_async_http_client", return_value=mock_client
        )
        return mock_client

    return _mock_requests


@pytest.fixture(autouse=True)
def _zendesk_settings(settings):
    """Assign default Zendesk settings."""
    settings.AI_ZENDESK_URL = ZENDESK_URL
    settings.AI_ZENDESK_SEARCH_LIMIT = 5
    settings.AI_ZENDESK_ARTICLE_MAX_CHARS = 1500
    settings.AI_MIT_LEARNING_RESOURCES_URL = LEARNING_RESOURCES_URL


@pytest.fixture(autouse=True)
def _local_platform_cache(mocker):
    """Cache course platforms in local memory rather than redis."""
    cache = caches["default"]
    cache.clear()
    mocker.patch("ai_chatbots.tools.get_django_cache", return_value=cache)
    return cache


def support_state(course_id=COURSE_ID):
    """Return a minimal syllabus agent state for the support search tool."""
    return {"course_id": [course_id]}


async def test_search_support_articles(
    settings, mock_support_requests, zendesk_article_results
):
    """The tool should search the category matching the course platform."""
    settings.AI_ZENDESK_SEARCH_LIMIT = 3
    settings.LEARN_ACCESS_TOKEN = "test_token"  # noqa: S105
    mock_client = mock_support_requests(platform="ocw")

    results = json.loads(
        await search_support_articles.ainvoke(
            {"q": "certificate", "state": support_state()}
        )
    )

    expected_params = {
        "query": "certificate",
        "per_page": 3,
        "category": OCW_CATEGORY,
    }
    # The platform of the course is looked up against the MIT Learn API
    assert mock_client.get.call_args_list[0].args[0] == LEARNING_RESOURCES_URL
    assert mock_client.get.call_args_list[0].kwargs["params"] == {
        "readable_id": COURSE_ID,
        "limit": 1,
    }
    # No authorization is sent to the public help center
    assert mock_client.get.call_args_list[1].args[0] == ZENDESK_SEARCH_URL
    assert mock_client.get.call_args_list[1].kwargs == {
        "params": expected_params,
        "headers": {},
        "timeout": 30,
    }
    expected_articles = zendesk_article_results["results"]
    assert len(results["results"]) == len(expected_articles)
    first_result = results["results"][0]
    first_article = expected_articles[0]
    assert first_result["id"] == str(first_article["id"])
    assert first_result["title"] == first_article["title"]
    assert first_result["url"] == first_article["html_url"]
    # The html body should be converted to plain text
    assert "<" not in first_result["content"]
    assert first_result["content"].startswith(
        "What Are The Two Course Enrollment Tracks?"
    )
    assert results["citation_sources"][first_result["id"]] == {
        "citation_url": first_article["html_url"],
        "citation_title": first_article["title"],
    }
    assert results["metadata"] == {
        "search_url": ZENDESK_SEARCH_URL,
        "parameters": expected_params,
        "platform": "ocw",
    }


@pytest.mark.parametrize(
    ("platform", "expected_category"),
    [
        ("ocw", ZENDESK_PLATFORM_CATEGORY_IDS["ocw"]),
        ("mitxonline", ZENDESK_PLATFORM_CATEGORY_IDS["mitxonline"]),
        ("edx", ZENDESK_PLATFORM_CATEGORY_IDS["edx"]),
        ("xpro", ZENDESK_PLATFORM_CATEGORY_IDS["xpro"]),
        ("emeritus", ZENDESK_PLATFORM_CATEGORY_IDS["xpro"]),
        # Platforms without a help center category of their own, and courses
        # whose platform cannot be determined, search the whole support center
        ("mitpe", None),
        ("canvas", None),
        (None, None),
    ],
)
async def test_search_support_articles_platform_category(
    mock_support_requests, platform, expected_category
):
    """Each platform should be mapped to its own help center category."""
    mock_client = mock_support_requests(platform=platform)

    results = json.loads(
        await search_support_articles.ainvoke(
            {"q": "certificate", "state": support_state()}
        )
    )

    search_params = mock_client.get.call_args_list[1].kwargs["params"]
    if expected_category:
        assert search_params["category"] == expected_category
    else:
        assert "category" not in search_params
    assert results["metadata"]["platform"] == platform


@pytest.mark.parametrize("state", [{}, {"course_id": [None]}])
async def test_search_support_articles_without_course(mock_support_requests, state):
    """With no course under discussion, the whole support center is searched."""
    mock_client = mock_support_requests()

    results = json.loads(
        await search_support_articles.ainvoke({"q": "refund", "state": state})
    )

    # The platform lookup is skipped entirely
    mock_client.get.assert_called_once_with(
        ZENDESK_SEARCH_URL,
        params={"query": "refund", "per_page": 5},
        headers={},
        timeout=30,
    )
    assert results["metadata"]["platform"] is None


async def test_search_support_articles_caches_platform(mock_support_requests):
    """The platform of a course should only be looked up once."""
    mock_client = mock_support_requests(platform="ocw")

    for query in ("certificate", "refund"):
        results = json.loads(
            await search_support_articles.ainvoke(
                {"q": query, "state": support_state()}
            )
        )
        assert results["metadata"]["parameters"]["category"] == OCW_CATEGORY

    lookup_urls = [call.args[0] for call in mock_client.get.call_args_list]
    assert lookup_urls.count(LEARNING_RESOURCES_URL) == 1
    assert lookup_urls.count(ZENDESK_SEARCH_URL) == 2


@pytest.mark.usefixtures("_no_retry_sleep")
async def test_search_support_articles_platform_lookup_error(
    mocker, zendesk_article_results
):
    """A failed platform lookup should not prevent an unfiltered search."""

    def _response(json_value):
        response = mocker.Mock()
        response.json.return_value = json_value
        response.status_code = 200
        response.raise_for_status = mocker.Mock()
        return response

    connection_error = RequestError("Connection error")

    async def _get(url, **kwargs):
        """Fail every platform lookup."""
        if url == LEARNING_RESOURCES_URL:
            raise connection_error
        return _response(zendesk_article_results)

    mock_client = mocker.Mock()
    mock_client.get = mocker.AsyncMock(side_effect=_get)
    mocker.patch("ai_chatbots.utils.get_async_http_client", return_value=mock_client)

    results = json.loads(
        await search_support_articles.ainvoke(
            {"q": "certificate", "state": support_state()}
        )
    )

    assert len(results["results"]) == len(zendesk_article_results["results"])
    assert "category" not in results["metadata"]["parameters"]
    assert results["metadata"]["platform"] is None


async def test_search_support_articles_truncates_content(
    settings, mock_support_requests
):
    """Article text should be truncated to the configured max length."""
    settings.AI_ZENDESK_ARTICLE_MAX_CHARS = 20
    mock_support_requests()
    results = json.loads(
        await search_support_articles.ainvoke(
            {"q": "certificate", "state": support_state()}
        )
    )
    assert [len(result["content"]) for result in results["results"]] == [20, 20, 20]


async def test_search_support_articles_no_portal(settings, mock_support_requests):
    """An empty result is returned if no support portal url is configured."""
    settings.AI_ZENDESK_URL = ""
    mock_client = mock_support_requests()
    result = json.loads(
        await search_support_articles.ainvoke({"q": "credit", "state": support_state()})
    )
    assert result == {"results": []}
    mock_client.get.assert_not_called()


@pytest.mark.usefixtures("_no_retry_sleep")
async def test_search_support_articles_portal_error(mocker, mock_support_requests):
    """An empty result is returned if the support portal is unavailable."""
    mock_client = mock_support_requests()
    mock_client.get = mocker.AsyncMock(side_effect=RequestError("Connection error"))

    results = json.loads(
        await search_support_articles.ainvoke({"q": "refund", "state": support_state()})
    )

    assert results == {"results": []}


def test_invalid_support_article_params():
    """Test that invalid support search parameters raise a validation error."""
    with pytest.raises(ValidationError):
        search_support_articles.invoke({"state": support_state()})
