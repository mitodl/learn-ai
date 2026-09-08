"""AI agent tools and schemas"""

import json

import pytest
from django.conf import settings
from django.core.cache import caches
from httpx import RequestError
from pydantic_core._pydantic_core import ValidationError

from ai_chatbots import learn
from ai_chatbots.constants import (
    ZENDESK_PLATFORM_CATEGORY_IDS,
    ZENDESK_UNIVERSAL_LEARNING_CATEGORY_ID,
)
from ai_chatbots.tools import (
    COURSE_PLATFORM_CACHE_PREFIX,
    LEARN_ERROR_ENVELOPE,
    SearchToolSchema,
    get_video_transcript_chunk,
    search_content_files,
    search_courses,
    search_related_course_content_files,
    search_support_articles,
)

ZENDESK_URL = "https://support.learn.mit.edu"
ZENDESK_SEARCH_URL = f"{ZENDESK_URL}/api/v2/help_center/articles/search.json"
LEARNING_RESOURCES_URL = "https://api.learn.mit.edu/api/v1/learning_resources/"
OCW_CATEGORY = ZENDESK_PLATFORM_CATEGORY_IDS["ocw"]
MITX_CATEGORY = ZENDESK_PLATFORM_CATEGORY_IDS["mitxonline"]
COURSE_ID = "8.20+january-iap_2021"


@pytest.fixture(autouse=True)
def _mock_feature_flag(mocker):
    """Default feature flags to disabled so tests don't touch the durable cache."""
    return mocker.patch("ai_chatbots.learn.is_enabled", return_value=False)


@pytest.fixture
def mock_search_content_files(mocker, content_chunk_results):
    """Fake the Learn content file search with the fixture chunks."""
    params = {"q": "main topics", "resource_readable_id": "x", "limit": 5}
    return mocker.patch(
        "ai_chatbots.learn.search_content_files",
        return_value=learn.SearchResult(
            content_chunk_results["results"], "https://mit.edu/search", params
        ),
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


@pytest.fixture
def mock_support_requests(mocker, zendesk_article_results):
    """
    Mock the two requests a support search makes: the MIT Learn lookup of the
    course platform, and the Zendesk help center article search.
    """

    def _mock_requests(platform="ocw", articles=None, platforms=None):
        def _response(json_value):
            response = mocker.Mock()
            response.json.return_value = json_value
            response.status_code = 200
            response.raise_for_status = mocker.Mock()
            return response

        codes = platforms if platforms is not None else ([platform] if platform else [])
        resources = {"results": [{"platform": {"code": c}} for c in codes]}

        async def _get(url, **kwargs):
            """Answer based on which of the two APIs is being called."""
            if url == LEARNING_RESOURCES_URL:
                return _response(resources)
            return _response(zendesk_article_results if articles is None else articles)

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


async def test_search_support_articles_duplicate_platforms(
    settings, mock_support_requests
):
    """A readable id on two platforms should not scope the search to either."""
    settings.LEARN_ACCESS_TOKEN = "test_token"  # noqa: S105
    mock_client = mock_support_requests(platforms=["mitxonline", "xpro"])

    await search_support_articles.ainvoke({"q": "pay", "state": support_state()})

    assert "category" not in mock_client.get.call_args_list[1].kwargs["params"]


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
    # The platform of the course is looked up against the MIT Learn API, with a
    # tighter timeout than a search since the support search waits on it
    assert mock_client.get.call_args_list[0].args[0] == LEARNING_RESOURCES_URL
    assert mock_client.get.call_args_list[0].kwargs["params"] == {
        "readable_id": COURSE_ID,
        "limit": 5,
    }
    assert (
        mock_client.get.call_args_list[0].kwargs["timeout"]
        == settings.AI_COURSE_PLATFORM_LOOKUP_TIMEOUT
    )
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


@pytest.mark.parametrize(
    ("course_id", "expected_category"),
    [
        # Universal AI courses and programs are hosted on MITx Online, so both
        # the MITx and the Universal Learning categories are searched
        (
            "program-v1:UAI+B2C.1",
            f"{MITX_CATEGORY},{ZENDESK_UNIVERSAL_LEARNING_CATEGORY_ID}",
        ),
        (
            "course-v1:UAI_SOURCE+UAI.HAIM.1",
            f"{MITX_CATEGORY},{ZENDESK_UNIVERSAL_LEARNING_CATEGORY_ID}",
        ),
        # Other MITx Online courses only search the MITx category
        ("course-v1:MITxT+3.012Sx+3T2024", MITX_CATEGORY),
        ("8.20+january-iap_2021", MITX_CATEGORY),
    ],
)
async def test_search_support_articles_uai_category(
    mock_support_requests, course_id, expected_category
):
    """UAI courses should also search the Universal Learning category."""
    mock_client = mock_support_requests(platform="mitxonline")

    results = json.loads(
        await search_support_articles.ainvoke(
            {"q": "GI Bill", "state": support_state(course_id)}
        )
    )

    search_params = mock_client.get.call_args_list[1].kwargs["params"]
    assert search_params["category"] == expected_category
    assert results["metadata"]["parameters"]["category"] == expected_category


async def test_search_support_articles_uai_unknown_platform(mock_support_requests):
    """A UAI course whose platform is unknown still searches its own category."""
    mock_client = mock_support_requests(platform=None)

    await search_support_articles.ainvoke(
        {"q": "GI Bill", "state": support_state("course-v1:UAI_SOURCE+UAI.HAIM.1")}
    )

    search_params = mock_client.get.call_args_list[1].kwargs["params"]
    assert search_params["category"] == ZENDESK_UNIVERSAL_LEARNING_CATEGORY_ID


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


@pytest.mark.usefixtures("_no_retry_sleep")
async def test_search_support_articles_caches_platform_lookup_error(
    settings, mocker, mock_support_requests
):
    """A failed platform lookup should be cached, and not retried every time."""
    settings.AI_COURSE_PLATFORM_ERROR_CACHE_DURATION = 60
    cache_write = mocker.spy(caches["default"], "aset")
    mock_client = mock_support_requests()
    connection_error = RequestError("Connection error")

    async def _get(url, **kwargs):
        """Fail every platform lookup, answer every article search."""
        if url == LEARNING_RESOURCES_URL:
            raise connection_error
        response = mocker.Mock()
        response.status_code = 200
        response.json.return_value = {"results": []}
        response.raise_for_status = mocker.Mock()
        return response

    mock_client.get = mocker.AsyncMock(side_effect=_get)

    for query in ("certificate", "refund"):
        results = json.loads(
            await search_support_articles.ainvoke(
                {"q": query, "state": support_state()}
            )
        )
        assert results["metadata"]["platform"] is None

    lookup_urls = [call.args[0] for call in mock_client.get.call_args_list]
    # Only the first support question pays for the failing lookup; the second
    # reads the cached failure
    assert lookup_urls.count(LEARNING_RESOURCES_URL) == 1
    assert lookup_urls.count(ZENDESK_SEARCH_URL) == 2
    # Cached briefly, so a transient outage does not blank out the platform of
    # the course for the whole of AI_COURSE_PLATFORM_CACHE_DURATION
    cache_write.assert_called_once_with(
        f"{COURSE_PLATFORM_CACHE_PREFIX}{COURSE_ID}", "", 60
    )


async def test_search_support_articles_warns_on_empty_category(
    mocker, mock_support_requests
):
    """An empty result from a category-scoped search should be logged."""
    mock_log = mocker.patch("ai_chatbots.tools.log.warning")
    mock_support_requests(platform="ocw", articles={"results": []})

    results = json.loads(
        await search_support_articles.ainvoke(
            {"q": "certificate", "state": support_state()}
        )
    )

    assert results["results"] == []
    # A stale category id looks exactly like this, so it should not pass silently
    assert mock_log.call_count == 1
    assert mock_log.call_args.args[1:] == (OCW_CATEGORY, "ocw", COURSE_ID)


async def test_search_support_articles_no_warning_without_category(
    mocker, mock_support_requests
):
    """An unscoped search coming back empty is not worth a warning."""
    mock_log = mocker.patch("ai_chatbots.tools.log.warning")
    mock_support_requests(platform="mitpe", articles={"results": []})

    results = json.loads(
        await search_support_articles.ainvoke(
            {"q": "certificate", "state": support_state()}
        )
    )

    assert results["results"] == []
    mock_log.assert_not_called()


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
    assert result == {
        "results": [],
        "support_center_url_to_offer_if_no_article_answers": None,
        "citation_sources": {},
        "metadata": {},
    }
    mock_client.get.assert_not_called()


@pytest.mark.usefixtures("_no_retry_sleep")
async def test_search_support_articles_portal_error(mocker, mock_support_requests):
    """An empty result is returned if the support portal is unavailable."""
    mock_client = mock_support_requests()
    mock_client.get = mocker.AsyncMock(side_effect=RequestError("Connection error"))

    results = json.loads(
        await search_support_articles.ainvoke({"q": "refund", "state": support_state()})
    )

    assert results == {
        "results": [],
        "support_center_url_to_offer_if_no_article_answers": settings.AI_ZENDESK_URL,
        "citation_sources": {},
        "metadata": {},
    }


def test_invalid_support_article_params():
    """Test that invalid support search parameters raise a validation error."""
    with pytest.raises(ValidationError):
        search_support_articles.invoke({"state": support_state()})


async def test_get_video_transcript_chunk(
    mocker, video_transcript_content_chunk_results
):
    """Transcript search returns chunk content and the URL it searched."""
    chunks = video_transcript_content_chunk_results["results"]
    params = {"q": "eigenvalues", "edx_module_id": "asset-1", "limit": 4}
    search = mocker.patch(
        "ai_chatbots.learn.search_transcript",
        return_value=learn.SearchResult(chunks, "https://mit.edu/transcripts", params),
    )

    results = json.loads(
        await get_video_transcript_chunk.ainvoke(
            {"q": "eigenvalues", "state": {"transcript_asset_id": ["asset-1"]}}
        )
    )

    search.assert_called_once_with("eigenvalues", "asset-1")
    assert results == {
        "results": [{"chunk_content": r["chunk_content"]} for r in chunks],
        "metadata": {"search_url": "https://mit.edu/transcripts", "parameters": params},
    }


async def test_get_video_transcript_chunk_error(mocker):
    """A Learn failure is reported to the LLM as the shared error envelope."""
    mocker.patch("ai_chatbots.learn.search_transcript", side_effect=learn.LearnError("502", "https://mit.edu/transcripts", {}))

    result = await get_video_transcript_chunk.ainvoke(
        {"q": "eigenvalues", "state": {"transcript_asset_id": ["asset-1"]}}
    )

    assert result == LEARN_ERROR_ENVELOPE


def test_offered_by_enum_is_named_offered_by():
    """The enum generated for offered_by must not be labelled resource_type."""
    defs = SearchToolSchema.model_json_schema()["$defs"]

    assert "offered_by" in defs


@pytest.mark.parametrize("no_collection_name", [True, False])
@pytest.mark.parametrize("platform", [None, "ocw"])
async def test_search_content_files(
    mock_search_content_files,
    syllabus_agent_state,
    content_chunk_results,
    no_collection_name,
    platform,
):
    """search_content_files searches the current course and shapes chunks + citations."""
    if no_collection_name:
        syllabus_agent_state.pop("collection_name")
    if platform:
        syllabus_agent_state["platform"] = [platform]

    results = json.loads(
        await search_content_files.ainvoke(
            {"q": "main topics", "state": syllabus_agent_state}
        )
    )

    mock_search_content_files.assert_called_once_with(
        "main topics",
        syllabus_agent_state["course_id"][-1],
        collection_name=None if no_collection_name else "vector512",
        platform=platform,
    )
    chunks = content_chunk_results["results"]
    assert results["results"] == [
        {
            "id": c["url"],
            "chunk_content": c.get("chunk_content"),
            "run_title": c.get("run_title"),
        }
        for c in chunks
    ]
    assert results["citation_sources"] == {
        c["url"]: {
            "citation_url": c["url"],
            "citation_title": c.get("title") or c["content_title"],
        }
        for c in chunks
        if c["url"]
    }
    assert results["metadata"] == {
        "search_url": "https://mit.edu/search",
        "parameters": mock_search_content_files.return_value.params,
    }


async def test_search_content_files_cites_each_file(
    mock_search_content_files, syllabus_agent_state, content_chunk_results
):
    """Chunks sharing a resource_point_id should each keep their own citation url."""
    for idx, result in enumerate(content_chunk_results["results"]):
        # every chunk of a course carries the same resource-level point id
        result["resource_point_id"] = "028cbf92-dded-5f7b-8b19-cd9c5a8b5c41"
        result["url"] = f"https://mit.edu/file/{idx}"

    results = json.loads(
        await search_content_files.ainvoke(
            {"q": "learning goals", "state": syllabus_agent_state}
        )
    )

    for idx, result in enumerate(results["results"]):
        assert (
            results["citation_sources"][result["id"]]["citation_url"]
            == f"https://mit.edu/file/{idx}"
        )


async def test_search_content_files_collapses_chunks_of_one_file(
    mock_search_content_files, syllabus_agent_state, content_chunk_results
):
    """Chunks sharing a url should share one citation."""
    url = "https://mit.edu/courses/some-course/pages/syllabus/"
    for result in content_chunk_results["results"]:
        result["url"] = url

    results = json.loads(
        await search_content_files.ainvoke(
            {"q": "learning goals", "state": syllabus_agent_state}
        )
    )

    assert len(results["results"]) == len(content_chunk_results["results"])
    assert {result["id"] for result in results["results"]} == {url}
    assert list(results["citation_sources"]) == [url]


async def test_search_content_files_keeps_uncitable_results(
    mock_search_content_files, syllabus_agent_state, content_chunk_results
):
    """A result with no url is still returned, with no citation to look up."""
    uncitable = [r for r in content_chunk_results["results"] if not r["url"]]
    assert uncitable, "fixture should include results with no url"

    results = json.loads(
        await search_content_files.ainvoke(
            {"q": "learning goals", "state": syllabus_agent_state}
        )
    )

    assert len(results["results"]) == len(content_chunk_results["results"])
    assert len([r for r in results["results"] if r["id"] is None]) == len(uncitable)
    assert None not in results["citation_sources"]


async def test_search_content_files_readable_id_fallback(
    mock_search_content_files, syllabus_agent_state
):
    """Without a course in state, the LLM-supplied readable_id is searched."""
    syllabus_agent_state.pop("course_id")

    await search_content_files.ainvoke(
        {"q": "main topics", "state": syllabus_agent_state, "readable_id": "MITx+8.01"}
    )

    assert mock_search_content_files.call_args.args == ("main topics", "MITx+8.01")


@pytest.mark.parametrize("exclude_canvas", [True, False])
async def test_search_canvas_content_files(
    mock_search_content_files,
    syllabus_agent_state,
    content_chunk_results,
    exclude_canvas,
):
    """Canvas chunks are returned only when exclude_canvas is False."""
    syllabus_agent_state["exclude_canvas"] = [str(exclude_canvas)]
    for result in content_chunk_results["results"]:
        result["platform"]["code"] = "canvas"

    results = json.loads(
        await search_content_files.ainvoke(
            {"q": "main topics", "state": syllabus_agent_state}
        )
    )

    expected = 0 if exclude_canvas else len(content_chunk_results["results"])
    assert len(results["results"]) == expected
    assert len(results["citation_sources"]) == (
        0
        if exclude_canvas
        else len({c["url"] for c in content_chunk_results["results"] if c["url"]})
    )


@pytest.mark.parametrize("exclude_canvas", [True, False])
async def test_search_related_course_content_files(
    mock_search_content_files,
    syllabus_agent_state,
    content_chunk_results,
    exclude_canvas,
):
    """The related-course search passes the related course list and honors exclude_canvas."""
    related = ["course-v1:UAI+12", "course-v1:UAI+13"]
    syllabus_agent_state["related_courses"] = related
    syllabus_agent_state["exclude_canvas"] = [str(exclude_canvas)]
    for result in content_chunk_results["results"]:
        result["platform"]["code"] = "canvas"

    results = json.loads(
        await search_related_course_content_files.ainvoke(
            {"q": "main topics", "state": syllabus_agent_state}
        )
    )

    mock_search_content_files.assert_called_once_with(
        "main topics", related, collection_name="vector512", platform=None
    )
    assert len(results["results"]) == (
        0 if exclude_canvas else len(content_chunk_results["results"])
    )


async def test_search_content_files_error(mocker, syllabus_agent_state):
    """A failed search still reports what it tried, so it isn't mistaken for no results."""
    params = {"q": "main topics", "resource_readable_id": "x", "limit": 5}
    mocker.patch(
        "ai_chatbots.learn.search_content_files",
        side_effect=learn.LearnError("502", "https://mit.edu/search", params),
    )

    result = await search_content_files.ainvoke(
        {"q": "main topics", "state": syllabus_agent_state}
    )

    assert json.loads(result) == {
        "error": "An error occurred while searching",
        "results": [],
        "citation_sources": {},
        "metadata": {"search_url": "https://mit.edu/search", "parameters": params},
    }


DETAIL_URL = "https://learn.mit.edu/?resource="
RESOURCE_ROWS = [
    {
        "id": 1,
        "title": "A",
        "readable_id": "a",
        "next_start_date": "2026-01-01",
        "runs": [
            {"start_date": "2025-01-01", "level": ["Old"], "instructors": ["I1"]},
            {"start_date": "2026-01-01", "level": ["Intro"], "instructors": ["I2"]},
        ],
    },
    {"id": 2, "title": "B", "runs": [{"level": ["Adv"], "instructors": []}]},
    {"id": 3, "title": "C"},
]


@pytest.fixture
def mock_search_learning_resources(mocker):
    """Fake the Learn resource search with three rows exercising run selection."""
    return mocker.patch(
        "ai_chatbots.learn.search_learning_resources",
        return_value=learn.SearchResult(
            [dict(r) for r in RESOURCE_ROWS],
            "https://mit.edu/search",
            {"q": "physics", "limit": 10},
        ),
    )


@pytest.mark.parametrize(
    ("params", "filters"),
    [
        ({}, {}),
        (
            {"resource_type": ["course", "video"], "free": True},
            {"resource_type": ["course", "video"], "free": True},
        ),
        (
            {"resource_type": None, "offered_by": ["xpro"], "certification": False},
            {"offered_by": ["xpro"], "certification": False},
        ),
    ],
)
async def test_search_courses(
    settings, mock_search_learning_resources, params, filters
):
    """search_courses passes stringified non-None filters and shapes rows for the LLM."""
    settings.AI_MIT_SEARCH_DETAIL_URL = DETAIL_URL

    results = json.loads(
        await search_courses.ainvoke(
            {
                "q": "physics",
                "state": {"search_url": ["https://mit.edu/search"]},
                **params,
            }
        )
    )

    mock_search_learning_resources.assert_called_once_with(
        "physics", filters, search_url="https://mit.edu/search"
    )
    base = dict.fromkeys(
        (
            "readable_id",
            "description",
            "offered_by",
            "free",
            "certification",
            "resource_type",
        )
    )
    assert results["results"] == [
        {
            **base,
            "id": 1,
            "title": "A",
            "readable_id": "a",
            "url": f"{DETAIL_URL}1",
            "level": ["Intro"],
            "instructors": ["I2"],
        },
        {
            **base,
            "id": 2,
            "title": "B",
            "url": f"{DETAIL_URL}2",
            "level": ["Adv"],
            "instructors": [],
        },
        {**base, "id": 3, "title": "C", "url": f"{DETAIL_URL}3"},
    ]
    assert results["metadata"] == {
        "search_url": "https://mit.edu/search",
        "parameters": {"q": "physics", "limit": 10},
    }


async def test_search_courses_without_search_url(mock_search_learning_resources):
    """Without a search_url in state, the default search URL is used."""
    await search_courses.ainvoke({"q": "physics", "state": {}})

    mock_search_learning_resources.assert_called_once_with(
        "physics", {}, search_url=None
    )


async def test_search_courses_error(mocker):
    """A failed search still reports what it tried, so it isn't mistaken for no results."""
    params = {"q": "physics", "limit": 10}
    mocker.patch(
        "ai_chatbots.learn.search_learning_resources",
        side_effect=learn.LearnError("502", "https://mit.edu/search", params),
    )

    result = await search_courses.ainvoke(
        {"q": "physics", "state": {"search_url": ["https://mit.edu/search"]}}
    )

    assert json.loads(result) == {
        "error": "An error occurred while searching",
        "results": [],
        "metadata": {"search_url": "https://mit.edu/search", "parameters": params},
    }
