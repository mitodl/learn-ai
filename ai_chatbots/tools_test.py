"""AI agent tools and schemas"""

import json

import pytest
from pydantic_core._pydantic_core import ValidationError
from requests import RequestException

from ai_chatbots.tools import search_content_files, search_courses


@pytest.fixture
def mock_get_resources(mocker, search_results):
    """Mock resource requests.get for all tests."""
    return mocker.patch(
        "ai_chatbots.tools.requests.get",
        return_value=mocker.Mock(
            json=mocker.Mock(return_value=search_results), status_code=200
        ),
    )


@pytest.fixture
def mock_get_content_files(mocker, content_chunk_results):
    """Mock resource requests.get for all tests."""
    return mocker.patch(
        "ai_chatbots.tools.requests.get",
        return_value=mocker.Mock(
            json=mocker.Mock(return_value=content_chunk_results), status_code=200
        ),
    )


@pytest.mark.parametrize(
    "params",
    [
        {"q": "physics"},
        {"q": "biology", "resource_type": ["course", "video", "article"]},
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
    [("https://mit.edu/search", 5), ("https://mit.edu/vector", 10)],
)
def test_search_courses(  # noqa: PLR0913
    settings, params, mock_get_resources, search_results, search_url, limit
):
    """Test that the search_courses tool returns expected results w/expected params."""
    settings.AI_MIT_SEARCH_URL = search_url
    settings.AI_MIT_SEARCH_LIMIT = limit
    params["state"] = {"search_url": [search_url]}
    results = json.loads(search_courses.invoke(params))
    params.pop("state")
    expected_params = {"limit": limit, **params}
    mock_get_resources.assert_called_once_with(
        search_url, params=expected_params, timeout=30
    )
    assert len(results["results"]) == len(search_results["results"])


@pytest.mark.parametrize(
    "search_url",
    ["https://mit.edu/search", "https://mit.edu/vector"],
)
def test_search_courses_override_url(settings, mock_get_resources, search_url):
    """Test that the search_courses tool returns expected results w/expected params."""
    settings.AI_MIT_SEARCH_URL = "http://default_url.edu"
    params = {
        "q": "physics",
        "limit": 10,
        "resource_type": ["course"],
        "state": {"search_url": [search_url]},
    }
    search_courses.invoke(params)
    params.pop("state")
    mock_get_resources.assert_called_once_with(search_url, params=params, timeout=30)


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


def test_request_exception(mocker):
    """Test that a request exception returns a JSON error msg"""
    mocker.patch("ai_chatbots.tools.requests.get", side_effect=RequestException)
    result = search_courses.invoke(
        {"q": "physics", "state": {"search_url": ["https://test.edu/search"]}}
    )
    assert result == '{"error": "An error occurred while searching"}'


@pytest.mark.parametrize(
    ("search_url", "limit"),
    [("https://mit.edu/search", 5), ("https://mit.edu/vector", 10)],
)
@pytest.mark.parametrize("no_collection_name", [True, False])
def test_search_content_files(  # noqa: PLR0913
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
        search_content_files.invoke({"q": "main topics", "state": syllabus_agent_state})
    )
    mock_get_content_files.assert_called_once_with(
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


@pytest.mark.parametrize("exclude_canvas", [True, False])
def test_search_canvas_content_files(
    settings, mocker, syllabus_agent_state, content_chunk_results, exclude_canvas
):
    """Test that search_content_files returns canvas results only if exclude_canvas is False."""
    settings.LEARN_ACCESS_TOKEN = "test_token"  # noqa: S105

    syllabus_agent_state["exclude_canvas"] = [str(exclude_canvas)]
    for result in content_chunk_results["results"]:
        result["platform"]["code"] = "canvas"
    mocker.patch(
        "ai_chatbots.tools.requests.get",
        return_value=mocker.Mock(
            json=mocker.Mock(return_value=content_chunk_results), status_code=200
        ),
    )
    results = json.loads(
        search_content_files.invoke({"q": "main topics", "state": syllabus_agent_state})
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
