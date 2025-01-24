"""AI agent tools and schemas"""

import json

import pytest
from pydantic_core._pydantic_core import ValidationError
from requests import RequestException

from ai_chatbots.tools import search_courses


@pytest.fixture(autouse=True)
def mock_get(mocker, search_results):
    """Mock requests.get for all tests."""
    return mocker.patch(
        "ai_chatbots.tools.requests.get",
        return_value=mocker.Mock(
            json=mocker.Mock(return_value=search_results), status_code=200
        ),
    )


@pytest.mark.parametrize(
    "params",
    [
        {"q": "physics"},
        {"q": "biology", "resource_type": ["course", "video"]},
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
    settings, params, mock_get, search_results, search_url, limit
):
    """Test the search_courses tool."""
    settings.AI_MIT_SEARCH_URL = search_url
    settings.AI_MIT_SEARCH_LIMIT = limit
    expected_params = {"limit": limit, **params}
    results = json.loads(search_courses.invoke(params))
    mock_get.assert_called_once_with(search_url, params=expected_params, timeout=30)
    assert len(results["results"]) == len(search_results["results"])


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
    result = search_courses.invoke({"q": "physics"})
    assert result == '{"error": "An error occurred while searching"}'
