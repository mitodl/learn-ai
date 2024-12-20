"""Tests for AI agent services."""

import json
from pathlib import Path

import pytest
from django.conf import settings
from llama_index.core.constants import DEFAULT_TEMPERATURE

from ai_agents.agents import RecommendationAgent
from main.test_utils import assert_json_equal


@pytest.fixture(autouse=True)
def ai_settings(settings):
    """Set the AI settings for the tests."""
    settings.AI_CACHE = "default"
    settings.AI_PROXY_URL = ""
    return settings


@pytest.fixture
def search_results():
    """Return search results for testing."""
    with Path.open("./test_json/search_results.json") as f:
        yield json.loads(f.read())


@pytest.mark.parametrize(
    ("model", "temperature", "instructions"),
    [
        ("gpt-3.5-turbo", 0.1, "Answer this question as best you can"),
        ("gpt-4o", 0.3, None),
        ("gpt-4", None, None),
        (None, None, None),
    ],
)
def test_search_agent_service_initialization_defaults(model, temperature, instructions):
    """Test the RecommendationAgent class instantiation."""
    name = "My search agent"

    search_agent = RecommendationAgent(
        "user",
        name=name,
        model=model,
        temperature=temperature,
        instructions=instructions,
    )
    assert search_agent.model == (model if model else settings.AI_MODEL)
    assert search_agent.temperature == (
        temperature if temperature else DEFAULT_TEMPERATURE
    )
    assert search_agent.instructions == (
        instructions if instructions else search_agent.instructions
    )
    assert search_agent.agent.__class__.__name__ == "OpenAIAgent"
    assert search_agent.agent.agent_worker._llm.model == (  # noqa: SLF001
        model if model else settings.AI_MODEL
    )


def test_clear_chat_history(client, user, chat_history):
    """Test that the RecommendationAgent clears chat_history."""
    search_agent = RecommendationAgent(user.username)
    search_agent.agent.chat_history.extend(chat_history)
    assert len(search_agent.agent.chat_history) == 4
    search_agent.clear_chat_history()
    assert search_agent.agent.chat_history == []


@pytest.mark.django_db
def test_search_agent_tool(settings, mocker, search_results):
    """The search agent tool should be created and function correctly."""
    settings.AI_MIT_SEARCH_LIMIT = 5
    retained_attributes = [
        "title",
        "url",
        "description",
        "offered_by",
        "free",
        "certification",
        "resource_type",
    ]
    raw_results = search_results.get("results")
    expected_results = []
    for resource in raw_results:
        simple_result = {key: resource.get(key) for key in retained_attributes}
        simple_result["instructors"] = resource.get("runs")[-1].get("instructors")
        simple_result["level"] = resource.get("runs")[-1].get("level")
        expected_results.append(simple_result)

    mock_post = mocker.patch(
        "ai_agents.agents.requests.get",
        return_value=mocker.Mock(json=mocker.Mock(return_value=search_results)),
    )
    search_agent = RecommendationAgent("anonymous", name="test agent")
    search_parameters = {
        "q": "physics",
        "resource_type": ["course", "program"],
        "free": False,
        "certificate": True,
        "offered_by": "xpro",
        "limit": 5,
    }
    tool = search_agent.create_tools()[0]
    results = tool._fn(**search_parameters)  # noqa: SLF001
    mock_post.assert_called_once_with(
        settings.AI_MIT_SEARCH_URL, params=search_parameters, timeout=30
    )
    assert_json_equal(json.loads(results), expected_results)


@pytest.mark.django_db
@pytest.mark.parametrize("debug", [True, False])
def test_get_completion(settings, mocker, debug, search_results):
    """Test that the RecommendationAgent get_completion method returns expected values."""
    settings.AI_DEBUG = debug
    metadata = {
        "metadata": {
            "search_parameters": {"q": "physics"},
            "search_results": search_results.get("results"),
            "system_prompt": RecommendationAgent.INSTRUCTIONS,
        }
    }
    comment_metadata = f"\n\n<!-- {json.dumps(metadata)} -->\n\n".encode()
    expected_return_value = [b"Here ", b"are ", b"some ", b"results"]
    if debug:
        expected_return_value.append(comment_metadata)
    mocker.patch(
        "ai_agents.agents.OpenAIAgent.stream_chat",
        return_value=mocker.Mock(response_gen=iter(expected_return_value)),
    )
    search_agent = RecommendationAgent("anonymous", name="test agent")
    search_agent.search_parameters = metadata["metadata"]["search_parameters"]
    search_agent.search_results = metadata["metadata"]["search_results"]
    search_agent.instructions = metadata["metadata"]["system_prompt"]
    search_agent.search_parameters = {"q": "physics"}
    search_agent.search_results = search_results
    results = "".join(
        [
            str(chunk)
            for chunk in search_agent.get_completion(
                "I want to learn physics", debug=debug
            )
        ]
    )
    search_agent.agent.stream_chat.assert_called_once_with("I want to learn physics")
    assert "".join([str(value) for value in expected_return_value]) in results
    if debug:
        assert '\n\n<!-- {"metadata":' in results
