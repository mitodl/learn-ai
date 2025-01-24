"""Tests for AI chatbots."""

import json

import pytest
from django.conf import settings

from ai_chatbots.chatbots import DEFAULT_TEMPERATURE, ResourceRecommendationBot
from ai_chatbots.conftest import MockAsyncIterator
from ai_chatbots.constants import LLMClassEnum
from ai_chatbots.factories import AIMessageChunkFactory, ToolMessageFactory
from main.test_utils import assert_json_equal


@pytest.mark.parametrize(
    ("model", "temperature", "instructions"),
    [
        ("gpt-3.5-turbo", 0.1, "Answer this question as best you can"),
        ("gpt-4o", 0.3, None),
        ("gpt-4", None, None),
        (None, None, None),
    ],
)
def test_chatbot_initialization_defaults(model, temperature, instructions):
    """Test the ResourceRecommendationBot class instantiation."""
    name = "My search bot"

    chatbot = ResourceRecommendationBot(
        "user",
        name=name,
        model=model,
        temperature=temperature,
        instructions=instructions,
    )
    assert chatbot.model == (model if model else settings.AI_MODEL)
    assert chatbot.temperature == (temperature if temperature else DEFAULT_TEMPERATURE)
    assert chatbot.instructions == (
        instructions if instructions else chatbot.instructions
    )
    worker_llm = chatbot.llm
    assert worker_llm.__class__ == LLMClassEnum.openai.value
    assert worker_llm.model_name == (model if model else settings.AI_MODEL)


@pytest.mark.django_db
def test_chatbot_tool(settings, mocker, search_results):
    """The ResourceRecommendationBot tool should be created and function correctly."""
    settings.AI_MIT_SEARCH_LIMIT = 5
    retained_attributes = [
        "title",
        "url",
        "description",
        "offered_by",
        "free",
        "certification",
        "resource_type",
        "resource_type",
    ]
    raw_results = search_results.get("results")
    expected_results = {"results": [], "metadata": {}}
    for resource in raw_results:
        simple_result = {key: resource.get(key) for key in retained_attributes}
        simple_result["instructors"] = resource.get("runs")[-1].get("instructors")
        simple_result["level"] = resource.get("runs")[-1].get("level")
        expected_results["results"].append(simple_result)

    mock_post = mocker.patch(
        "ai_chatbots.tools.requests.get",
        return_value=mocker.Mock(json=mocker.Mock(return_value=search_results)),
    )
    chatbot = ResourceRecommendationBot("anonymous", name="test agent")
    search_parameters = {
        "q": "physics",
        "resource_type": ["course", "program"],
        "free": False,
        "certification": True,
        "offered_by": ["xpro"],
        "limit": 5,
    }
    expected_results["metadata"]["parameters"] = search_parameters.copy()
    tool = chatbot.create_tools()[0]
    results = tool(search_parameters)
    mock_post.assert_called_once_with(
        settings.AI_MIT_SEARCH_URL,
        params={"q": "physics", **search_parameters},
        timeout=30,
    )
    assert_json_equal(json.loads(results), expected_results)


@pytest.mark.django_db
@pytest.mark.parametrize("debug", [True, False])
async def test_get_completion(settings, mocker, debug, search_results):
    """Test that the ResourceRecommendationBot get_completion method returns expected values."""
    settings.AI_DEBUG = debug
    mocker.patch(
        "ai_chatbots.chatbots.CompiledGraph.aget_state_history",
        return_value=MockAsyncIterator(
            [
                ToolMessageFactory.create(content="Here "),
            ]
        ),
    )
    user_msg = "I want to learn physics"
    metadata = {
        "metadata": {
            "search_parameters": {"q": "physics"},
        },
        "search_results": search_results,
    }
    comment_metadata = f"\n\n<!-- {json.dumps(metadata)} -->\n\n".encode()
    expected_return_value = [b"Here ", b"are ", b"some ", b"results"]
    if debug:
        expected_return_value.append(comment_metadata)
    chatbot = ResourceRecommendationBot("anonymous", name="test agent")
    mock_stream = mocker.patch(
        "ai_chatbots.chatbots.CompiledGraph.astream",
        return_value=mocker.Mock(
            __aiter__=mocker.Mock(
                return_value=MockAsyncIterator(
                    [
                        (AIMessageChunkFactory.create(content=val),)
                        for val in expected_return_value
                    ]
                )
            )
        ),
    )
    chatbot.search_parameters = metadata["metadata"]["search_parameters"]
    chatbot.search_results = metadata["search_results"]
    chatbot.search_parameters = {"q": "physics"}
    chatbot.search_results = search_results
    results = ""
    async for chunk in chatbot.get_completion(user_msg, debug=debug):
        results += str(chunk)
    mock_stream.assert_called_once_with(
        {"messages": [{"role": "user", "content": user_msg}]},
        chatbot.config,
        stream_mode="messages",
    )
    if debug:
        assert '<!-- {"metadata"' in results
    assert "".join([value.decode() for value in expected_return_value]) in results
