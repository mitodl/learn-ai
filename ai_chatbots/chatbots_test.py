"""Tests for AI chatbots."""

import json
from unittest.mock import AsyncMock

import pytest
from django.conf import settings
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableBinding

from ai_chatbots.chatbots import (
    DEFAULT_TEMPERATURE,
    ResourceRecommendationBot,
    SyllabusAgentState,
    SyllabusBot,
)
from ai_chatbots.conftest import MockAsyncIterator
from ai_chatbots.constants import LLMClassEnum
from ai_chatbots.factories import (
    AIMessageChunkFactory,
    HumanMessageFactory,
    SystemMessageFactory,
    ToolMessageFactory,
)
from ai_chatbots.tools import SearchToolSchema
from main.test_utils import assert_json_equal


@pytest.fixture(autouse=True)
def mock_openai_astream(mocker):
    """Mock the CompiledGraph astream function"""
    return mocker.patch(
        "ai_chatbots.chatbots.CompiledGraph.astream",
        return_value="Here are some results",
    )


@pytest.fixture
def mock_latest_state_history(mocker):
    """Mock the CompiledGraph aget_state_history function"""
    return mocker.patch(
        "ai_chatbots.chatbots.CompiledGraph.aget_state_history",
        return_value=MockAsyncIterator(
            [
                SyllabusAgentState(
                    messages=[
                        HumanMessageFactory.create(content="Who am I"),
                        ToolMessageFactory.create(),
                        SystemMessageFactory.create(content="You are you"),
                        HumanMessageFactory.create(content="Not a useful answer"),
                    ],
                    course_id="mitx1.23",
                    collection_name="vector512",
                )
            ]
        ),
    )


@pytest.mark.parametrize(
    ("model", "temperature", "instructions", "has_tools"),
    [
        ("gpt-3.5-turbo", 0.1, "Answer this question as best you can", True),
        ("gpt-4o", 0.3, None, False),
        ("gpt-4", None, None, True),
        (None, None, None, False),
    ],
)
def test_recommendation_bot_initialization_defaults(
    mocker, model, temperature, instructions, has_tools
):
    """Test the ResourceRecommendationBot class instantiation."""
    name = "My search bot"

    if not has_tools:
        mocker.patch(
            "ai_chatbots.chatbots.ResourceRecommendationBot.create_tools",
            return_value=[],
        )

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
    assert (
        worker_llm.__class__ == RunnableBinding
        if has_tools
        else LLMClassEnum.openai.value
    )
    assert worker_llm.model_name == (model if model else settings.AI_MODEL)


@pytest.mark.django_db
def test_recommendation_bot_tool(settings, mocker, search_results):
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
    results = tool.invoke(search_parameters)
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
        {"messages": [HumanMessage(user_msg)]},
        chatbot.config,
        stream_mode="messages",
    )
    if debug:
        assert '<!-- {"metadata"' in results
    assert "".join([value.decode() for value in expected_return_value]) in results


def test_recommendation_bot_create_agent_graph_(mocker):
    """Test that create_agent_graph function creates a graph with expected nodes/edges"""
    chatbot = ResourceRecommendationBot("anonymous", name="test agent", thread_id="foo")
    agent = chatbot.create_agent_graph()
    for node in ("agent", "tools"):
        assert node in agent.nodes
    graph = agent.get_graph()
    tool = graph.nodes["tools"].data.tools_by_name["search_courses"]
    assert tool.args_schema == SearchToolSchema
    assert tool.func.__name__ == "search_courses"
    edges = graph.edges
    assert len(edges) == 4
    tool_agent_edge = edges[1]
    for test_condition in (
        tool_agent_edge.source == "tools",
        tool_agent_edge.target == "agent",
        not tool_agent_edge.conditional,
    ):
        assert test_condition
    agent_tool_edge = edges[2]
    for test_condition in (
        agent_tool_edge.source == "agent",
        agent_tool_edge.target == "tools",
        agent_tool_edge.conditional,
    ):
        assert test_condition
    agent_end_edge = edges[3]
    for test_condition in (
        agent_end_edge.source == "agent",
        agent_end_edge.target == "__end__",
        agent_end_edge.conditional,
    ):
        assert test_condition


async def test_syllabus_bot_create_agent_graph_(mocker):
    """Test that create_agent_graph function calls create_react_agent with expected arguments"""
    mock_create_agent = mocker.patch("ai_chatbots.chatbots.create_react_agent")
    chatbot = SyllabusBot("anonymous", name="test agent", thread_id="foo")
    mock_create_agent.assert_called_once_with(
        chatbot.llm,
        tools=chatbot.tools,
        checkpointer=chatbot.memory,
        state_schema=SyllabusAgentState,
        state_modifier=chatbot.instructions,
    )


async def test_syllabus_bot_get_completion_state(mocker, mock_openai_astream):
    """Proper state should get passed along by get_completion"""
    chatbot = SyllabusBot("anonymous", name="test agent", thread_id="foo")
    extra_state = {
        "course_id": "mitx1.23",
        "collection_name": "vector512",
    }
    state = SyllabusAgentState(messages=[HumanMessage("hello")], **extra_state)
    async for _ in chatbot.get_completion("hello", extra_state=extra_state):
        mock_openai_astream.assert_called_once_with(
            state,
            chatbot.config,
            stream_mode="messages",
        )


@pytest.mark.django_db
def test_syllabus_bot_tool(
    settings, mocker, syllabus_agent_state, content_chunk_results
):
    """The SyllabusBot tool should call the correct tool"""
    settings.AI_MIT_CONTENT_SEARCH_LIMIT = 5
    retained_attributes = [
        "run_title",
        "chunk_content",
    ]
    raw_results = content_chunk_results.get("results")
    expected_results = {
        "results": [
            {key: resource.get(key) for key in retained_attributes}
            for resource in raw_results
        ],
        "metadata": {},
    }

    mock_post = mocker.patch(
        "ai_chatbots.tools.requests.get",
        return_value=mocker.Mock(json=mocker.Mock(return_value=content_chunk_results)),
    )
    chatbot = SyllabusBot("anonymous", name="test agent")

    search_parameters = {
        "q": "main topics",
        "resource_readable_id": syllabus_agent_state["course_id"],
        "collection_name": syllabus_agent_state["collection_name"],
        "limit": 5,
    }
    expected_results["metadata"]["parameters"] = search_parameters
    tool = chatbot.create_tools()[0]
    results = tool.invoke({"q": "main topics", "state": syllabus_agent_state})
    mock_post.assert_called_once_with(
        settings.AI_MIT_SYLLABUS_URL,
        params=search_parameters,
        timeout=30,
    )
    assert_json_equal(json.loads(results), expected_results)


async def test_get_tool_metadata(mocker):
    """Test that the get_tool_metadata function returns the expected metadata"""
    chatbot = ResourceRecommendationBot("anonymous", name="test agent")
    mock_tool_content = {
        "metadata": {
            "parameters": {
                "q": "main topics",
                "resource_readable_id": "MITx+6.00.1x",
                "collection_name": "vector512",
            }
        },
        "results": [
            {
                "run_title": "Main topics",
                "chunk_content": "Here are the main topics",
            }
        ],
    }
    mock_state_history = mocker.patch(
        "ai_chatbots.chatbots.CompiledGraph.aget_state_history",
        return_value=MockAsyncIterator(
            [
                AsyncMock(
                    values={
                        "messages": [
                            SystemMessageFactory.create(),
                            ToolMessageFactory.create(),
                            HumanMessageFactory.create(),
                            ToolMessageFactory.create(
                                tool_call="search_contentfiles",
                                tool_args={"q": "main topics"},
                                content=json.dumps(mock_tool_content),
                            ),
                        ]
                    }
                )
            ]
        ),
    )

    metadata = await chatbot.get_tool_metadata()
    mock_state_history.assert_called_once()
    assert metadata == json.dumps(
        {
            "metadata": {
                "search_parameters": mock_tool_content.get("metadata", {}).get(
                    "parameters", []
                ),
                "search_results": mock_tool_content.get("results", []),
                "thread_id": chatbot.config["configurable"]["thread_id"],
            }
        }
    )


async def test_get_tool_metadata_none(mocker):
    """Test that the get_tool_metadata function returns an empty dict JSON string"""
    chatbot = SyllabusBot("anonymous", name="test agent")
    mocker.patch(
        "ai_chatbots.chatbots.CompiledGraph.aget_state_history",
        return_value=MockAsyncIterator(
            [
                AsyncMock(
                    values={
                        "messages": [
                            HumanMessageFactory.create(content="hello"),
                        ]
                    }
                )
            ]
        ),
    )
    metadata = await chatbot.get_tool_metadata()
    assert metadata == "{}"


async def test_get_tool_metadata_error(mocker):
    """Test that the get_tool_metadata function returns the expected error response"""
    chatbot = SyllabusBot("anonymous", name="test agent")
    mocker.patch(
        "ai_chatbots.chatbots.CompiledGraph.aget_state_history",
        return_value=MockAsyncIterator(
            [
                AsyncMock(
                    values={
                        "messages": [
                            ToolMessageFactory.create(
                                tool_call="search_contentfiles",
                                tool_args={"q": "main topics"},
                                content="Could not connect to api",
                            )
                        ]
                    }
                )
            ]
        ),
    )
    metadata = await chatbot.get_tool_metadata()

    assert metadata == json.dumps(
        {"error": "Error parsing tool metadata", "content": "Could not connect to api"}
    )
