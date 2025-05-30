"""Tests for AI chatbots."""

import json
import os
from unittest.mock import ANY, AsyncMock
from uuid import uuid4

import pytest
from django.conf import settings
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableBinding
from open_learning_ai_tutor.constants import Intent
from open_learning_ai_tutor.utils import tutor_output_to_json
from openai import BadRequestError

from ai_chatbots.chatbots import (
    ResourceRecommendationBot,
    SyllabusAgentState,
    SyllabusBot,
    TutorBot,
    VideoGPTAgentState,
    VideoGPTBot,
    get_history,
    get_problem_from_edx_block,
    replace_math_tags,
)
from ai_chatbots.checkpointers import AsyncDjangoSaver
from ai_chatbots.conftest import MockAsyncIterator
from ai_chatbots.factories import (
    AIMessageChunkFactory,
    HumanMessageFactory,
    SystemMessageFactory,
    ToolMessageFactory,
)
from ai_chatbots.proxies import LiteLLMProxy
from ai_chatbots.tools import SearchToolSchema
from main.test_utils import assert_json_equal

pytestmark = pytest.mark.django_db


@pytest.fixture(autouse=True)
def mock_settings(settings):
    """Langsmith API should be blank for most tests"""
    os.environ["LANGSMITH_API_KEY"] = ""
    os.environ["LANGSMITH_TRACING"] = ""
    settings.LANGSMITH_API_KEY = ""
    return settings


@pytest.fixture(autouse=True)
def mock_openai_astream(mocker):
    """Mock the CompiledGraph astream function"""
    return mocker.patch(
        "ai_chatbots.chatbots.CompiledGraph.astream",
        return_value="Here are some results",
    )


@pytest.fixture
async def mock_checkpointer(mocker):
    """Mock the checkpointer"""
    return await AsyncDjangoSaver.create_with_session(
        uuid4(), "test message", "test_bot"
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
                    course_id=["mitx1.23"],
                    collection_name=["vector512"],
                )
            ]
        ),
    )


@pytest.fixture
def posthog_settings(settings):
    """Mock the PostHog settings"""
    settings.POSTHOG_PROJECT_API_KEY = "testkey"
    settings.POSTHOG_HOST = "testhost"
    return settings


@pytest.mark.parametrize(
    ("model", "temperature", "instructions", "has_tools"),
    [
        ("gpt-3.5-turbo", 0.1, "Answer this question as best you can", True),
        ("gpt-4o", 0.3, None, False),
        ("gpt-4", None, None, True),
        (None, None, None, False),
    ],
)
async def test_recommendation_bot_initialization_defaults(
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
    assert chatbot.model == (
        model if model else settings.AI_DEFAULT_RECOMMENDATION_MODEL
    )
    assert chatbot.temperature == (
        temperature if temperature else settings.AI_DEFAULT_TEMPERATURE
    )
    assert chatbot.instructions == (
        instructions if instructions else chatbot.instructions
    )
    worker_llm = chatbot.llm
    assert worker_llm.__class__ == RunnableBinding if has_tools else ChatLiteLLM
    assert worker_llm.model == (
        model if model else settings.AI_DEFAULT_RECOMMENDATION_MODEL
    )


async def test_recommendation_bot_tool(settings, mocker, search_results):
    """The ResourceRecommendationBot tool should be created and function correctly."""
    settings.AI_MIT_SEARCH_LIMIT = 5
    settings.AI_MIT_SEARCH_DETAIL_URL = "https://test.mit.edu/resource="
    retained_attributes = [
        "title",
        "id",
        "readable_id",
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
        simple_result["url"] = f"https://test.mit.edu/resource={resource.get('id')}"
        expected_results["results"].append(simple_result)

    mock_post = mocker.patch(
        "ai_chatbots.tools.requests.get",
        return_value=mocker.Mock(json=mocker.Mock(return_value=search_results)),
    )
    chatbot = ResourceRecommendationBot("anonymous")
    search_parameters = {
        "q": "physics",
        "resource_type": ["course", "program"],
        "free": False,
        "certification": True,
        "offered_by": ["xpro"],
        "limit": 5,
        "state": {"search_url": [settings.AI_MIT_SEARCH_URL]},
    }
    tool = chatbot.create_tools()[0]
    results = tool.invoke(search_parameters)
    search_parameters.pop("state")
    expected_results["metadata"]["parameters"] = search_parameters
    expected_results["metadata"]["search_url"] = settings.AI_MIT_SEARCH_URL
    mock_post.assert_called_once_with(
        settings.AI_MIT_SEARCH_URL,
        params={"q": "physics", **search_parameters},
        timeout=30,
    )
    assert_json_equal(json.loads(results), expected_results)


@pytest.mark.parametrize("debug", [True, False])
async def test_get_completion(
    posthog_settings, mocker, mock_checkpointer, debug, search_results
):
    """Test that the ResourceRecommendationBot get_completion method returns expected values."""
    posthog_settings.AI_DEBUG = debug
    mock_posthog = mocker.patch("ai_chatbots.chatbots.posthog", autospec=True)
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
    chatbot = ResourceRecommendationBot("anonymous", mock_checkpointer)
    mock_stream = mocker.patch(
        "ai_chatbots.chatbots.CompiledGraph.astream",
        return_value=mocker.Mock(
            __aiter__=mocker.Mock(
                return_value=MockAsyncIterator(
                    [
                        (
                            AIMessageChunkFactory.create(content=val),
                            {"langgraph_node": "agent"},
                        )
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
    mock_posthog.Posthog.return_value.capture.assert_called_once_with(
        "anonymous",
        event="RECOMMENDATION_JOB",
        properties={
            "question": user_msg,
            "answer": ANY,
            "metadata": "{}",
            "user": "anonymous",
        },
    )


async def test_recommendation_bot_create_agent_graph(mocker, mock_checkpointer):
    """Test that create_agent_graph function creates a graph with expected nodes/edges"""
    chatbot = ResourceRecommendationBot("anonymous", mock_checkpointer, thread_id="foo")
    agent = chatbot.create_agent_graph()
    for node in ("agent", "tools", "pre_model_hook"):
        assert node in agent.nodes
    graph = agent.get_graph()
    tool = graph.nodes["tools"].data.tools_by_name["search_courses"]
    assert tool.args_schema == SearchToolSchema
    assert tool.func.__name__ == "search_courses"
    edges = graph.edges
    assert len(edges) == 5
    summary_edge = edges[1]
    for test_condition in (
        summary_edge.source == "pre_model_hook",
        summary_edge.target == "agent",
        not summary_edge.conditional,
    ):
        assert test_condition
    tool_agent_edge = edges[2]
    for test_condition in (
        tool_agent_edge.source == "tools",
        tool_agent_edge.target == "pre_model_hook",
        not tool_agent_edge.conditional,
    ):
        assert test_condition
    agent_tool_edge = edges[3]
    for test_condition in (
        agent_tool_edge.source == "agent",
        agent_tool_edge.target == "tools",
        agent_tool_edge.conditional,
    ):
        assert test_condition
    agent_end_edge = edges[4]
    for test_condition in (
        agent_end_edge.source == "agent",
        agent_end_edge.target == "__end__",
        agent_end_edge.conditional,
    ):
        assert test_condition


async def test_syllabus_bot_create_agent_graph(mocker, mock_checkpointer):
    """Test that create_agent_graph function calls create_react_agent with expected arguments"""
    mock_create_agent = mocker.patch("ai_chatbots.chatbots.create_react_agent")
    chatbot = SyllabusBot("anonymous", mock_checkpointer, thread_id="foo")
    mock_create_agent.assert_called_once_with(
        chatbot.llm,
        tools=chatbot.tools,
        checkpointer=chatbot.checkpointer,
        state_schema=SyllabusAgentState,
        pre_model_hook=ANY,
        state_modifier=chatbot.instructions,
    )


@pytest.mark.parametrize("default_model", ["gpt-3.5-turbo", "gpt-4", "gpt-4o"])
async def test_syllabus_bot_get_completion_state(
    mock_checkpointer, mock_openai_astream, default_model
):
    """Proper state should get passed along by get_completion"""
    settings.AI_DEFAULT_SYLLABUS_MODEL = default_model
    chatbot = SyllabusBot("anonymous", mock_checkpointer, thread_id="foo")
    extra_state = {
        "course_id": ["mitx1.23"],
        "collection_name": ["vector512"],
    }
    state = SyllabusAgentState(messages=[HumanMessage("hello")], **extra_state)
    async for _ in chatbot.get_completion("hello", extra_state=extra_state):
        mock_openai_astream.assert_called_once_with(
            state,
            chatbot.config,
            stream_mode="messages",
        )
    assert chatbot.llm.model == default_model


async def test_syllabus_bot_tool(
    settings, mocker, mock_checkpointer, syllabus_agent_state, content_chunk_results
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
    chatbot = SyllabusBot("anonymous", mock_checkpointer)

    search_parameters = {
        "q": "main topics",
        "resource_readable_id": syllabus_agent_state["course_id"][-1],
        "collection_name": syllabus_agent_state["collection_name"][-1],
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


async def test_get_tool_metadata(mocker, mock_checkpointer):
    """Test that the get_tool_metadata function returns the expected metadata"""
    chatbot = ResourceRecommendationBot("anonymous", mock_checkpointer)
    mock_tool_content = {
        "metadata": {
            "parameters": {
                "q": "main topics",
                "resource_readable_id": "MITx+6.00.1x",
                "collection_name": "vector512",
            },
            "search_url": "https://test.mit.edu/search2",
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
                        ],
                        "search_url": [
                            "https://test.mit.edu/search0",
                            "https://test.mit.edu/search1",
                        ],
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
                "search_url": mock_tool_content.get("metadata", {}).get(
                    "search_url", []
                ),
                "search_parameters": mock_tool_content.get("metadata", {}).get(
                    "parameters", []
                ),
                "search_results": mock_tool_content.get("results", []),
                "thread_id": chatbot.config["configurable"]["thread_id"],
            }
        }
    )


async def test_get_tool_metadata_none(mocker, mock_checkpointer):
    """Test that the get_tool_metadata function returns an empty dict JSON string"""
    chatbot = SyllabusBot("anonymous", mock_checkpointer)
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


async def test_get_tool_metadata_error(mocker, mock_checkpointer):
    """Test that the get_tool_metadata function returns the expected error response"""
    chatbot = SyllabusBot("anonymous", mock_checkpointer)
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


@pytest.mark.parametrize("use_proxy", [True, False])
async def test_proxy_settings(settings, mocker, mock_checkpointer, use_proxy):
    """Test that the proxy settings are set correctly"""
    mock_create_proxy_user = mocker.patch(
        "ai_chatbots.proxies.LiteLLMProxy.create_proxy_user"
    )
    mock_llm = mocker.patch("ai_chatbots.chatbots.ChatLiteLLM")
    settings.AI_PROXY_CLASS = "LiteLLMProxy" if use_proxy else None
    settings.AI_PROXY_URL = "http://proxy.url"
    settings.AI_PROXY_AUTH_TOKEN = "test"  # noqa: S105
    model_name = "openai/o9-turbo"
    settings.AI_DEFAULT_RECOMMENDATION_MODEL = model_name
    chatbot = ResourceRecommendationBot("user1", mock_checkpointer)
    if use_proxy:
        mock_create_proxy_user.assert_any_call("user1")
        assert chatbot.proxy_prefix == LiteLLMProxy.PROXY_MODEL_PREFIX
        assert isinstance(chatbot.proxy, LiteLLMProxy)
        mock_llm.assert_any_call(
            model=f"{LiteLLMProxy.PROXY_MODEL_PREFIX}{model_name}",
            **chatbot.proxy.get_api_kwargs(),
            **chatbot.proxy.get_additional_kwargs(chatbot),
        )
    else:
        mock_create_proxy_user.assert_not_called()
        assert chatbot.proxy_prefix == ""
        assert chatbot.proxy is None
        mock_llm.assert_any_call(
            model=model_name,
            **{},
            **{},
        )


@pytest.mark.parametrize(
    ("model", "temperature"),
    [
        ("gpt-3.5-turbo", 0.1),
        ("gpt-4", None),
        (None, None),
    ],
)
async def test_tutor_bot_intitiation(mocker, model, temperature):
    """Test the tutor class instantiation."""
    name = "My tutor bot"
    edx_module_id = "block1"
    block_siblings = ["block1", "block2"]
    mocker.patch(
        "ai_chatbots.chatbots.get_problem_from_edx_block",
        return_value=("problem_xml", "problem_set_xml"),
    )
    chatbot = TutorBot(
        "user",
        name=name,
        model=model,
        temperature=temperature,
        edx_module_id=edx_module_id,
        block_siblings=block_siblings,
    )
    assert chatbot.model == (model if model else settings.AI_DEFAULT_TUTOR_MODEL)
    assert chatbot.temperature == (
        temperature if temperature else settings.AI_DEFAULT_TEMPERATURE
    )
    assert chatbot.problem == "problem_xml"
    assert chatbot.problem_set == "problem_set_xml"
    assert chatbot.model == model if model else settings.AI_DEFAULT_TUTOR_MODEL


async def test_tutor_get_completion(posthog_settings, mocker, mock_checkpointer):
    """Test that the tutor bot get_completion method returns expected values."""
    mock_posthog = mocker.patch("ai_chatbots.chatbots.posthog", autospec=True)
    output = (
        [
            HumanMessage(
                content="what should i try first",
                additional_kwargs={},
                response_metadata={},
            ),
            AIMessage(
                content="Let's start by thinking about the problem.",
                additional_kwargs={},
                response_metadata={},
            ),
        ],
        [
            [Intent.P_HYPOTHESIS],
        ],
        [
            HumanMessage(
                content='Student: "what should i try first"',
                additional_kwargs={},
                response_metadata={},
            ),
            AIMessage(
                content='{"justification": "test", "selection": "g"}',
                additional_kwargs={},
                response_metadata={},
            ),
        ],
    )
    mocker.patch(
        "ai_chatbots.chatbots.get_problem_from_edx_block",
        return_value=("problem_xml", "problem_set_xml"),
    )
    mocker.patch("ai_chatbots.chatbots.message_tutor", return_value=output)
    user_msg = "what should i try next?"
    thread_id = "TEST"

    chatbot = TutorBot(
        "anonymous",
        mock_checkpointer,
        edx_module_id="block1",
        block_siblings=["block1", "block2"],
        thread_id=thread_id,
    )

    results = ""
    async for chunk in chatbot.get_completion(user_msg):
        results += str(chunk)
    assert results == "Let's start by thinking about the problem."

    history = await get_history(thread_id)
    assert history.thread_id == thread_id
    metadata = {"edx_module_id": "block1", "tutor_model": chatbot.model}

    assert history.chat_json == tutor_output_to_json(*output, metadata)
    mock_posthog.Posthog.return_value.capture.assert_called_once_with(
        "anonymous",
        event="TUTOR_JOB",
        properties={
            "question": user_msg,
            "answer": results,
            "metadata": json.dumps(
                {
                    "edx_module_id": "block1",
                    "block_siblings": [
                        "block1",
                        "block2",
                    ],
                    "problem": "problem_xml",
                    "problem_set": "problem_set_xml",
                }
            ),
            "user": "anonymous",
        },
    )


async def test_video_gpt_bot_create_agent_graph(mocker, mock_checkpointer):
    """Test that create_agent_graph function calls create_react_agent with expected arguments"""
    mock_create_agent = mocker.patch("ai_chatbots.chatbots.create_react_agent")
    chatbot = VideoGPTBot("anonymous", mock_checkpointer, thread_id="foo")
    mock_create_agent.assert_called_once_with(
        chatbot.llm,
        tools=chatbot.tools,
        checkpointer=chatbot.checkpointer,
        state_schema=VideoGPTAgentState,
        pre_model_hook=ANY,
        state_modifier=chatbot.instructions,
    )


def test_get_problem_from_edx_block(mocker):
    """Test that the get_problem_from_edx_block function returns the expected problem and problem set"""
    edx_module_id = "block1"
    block_siblings = ["block1", "block2"]

    contentfile_api_results = {
        "results": [
            {
                "edx_module_id": "block1",
                "content": "<problem>problem 1</problem>",
            },
            {
                "edx_module_id": "block2",
                "content": "<problem>problem 2</problem>",
            },
        ]
    }

    mocker.patch(
        "ai_chatbots.tools.requests.get",
        return_value=mocker.Mock(
            json=mocker.Mock(return_value=contentfile_api_results)
        ),
    )

    problem, problem_set = get_problem_from_edx_block(edx_module_id, block_siblings)
    assert problem == "<problem>problem 1</problem>"
    assert problem_set == "<problem>problem 1</problem><problem>problem 2</problem>"


@pytest.mark.parametrize("default_model", ["gpt-3.5-turbo", "gpt-4", "gpt-4o"])
async def test_video_gpt_bot_get_completion_state(
    mock_checkpointer, mock_openai_astream, default_model
):
    """Proper state should get passed along by get_completion"""
    settings.AI_DEFAULT_VIDEO_GPT_MODEL = default_model
    chatbot = VideoGPTBot("anonymous", mock_checkpointer, thread_id="foo")
    extra_state = {
        "transcript_asset_id": [
            "asset-v1:xPRO+LASERxE3+R15+type@asset+block@469c03c4-581a-4687-a9ca-7a1c4047832d-en"
        ]
    }
    state = VideoGPTAgentState(
        messages=[HumanMessage("What is this video about?")], **extra_state
    )
    async for _ in chatbot.get_completion(
        "What is this video about?", extra_state=extra_state
    ):
        mock_openai_astream.assert_called_once_with(
            state,
            chatbot.config,
            stream_mode="messages",
        )
    assert chatbot.llm.model == default_model


async def test_video_gpt_bot_tool(
    settings,
    mocker,
    mock_checkpointer,
    video_gpt_agent_state,
    video_transcript_content_chunk_results,
):
    """The VideoGPTBot should call the correct tool"""
    settings.AI_MIT_TRANSCRIPT_SEARCH_LIMIT = 2
    retained_attributes = [
        "chunk_content",
    ]
    raw_results = video_transcript_content_chunk_results.get("results")
    expected_results = {
        "results": [
            {key: resource.get(key) for key in retained_attributes}
            for resource in raw_results
        ],
        "metadata": {},
    }

    mock_post = mocker.patch(
        "ai_chatbots.tools.requests.get",
        return_value=mocker.Mock(
            json=mocker.Mock(return_value=video_transcript_content_chunk_results)
        ),
    )
    chatbot = VideoGPTBot("anonymous", mock_checkpointer)

    search_parameters = {
        "q": "What is this video about?",
        "edx_module_id": video_gpt_agent_state["transcript_asset_id"][-1],
        "limit": 2,
    }
    expected_results["metadata"]["parameters"] = search_parameters
    tool = chatbot.create_tools()[0]
    results = tool.invoke(
        {"q": "What is this video about?", "state": video_gpt_agent_state}
    )
    mock_post.assert_called_once_with(
        settings.AI_MIT_VIDEO_TRANSCRIPT_URL,
        params=search_parameters,
        timeout=30,
    )
    assert_json_equal(json.loads(results), expected_results)


async def test_bad_request(mocker, mock_checkpointer):
    """Test that the bad_request function logs the exception"""
    mock_log = mocker.patch("ai_chatbots.chatbots.log.exception")
    chatbot = VideoGPTBot("anonymous", mock_checkpointer)
    chatbot.agent.astream = mocker.Mock(
        side_effect=BadRequestError(
            response=mocker.Mock(
                json=mocker.Mock(return_value={"error": {"message": "Bad request"}})
            ),
            message="",
            body="",
        )
    )
    async for _ in chatbot.get_completion("hello"):
        chatbot.agent.astream.assert_called_once()
        mock_log.assert_called_once_with("Bad request error")


def test_math_replacements():
    """Test math replacement for tutor bot"""
    input_text = r"""Hello \(E=mc^2\) and \(a^2 + b^2 = c^2\). Also

\[
F = ma
\]

and

\[ PV = NkT \]

Bye now.
"""

    expected_output = r"""Hello $E=mc^2$ and $a^2 + b^2 = c^2$. Also

$$
F = ma
$$

and

$$ PV = NkT $$

Bye now.
"""
    output_text = replace_math_tags(input_text)
    assert output_text == expected_output
