"""Tests for AI chatbots."""

import json
import os
import re
from unittest.mock import ANY, AsyncMock
from uuid import uuid4

import pytest
from asgiref.sync import sync_to_async
from channels.db import database_sync_to_async
from django.conf import settings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableBinding
from langchain_litellm import ChatLiteLLM
from open_learning_ai_tutor.constants import Intent
from open_learning_ai_tutor.utils import (
    filter_out_system_messages,
    tutor_output_to_json,
)
from openai import BadRequestError

from ai_chatbots.chatbots import (
    ResourceRecommendationBot,
    SearchSummaryBot,
    SupportBot,
    SyllabusAgentState,
    SyllabusBot,
    TutorBot,
    VideoGPTAgentState,
    VideoGPTBot,
    get_canvas_problem_set,
    get_problem_from_edx_block,
)
from ai_chatbots.checkpointers import AsyncDjangoSaver
from ai_chatbots.conftest import MockAsyncIterator
from ai_chatbots.consumers import TutorBotHttpConsumer
from ai_chatbots.factories import (
    AIMessageChunkFactory,
    HumanMessageFactory,
    SystemMessageFactory,
    ToolMessageFactory,
)
from ai_chatbots.models import DjangoCheckpoint, TutorBotOutput, UserChatSession
from ai_chatbots.prompts import PROMPT_SEARCH_SUMMARY_QUERY, SYSTEM_PROMPT_MAPPING
from ai_chatbots.proxies import LiteLLMProxy
from ai_chatbots.tools import SearchToolSchema
from main.test_utils import assert_json_equal

pytestmark = pytest.mark.django_db


@pytest.fixture(autouse=True)
def _setup_test_llm_models():
    """Create LLMModel entries for test models."""
    from ai_chatbots.models import LLMModel

    test_models = [
        {
            "litellm_id": "gpt-3.5-turbo",
            "provider": "openai",
            "name": "gpt-3.5-turbo",
            "temperature": 0.1,
        },
        {
            "litellm_id": "gpt-4",
            "provider": "openai",
            "name": "gpt-4",
            "temperature": 0.1,
        },
        {
            "litellm_id": "gpt-4o",
            "provider": "openai",
            "name": "gpt-4o",
            "temperature": 0.2,
        },
        {
            "litellm_id": "gpt-4-turbo",
            "provider": "openai",
            "name": "gpt-4-turbo",
            "temperature": 0.3,
        },
        {"litellm_id": "openai/o9-turbo", "provider": "openai", "name": "o9-turbo"},
    ]
    for model_data in test_models:
        LLMModel.objects.get_or_create(
            litellm_id=model_data["litellm_id"],
            defaults=model_data,
        )


@pytest.fixture(autouse=True)
def mock_settings(settings):
    """Langsmith API should be blank for most tests"""
    os.environ["LANGSMITH_API_KEY"] = ""
    os.environ["LANGSMITH_TRACING"] = ""
    settings.LANGSMITH_API_KEY = ""
    return settings


@pytest.fixture(autouse=True)
def mock_openai_astream(mocker):
    """Mock the CompiledStateGraph astream function"""
    return mocker.patch(
        "ai_chatbots.chatbots.CompiledStateGraph.astream",
        return_value="Here are some results",
    )


@pytest.fixture
def mock_latest_state_history(mocker):
    """Mock the CompiledStateGraph aget_state_history function"""
    return mocker.patch(
        "ai_chatbots.chatbots.CompiledStateGraph.aget_state_history",
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
@pytest.mark.asyncio
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

    chatbot = await sync_to_async(ResourceRecommendationBot)(
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
    # tools bound -> wrapped runnable (RunnableBinding or subclass); else a bare model.
    # Use isinstance rather than exact class identity, which is brittle across
    # langchain-core/langchain-litellm versions.
    assert isinstance(worker_llm, RunnableBinding if has_tools else ChatLiteLLM)
    assert worker_llm.model == (
        model if model else settings.AI_DEFAULT_RECOMMENDATION_MODEL
    )


@pytest.mark.asyncio
async def test_recommendation_bot_tool(
    settings, mock_httpx_async_client, search_results
):
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
    raw_results = search_results.get("results")[: settings.AI_MIT_SEARCH_LIMIT]
    expected_results = {"results": [], "metadata": {}}
    for resource in raw_results:
        simple_result = {key: resource.get(key) for key in retained_attributes}
        simple_result["instructors"] = resource.get("runs")[-1].get("instructors")
        simple_result["level"] = resource.get("runs")[-1].get("level")
        simple_result["url"] = f"https://test.mit.edu/resource={resource.get('id')}"
        expected_results["results"].append(simple_result)

    mock_client_patch = mock_httpx_async_client(
        search_results, patch_path="ai_chatbots.utils.get_async_http_client"
    )
    chatbot = await sync_to_async(ResourceRecommendationBot)("anonymous")
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
    results = await tool.ainvoke(search_parameters)
    search_parameters.pop("state")
    expected_results["metadata"]["parameters"] = search_parameters
    expected_results["metadata"]["search_url"] = settings.AI_MIT_SEARCH_URL
    mock_client_instance = mock_client_patch.return_value
    mock_client_instance.get.assert_called_once_with(
        settings.AI_MIT_SEARCH_URL,
        params={"q": "physics", **search_parameters},
        headers={"Authorization": f"Bearer {settings.AI_PROXY_AUTH_TOKEN}"},
        timeout=30,
    )
    assert_json_equal(json.loads(results), expected_results)


@pytest.mark.parametrize("debug", [True, False])
@pytest.mark.asyncio
async def test_get_completion(
    posthog_settings, mocker, mock_checkpointer, debug, search_results
):
    """Test that the ResourceRecommendationBot get_completion method returns expected values."""
    mocker.patch(
        "ai_chatbots.chatbots.CompiledStateGraph.aget_state_history",
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
    chatbot = await sync_to_async(ResourceRecommendationBot)(
        "anonymous", mock_checkpointer
    )
    mock_stream = mocker.patch(
        "ai_chatbots.chatbots.CompiledStateGraph.astream",
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


@pytest.mark.parametrize("has_history", [True, False])
async def test_search_summary_bot_get_completion(
    mocker, mock_checkpointer, has_history
):
    """
    Only the first message of a search summary thread should be wrapped in the
    summary instructions; follow-ups should be sent as-is.
    """
    mock_parent_completion = mocker.patch(
        "ai_chatbots.chatbots.ResourceRecommendationBot.get_completion",
        return_value=MockAsyncIterator(["summary"]),
    )
    chatbot = await sync_to_async(SearchSummaryBot)("anonymous", mock_checkpointer)
    assert chatbot.instructions == SYSTEM_PROMPT_MAPPING["recommendation"]
    mocker.patch.object(
        chatbot.agent,
        "aget_state",
        return_value=mocker.Mock(
            values={"messages": [HumanMessageFactory.create()] if has_history else []}
        ),
    )
    extra_state = {"search_url": ["https://test.mit.edu/search"]}

    results = [
        chunk
        async for chunk in chatbot.get_completion("physics", extra_state=extra_state)
    ]

    assert results == ["summary"]
    expected_message = (
        "physics"
        if has_history
        else PROMPT_SEARCH_SUMMARY_QUERY.format(query="physics")
    )
    mock_parent_completion.assert_called_once_with(
        expected_message, extra_state=extra_state
    )


@pytest.mark.asyncio
async def test_recommendation_bot_create_agent_graph(mocker, mock_checkpointer):
    """Test that create_agent_graph function creates a graph with expected nodes/edges"""
    chatbot = await sync_to_async(ResourceRecommendationBot)(
        "anonymous", mock_checkpointer, thread_id="12345678-1234-5678-9abc-123456789abc"
    )
    agent = chatbot.create_agent_graph()
    for node in ("agent", "tools", "pre_model_hook"):
        assert node in agent.nodes
    graph = agent.get_graph()
    tool = graph.nodes["tools"].data.tools_by_name["search_courses"]
    assert tool.args_schema == SearchToolSchema
    assert tool.coroutine.__name__ == "search_courses"
    edges = graph.edges
    assert len(edges) == 5
    summary_edge = edges[3]
    for test_condition in (
        summary_edge.source == "pre_model_hook",
        summary_edge.target == "agent",
        not summary_edge.conditional,
    ):
        assert test_condition
    tool_agent_edge = edges[4]
    for test_condition in (
        tool_agent_edge.source == "tools",
        tool_agent_edge.target == "pre_model_hook",
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
    agent_end_edge = edges[1]
    for test_condition in (
        agent_end_edge.source == "agent",
        agent_end_edge.target == "__end__",
        agent_end_edge.conditional,
    ):
        assert test_condition


@pytest.mark.asyncio
async def test_set_callbacks_logs_agent_graph_to_opik(mocker, mock_checkpointer):
    """set_callbacks should pass the agent graph to the Opik tracer for logging."""
    # Disable PostHog so the Opik tracer is the only callback added
    mocker.patch.object(settings, "POSTHOG_API_HOST", None)
    mocker.patch("ai_chatbots.chatbots.is_opik_configured", return_value=True)
    mock_tracer = mocker.patch("ai_chatbots.opik_tracing.CostTrackingOpikTracer")
    chatbot = await sync_to_async(ResourceRecommendationBot)(
        "anonymous", mock_checkpointer, thread_id="12345678-1234-5678-9abc-123456789abc"
    )
    graph = mocker.Mock()
    mocker.patch.object(chatbot.agent, "get_graph", return_value=graph)

    callbacks = await chatbot.set_callbacks()

    chatbot.agent.get_graph.assert_called_once_with(xray=True)
    mock_tracer.assert_called_once()
    assert mock_tracer.call_args.kwargs["graph"] is graph
    assert mock_tracer.return_value in callbacks


@pytest.mark.asyncio
async def test_syllabus_bot_trace_properties(mocker, mock_checkpointer):
    """The syllabus bot should tag traces with the resource it was asked about."""
    mocker.patch.object(settings, "POSTHOG_API_HOST", None)
    mocker.patch("ai_chatbots.chatbots.is_opik_configured", return_value=True)
    mock_tracer = mocker.patch("ai_chatbots.opik_tracing.CostTrackingOpikTracer")
    chatbot = await sync_to_async(SyllabusBot)(
        "anonymous", mock_checkpointer, thread_id="12345678-1234-5678-9abc-123456789abc"
    )
    extra_state = {
        "course_id": ["MITx+6.00.1x"],
        "collection_name": ["content_files"],
        "exclude_canvas": ["True"],
    }

    properties = chatbot.get_trace_properties(extra_state)
    assert properties == {
        "course_id": "MITx+6.00.1x",
        "collection_name": "content_files",
    }

    await chatbot.set_callbacks(properties=properties)
    metadata = mock_tracer.call_args.kwargs["metadata"]
    assert metadata["course_id"] == "MITx+6.00.1x"
    assert metadata["collection_name"] == "content_files"


@pytest.mark.asyncio
async def test_base_bot_trace_properties_default(mock_checkpointer):
    """Bots without TRACE_STATE_KEYS should add no extra trace properties."""
    chatbot = await sync_to_async(ResourceRecommendationBot)(
        "anonymous", mock_checkpointer, thread_id="12345678-1234-5678-9abc-123456789abc"
    )
    assert chatbot.get_trace_properties({"search_url": ["https://example.com"]}) == {}


@pytest.mark.asyncio
async def test_canvas_tutor_bot_trace_properties(mock_checkpointer):
    """The canvas tutor traces the course run and problem set it was given."""
    chatbot = await sync_to_async(TutorBot)(
        "anonymous",
        mock_checkpointer,
        run_readable_id="course-v1:MITxT+14.01x+2T2024",
        problem_set_title="Problem Set 4",
    )

    assert chatbot.get_trace_properties() == {
        "run_readable_id": "course-v1:MITxT+14.01x+2T2024",
        "problem_set_title": "Problem Set 4",
    }


@pytest.mark.asyncio
async def test_edx_tutor_bot_trace_properties(mock_checkpointer):
    """The edx tutor traces its module id plus the run derived from it."""
    chatbot = await sync_to_async(TutorBot)(
        "anonymous",
        mock_checkpointer,
        edx_module_id="block-v1:MITxT+3.012Sx+3T2024+type@problem+block@abc123",
        block_siblings=["block1", "block2"],
    )

    properties = chatbot.get_trace_properties()

    assert properties == {
        "edx_module_id": ("block-v1:MITxT+3.012Sx+3T2024+type@problem+block@abc123"),
        "run_readable_id": "course-v1:MITxT+3.012Sx+3T2024",
    }
    # the edx request has no problem set title, so it is not traced as None
    assert "problem_set_title" not in properties


@pytest.mark.asyncio
async def test_edx_tutor_bot_callback_metadata_has_derived_run(
    mocker, mock_checkpointer
):
    """
    The derived run must reach the Opik/PostHog callbacks too, not just the
    LangSmith wrapper the consumer opens.  TutorBot.get_completion builds its
    callbacks from get_tool_metadata(), bypassing BaseChatbot.get_completion.
    """
    mocker.patch.object(settings, "POSTHOG_API_HOST", None)
    mocker.patch("ai_chatbots.chatbots.is_opik_configured", return_value=True)
    mock_tracer = mocker.patch("ai_chatbots.opik_tracing.CostTrackingOpikTracer")
    mocker.patch(
        "ai_chatbots.chatbots.get_problem_from_edx_block",
        new_callable=AsyncMock,
        return_value=("problem_xml", "problem_set_xml"),
    )
    chatbot = await sync_to_async(TutorBot)(
        "anonymous",
        mock_checkpointer,
        edx_module_id="block-v1:MITxT+3.012Sx+3T2024+type@problem+block@abc123",
        block_siblings=["block1", "block2"],
    )

    await chatbot.set_callbacks(properties=await chatbot.get_tool_metadata())

    metadata = mock_tracer.call_args.kwargs["metadata"]
    assert metadata["run_readable_id"] == "course-v1:MITxT+3.012Sx+3T2024"
    assert metadata["edx_module_id"] == chatbot.edx_module_id


@pytest.mark.asyncio
async def test_edx_tutor_bot_trace_properties_undecipherable_id(mock_checkpointer):
    """A module id with no derivable run should still trace the module id."""
    chatbot = await sync_to_async(TutorBot)(
        "anonymous",
        mock_checkpointer,
        edx_module_id="block1",
        block_siblings=["block1"],
    )

    assert chatbot.get_trace_properties() == {"edx_module_id": "block1"}


@pytest.mark.asyncio
async def test_video_gpt_bot_trace_properties(mock_checkpointer):
    """The video bot traces the asset id plus the run derived from it."""
    chatbot = await sync_to_async(VideoGPTBot)("anonymous", mock_checkpointer)
    asset_id = "asset-v1:xPRO+LASERxE3+R15+type@asset+block@469c03c4-en"

    properties = chatbot.get_trace_properties({"transcript_asset_id": [asset_id]})

    assert properties == {
        "transcript_asset_id": asset_id,
        "run_readable_id": "course-v1:xPRO+LASERxE3+R15",
    }


@pytest.mark.asyncio
async def test_video_gpt_bot_trace_properties_undecipherable_id(mock_checkpointer):
    """An asset id with no derivable run should still trace the asset id."""
    chatbot = await sync_to_async(VideoGPTBot)("anonymous", mock_checkpointer)

    properties = chatbot.get_trace_properties({"transcript_asset_id": ["asset1"]})

    assert properties == {"transcript_asset_id": "asset1"}


@pytest.mark.asyncio
async def test_syllabus_bot_create_agent_graph(mocker, mock_checkpointer):
    """Test that create_agent_graph function calls create_react_agent with expected arguments"""
    mock_create_agent = mocker.patch("ai_chatbots.chatbots.create_react_agent")
    chatbot = await sync_to_async(SyllabusBot)(
        "anonymous", mock_checkpointer, thread_id="12345678-1234-5678-9abc-123456789abc"
    )
    mock_create_agent.assert_called_once_with(
        chatbot.llm,
        tools=chatbot.tools,
        checkpointer=chatbot.checkpointer,
        state_schema=SyllabusAgentState,
        pre_model_hook=ANY,
        prompt=chatbot.instructions,
    )


@pytest.mark.asyncio
async def test_syllabus_bot_related_courses_instructions(mocker, mock_checkpointer):
    """SyllabusBot should append related courses instructions when enabled."""
    mocker.patch("ai_chatbots.chatbots.create_react_agent")
    chatbot = await sync_to_async(SyllabusBot)(
        "anonymous",
        mock_checkpointer,
        thread_id="12345678-1234-5678-9abc-123456789abc",
        enable_related_courses=True,
    )
    assert "search_related_course_content_files" in chatbot.instructions
    assert "BOTH" in chatbot.instructions
    # The content search tools must not be presented as the only tools, or as
    # mandatory for every question, or the support tool never gets called
    assert "two search tools available" not in chatbot.instructions
    assert "for every user question" not in chatbot.instructions
    assert "search_support_articles" in chatbot.instructions


@pytest.mark.asyncio
async def test_syllabus_bot_resource_facts_instructions(mocker, mock_checkpointer):
    """SyllabusBot should put the resource facts in its system prompt."""
    mocker.patch("ai_chatbots.chatbots.create_react_agent")
    mock_facts = mocker.patch(
        "ai_chatbots.chatbots.get_resource_facts",
        return_value="Facts about this resource:\n- Price: $250.00",
    )
    chatbot = await sync_to_async(SyllabusBot)(
        "anonymous",
        mock_checkpointer,
        thread_id="12345678-1234-5678-9abc-123456789abc",
        course_id="course-v1:PRO+AIGE",
        platform="xpro",
    )
    mock_facts.assert_called_once_with("course-v1:PRO+AIGE", "xpro")
    # the facts are added to the prompt, not in place of it
    assert chatbot.instructions == (
        f"{SYSTEM_PROMPT_MAPPING['syllabus'].rstrip()}\n\n"
        "Facts about this resource:\n- Price: $250.00"
    )


@pytest.mark.asyncio
async def test_syllabus_bot_no_resource_facts(mocker, mock_checkpointer):
    """An unknown resource should leave the system prompt alone."""
    mocker.patch("ai_chatbots.chatbots.create_react_agent")
    mocker.patch("ai_chatbots.chatbots.get_resource_facts", return_value="")
    chatbot = await sync_to_async(SyllabusBot)(
        "anonymous",
        mock_checkpointer,
        thread_id="12345678-1234-5678-9abc-123456789abc",
        course_id="course-v1:No+Such",
    )
    assert chatbot.instructions == SYSTEM_PROMPT_MAPPING["syllabus"]


@pytest.mark.asyncio
async def test_syllabus_bot_no_related_courses_instructions(mocker, mock_checkpointer):
    """SyllabusBot should not append related courses instructions when disabled."""
    mocker.patch("ai_chatbots.chatbots.create_react_agent")
    chatbot = await sync_to_async(SyllabusBot)(
        "anonymous",
        mock_checkpointer,
        thread_id="12345678-1234-5678-9abc-123456789abc",
        enable_related_courses=False,
    )
    assert "search_related_course_content_files" not in chatbot.instructions


@pytest.mark.parametrize("enable_related_courses", [True, False])
@pytest.mark.asyncio
async def test_syllabus_bot_tools(mocker, mock_checkpointer, enable_related_courses):
    """SyllabusBot should have the support article search tool."""
    mocker.patch("ai_chatbots.chatbots.create_react_agent")
    chatbot = await sync_to_async(SyllabusBot)(
        "anonymous",
        mock_checkpointer,
        thread_id="12345678-1234-5678-9abc-123456789abc",
        enable_related_courses=enable_related_courses,
    )
    expected_tools = ["search_content_files", "search_support_articles"]
    if enable_related_courses:
        expected_tools.insert(1, "search_related_course_content_files")
    assert [tool.name for tool in chatbot.create_tools()] == expected_tools


@pytest.mark.parametrize("default_model", ["gpt-3.5-turbo", "gpt-4", "gpt-4o"])
@pytest.mark.asyncio
async def test_syllabus_bot_get_completion_state(
    mock_checkpointer, mock_openai_astream, default_model
):
    """Proper state should get passed along by get_completion"""
    settings.AI_DEFAULT_SYLLABUS_MODEL = default_model
    chatbot = await sync_to_async(SyllabusBot)(
        "anonymous", mock_checkpointer, thread_id="12345678-1234-5678-9abc-123456789abc"
    )
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
    # config metadata is how the resource reaches every tracer, including
    # LangSmith, whose tracer is installed globally rather than via callbacks
    assert chatbot.config["metadata"] == {
        "course_id": "mitx1.23",
        "collection_name": "vector512",
    }


@pytest.mark.asyncio
async def test_syllabus_bot_tool(
    settings,
    mock_checkpointer,
    syllabus_agent_state,
    content_chunk_results,
    mock_httpx_async_client,
):
    """The SyllabusBot tool should call the correct tool"""
    settings.AI_MIT_CONTENT_SEARCH_LIMIT = 5
    settings.LEARN_ACCESS_TOKEN = "test_token"  # noqa: S105
    retained_attributes = [
        "run_title",
        "chunk_content",
    ]
    raw_results = content_chunk_results.get("results")
    expected_results = {
        "results": [
            {
                "id": resource.get("url"),
                **{key: resource.get(key) for key in retained_attributes},
            }
            for resource in raw_results
        ],
        "citation_sources": {
            resource.get("url"): {
                "citation_title": resource.get("title")
                or resource.get("content_title"),
                "citation_url": resource.get("url"),
            }
            for resource in raw_results
            if resource.get("url")
        },
        "metadata": {},
    }

    mock_client_patch = mock_httpx_async_client(
        content_chunk_results, patch_path="ai_chatbots.utils.get_async_http_client"
    )
    chatbot = await sync_to_async(SyllabusBot)("anonymous", mock_checkpointer)

    search_parameters = {
        "q": "main topics",
        "resource_readable_id": syllabus_agent_state["course_id"][-1],
        "collection_name": syllabus_agent_state["collection_name"][-1],
        "limit": 5,
    }
    expected_results["metadata"]["parameters"] = search_parameters
    expected_results["metadata"]["search_url"] = settings.AI_MIT_SYLLABUS_URL
    tool = chatbot.create_tools()[0]
    results = await tool.ainvoke({"q": "main topics", "state": syllabus_agent_state})
    mock_client_instance = mock_client_patch.return_value
    mock_client_instance.get.assert_called_once_with(
        settings.AI_MIT_SYLLABUS_URL,
        params=search_parameters,
        headers={"Authorization": f"Bearer {settings.LEARN_ACCESS_TOKEN}"},
        timeout=30,
    )
    assert_json_equal(json.loads(results), expected_results)


@pytest.mark.asyncio
async def test_get_tool_metadata(mocker, mock_checkpointer):
    """Test that the get_tool_metadata function returns the expected metadata"""
    chatbot = await sync_to_async(ResourceRecommendationBot)(
        "anonymous", mock_checkpointer
    )
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
                "id": "fake_id",
                "run_title": "Main topics",
                "chunk_content": "Here are the main topics",
            }
        ],
        "citation_sources": [
            {"id": "fake_id", "citation_url": "http://www.ocw.mit.edu"}
        ],
    }
    mock_state_history = mocker.patch(
        "ai_chatbots.chatbots.CompiledStateGraph.aget_state_history",
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
    assert metadata == {
        "metadata": {
            "search_url": mock_tool_content.get("metadata", {}).get("search_url", []),
            "search_parameters": mock_tool_content.get("metadata", {}).get(
                "parameters", []
            ),
            "search_results": mock_tool_content.get("results", []),
            "citation_sources": mock_tool_content.get("citation_sources", []),
            "thread_id": chatbot.config["configurable"]["thread_id"],
        }
    }


@pytest.mark.asyncio
async def test_get_tool_metadata_none(mocker, mock_checkpointer):
    """Test that the get_tool_metadata function returns an empty dict JSON string"""
    chatbot = await sync_to_async(SyllabusBot)("anonymous", mock_checkpointer)
    mocker.patch(
        "ai_chatbots.chatbots.CompiledStateGraph.aget_state_history",
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
    assert metadata == {}


@pytest.mark.asyncio
async def test_get_tool_metadata_error(mocker, mock_checkpointer):
    """Test that the get_tool_metadata function returns the expected error response"""
    chatbot = await sync_to_async(SyllabusBot)("anonymous", mock_checkpointer)
    mocker.patch(
        "ai_chatbots.chatbots.CompiledStateGraph.aget_state_history",
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

    assert metadata == {
        "error": "Error parsing tool metadata",
        "content": "Could not connect to api",
    }


@pytest.mark.asyncio
async def test_get_metadata_is_comment_safe(mocker, mock_checkpointer):
    """Metadata JSON must not contain '-->', which would end the HTML comment early"""
    chatbot = await sync_to_async(TutorBot)(
        "anonymous",
        mock_checkpointer,
        run_readable_id="course-v1:MITxT+14.01x",
        problem_set_title="Problem Set 4",
    )
    content = "<!-- Image content: scanned page -->\nProblem text --!> here"
    chatbot.problem_set = {"problem_set_files": [{"content": content}]}
    chatbot.problem_data_loaded = True
    mocker.patch.object(
        chatbot, "_get_latest_checkpoint_id", AsyncMock(return_value=123)
    )

    metadata = await chatbot.get_metadata(debug=True)

    # '-->' and '--!>' both terminate HTML comments; no '>' may survive
    assert ">" not in metadata
    parsed = json.loads(metadata)
    assert parsed["problem_set"]["problem_set_files"][0]["content"] == content
    assert parsed["checkpoint_pk"] == 123


@pytest.mark.parametrize("use_proxy", [True, False])
@pytest.mark.asyncio
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
    chatbot = await sync_to_async(ResourceRecommendationBot)("user1", mock_checkpointer)
    if use_proxy:
        mock_create_proxy_user.assert_any_call("user1")
        assert chatbot.proxy_prefix == LiteLLMProxy.PROXY_MODEL_PREFIX
        assert isinstance(chatbot.proxy, LiteLLMProxy)
        mock_llm.assert_any_call(
            model=f"{LiteLLMProxy.PROXY_MODEL_PREFIX}{model_name}",
            streaming=True,
            stream_options={"include_usage": True},
            model_kwargs={},
            **chatbot.proxy.get_api_kwargs(),
            **chatbot.proxy.get_additional_kwargs(chatbot),
        )
    else:
        mock_create_proxy_user.assert_not_called()
        assert chatbot.proxy_prefix == ""
        assert chatbot.proxy is None
        mock_llm.assert_any_call(
            model=model_name,
            streaming=True,
            stream_options={"include_usage": True},
            model_kwargs={},
        )


@pytest.mark.asyncio
async def test_get_llm_with_reasoning_effort(mocker, mock_checkpointer):
    """Test that get_llm passes reasoning_effort in model_kwargs when model_spec has it."""
    from ai_chatbots.models import LLMModel

    # Create a model with reasoning_effort
    await sync_to_async(LLMModel.objects.update_or_create)(
        litellm_id="test-reasoning-model",
        defaults={
            "provider": "openai",
            "name": "test-reasoning",
            "reasoning_effort": "high",
        },
    )
    mocker.patch(
        "ai_chatbots.chatbots.ResourceRecommendationBot.create_tools",
        return_value=[],
    )

    chatbot = await sync_to_async(ResourceRecommendationBot)(
        "user", mock_checkpointer, model="test-reasoning-model"
    )

    assert chatbot.llm.model_kwargs == {"reasoning_effort": "high"}


@pytest.mark.asyncio
async def test_get_llm_without_reasoning_effort(mocker, mock_checkpointer):
    """Test that get_llm passes empty model_kwargs when model_spec has no reasoning_effort."""
    from ai_chatbots.models import LLMModel

    # Create a model without reasoning_effort
    await sync_to_async(LLMModel.objects.update_or_create)(
        litellm_id="test-no-reasoning-model",
        defaults={
            "provider": "openai",
            "name": "test-no-reasoning",
            "reasoning_effort": "",
        },
    )

    mocker.patch(
        "ai_chatbots.chatbots.ResourceRecommendationBot.create_tools",
        return_value=[],
    )

    chatbot = await sync_to_async(ResourceRecommendationBot)(
        "user", mock_checkpointer, model="test-no-reasoning-model"
    )

    assert chatbot.llm.model_kwargs == {}


@pytest.mark.asyncio
async def test_get_llm_mock_response(settings, mocker, mock_checkpointer):
    """AI_MOCK_RESPONSE makes litellm return a canned reply instead of calling a provider."""
    settings.AI_MOCK_RESPONSE = "Thanks, what is a good email to reach you?"
    mocker.patch(
        "ai_chatbots.chatbots.ResourceRecommendationBot.create_tools",
        return_value=[],
    )

    chatbot = await sync_to_async(ResourceRecommendationBot)(
        "user", mock_checkpointer, model="nonexistent-model"
    )

    assert chatbot.llm.model_kwargs["mock_response"] == (
        "Thanks, what is a good email to reach you?"
    )


@pytest.mark.asyncio
async def test_get_llm_no_mock_response_by_default(mocker, mock_checkpointer):
    """Without the setting, deployed envs must still call the real model."""
    mocker.patch(
        "ai_chatbots.chatbots.ResourceRecommendationBot.create_tools",
        return_value=[],
    )

    chatbot = await sync_to_async(ResourceRecommendationBot)(
        "user", mock_checkpointer, model="nonexistent-model"
    )

    assert "mock_response" not in chatbot.llm.model_kwargs


@pytest.mark.asyncio
async def test_get_llm_model_not_found(mocker, mock_checkpointer):
    """Test that get_llm handles missing model_spec gracefully."""
    mocker.patch(
        "ai_chatbots.chatbots.ResourceRecommendationBot.create_tools",
        return_value=[],
    )

    # Use a model that doesn't exist in the database
    chatbot = await sync_to_async(ResourceRecommendationBot)(
        "user", mock_checkpointer, model="nonexistent-model"
    )

    assert chatbot.llm.temperature is None
    assert chatbot.llm.model_kwargs == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("temp_value", [0.25, None])
async def test_get_llm_temperature_supported(mocker, mock_checkpointer, temp_value):
    """Test that temperature is set when model_spec.temperature is not null."""
    from ai_chatbots.models import LLMModel

    await sync_to_async(LLMModel.objects.update_or_create)(
        litellm_id="temp-supported-model",
        defaults={
            "provider": "openai",
            "name": "temp-supported",
            "temperature": temp_value,
        },
    )

    mocker.patch(
        "ai_chatbots.chatbots.ResourceRecommendationBot.create_tools",
        return_value=[],
    )

    chatbot = await sync_to_async(ResourceRecommendationBot)(
        "user", mock_checkpointer, model="temp-supported-model", temperature=temp_value
    )

    assert chatbot.llm.temperature == temp_value


@pytest.mark.asyncio
async def test_get_llm_temperature_not_supported(mocker, mock_checkpointer):
    """Test that temperature is NOT set when model_spec.temperature is None."""
    from ai_chatbots.models import LLMModel

    await sync_to_async(LLMModel.objects.update_or_create)(
        litellm_id="temp-not-supported-model",
        defaults={
            "provider": "openai",
            "name": "temp-not-supported",
        },
    )

    mocker.patch(
        "ai_chatbots.chatbots.ResourceRecommendationBot.create_tools",
        return_value=[],
    )

    chatbot = await sync_to_async(ResourceRecommendationBot)(
        "user", mock_checkpointer, model="temp-not-supported-model", temperature=0.8
    )

    # Temperature should remain None since model_spec.temperature is None
    assert chatbot.llm.temperature is None


@pytest.mark.asyncio
async def test_get_llm_binds_tools(mocker, mock_checkpointer):
    """Test that get_llm binds tools when chatbot has tools."""
    mock_llm_instance = mocker.Mock()
    mock_bound_llm = mocker.Mock()
    mock_llm_instance.bind_tools.return_value = mock_bound_llm
    mocker.patch("ai_chatbots.chatbots.ChatLiteLLM", return_value=mock_llm_instance)

    # ResourceRecommendationBot has tools by default
    chatbot = await sync_to_async(ResourceRecommendationBot)("user", mock_checkpointer)

    # Verify bind_tools was called
    mock_llm_instance.bind_tools.assert_called_once()
    assert chatbot.llm == mock_bound_llm


@pytest.mark.asyncio
async def test_get_llm_no_tools(mocker, mock_checkpointer):
    """Test that get_llm does NOT bind tools when chatbot has no tools."""
    mock_llm_instance = mocker.Mock()
    mocker.patch("ai_chatbots.chatbots.ChatLiteLLM", return_value=mock_llm_instance)
    mocker.patch(
        "ai_chatbots.chatbots.ResourceRecommendationBot.create_tools",
        return_value=[],
    )

    chatbot = await sync_to_async(ResourceRecommendationBot)("user", mock_checkpointer)

    # Verify bind_tools was NOT called
    mock_llm_instance.bind_tools.assert_not_called()
    assert chatbot.llm == mock_llm_instance


@pytest.mark.parametrize(
    ("model", "temperature"),
    [
        ("gpt-3.5-turbo", 0.1),
        ("gpt-4", None),
        (None, None),
    ],
)
@pytest.mark.parametrize("variant", ["edx", "canvas"])
@pytest.mark.asyncio
async def test_tutor_bot_intitiation(mocker, model, temperature, variant):
    """Test the tutor class instantiation."""
    name = "My tutor bot"
    if variant == "edx":
        edx_module_id = "block1"
        block_siblings = ["block1", "block2"]
        problem_set_title = None
        run_readable_id = None
        mocker.patch(
            "ai_chatbots.chatbots.get_problem_from_edx_block",
            new_callable=AsyncMock,
            return_value=("problem_xml", "problem_set_xml"),
        )
    else:
        edx_module_id = None
        block_siblings = None
        problem_set_title = "Problem Set Title"
        run_readable_id = "run_readable_id"
        mocker.patch(
            "ai_chatbots.chatbots.get_canvas_problem_set",
            new_callable=AsyncMock,
            return_value="problem_set",
        )

    chatbot = await sync_to_async(TutorBot)(
        "user",
        name=name,
        model=model,
        temperature=temperature,
        edx_module_id=edx_module_id,
        block_siblings=block_siblings,
        problem_set_title=problem_set_title,
        run_readable_id=run_readable_id,
    )
    assert chatbot.model == (model if model else settings.AI_DEFAULT_TUTOR_MODEL)
    assert chatbot.temperature == (
        temperature if temperature else settings.AI_DEFAULT_TEMPERATURE
    )
    assert chatbot.problem_data_loaded is False
    await chatbot.load_problem_data()
    assert chatbot.problem == ("problem_xml" if variant == "edx" else "")
    assert chatbot.problem_set == (
        "problem_set_xml" if variant == "edx" else "problem_set"
    )
    assert chatbot.problem_data_loaded is True
    assert chatbot.model == model if model else settings.AI_DEFAULT_TUTOR_MODEL


@pytest.mark.parametrize("variant", ["edx", "canvas"])
@pytest.mark.asyncio
async def test_tutor_get_completion(posthog_settings, mocker, variant):
    """Test that the tutor bot get_completion method returns expected values."""
    final_message = [
        "values",
        {
            "messages": [
                SystemMessage(
                    content="problem prompt",
                    additional_kwargs={},
                    response_metadata={},
                ),
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
            ]
        },
    ]
    generator_return_values = [
        [
            "messages",
            [AIMessageChunkFactory.create(content="Let's start by thinking ")],
            {"langgraph_node": "agent"},
        ],
        [
            "messages",
            [AIMessageChunkFactory.create(content="about the problem. ")],
            {"langgraph_node": "agent"},
        ],
        final_message,
    ]

    mock_stream = mocker.Mock(
        __aiter__=mocker.Mock(return_value=MockAsyncIterator(generator_return_values))
    )
    intents = [
        [Intent.P_HYPOTHESIS],
    ]
    assessment_history = [
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
    ]
    output = (
        mock_stream,
        intents,
        assessment_history,
    )

    if variant == "edx":
        mocker.patch(
            "ai_chatbots.chatbots.get_problem_from_edx_block",
            new_callable=AsyncMock,
            return_value=("problem_xml", "problem_set_xml"),
        )
    else:
        mocker.patch(
            "ai_chatbots.chatbots.get_canvas_problem_set",
            new_callable=AsyncMock,
            return_value="problem_set",
        )
    mocker.patch("ai_chatbots.chatbots.message_tutor", return_value=output)
    user_msg = "what should i try next?"
    # Use unique UUID-based thread_id per variant
    if variant == "edx":
        thread_id = "12345678-1234-5678-9abc-123456789abc"
    else:
        thread_id = "87654321-4321-8765-cba9-987654321def"

    if variant == "canvas":
        problem_set_title = "Problem Set Title"
        run_readable_id = "Run Readable ID"
        edx_module_id = None
        block_siblings = None
    else:
        problem_set_title = None
        run_readable_id = None
        edx_module_id = "block1"
        block_siblings = ["block1", "block2"]

    # Create metadata for the expected history
    expected_metadata = {
        "edx_module_id": edx_module_id,
        "tutor_model": settings.AI_DEFAULT_TUTOR_MODEL,
        "problem_set_title": problem_set_title,
        "run_readable_id": run_readable_id,
    }
    new_history = filter_out_system_messages(final_message[1]["messages"])
    expected_chat_json = tutor_output_to_json(
        new_history, intents, assessment_history, expected_metadata
    )

    # Mock the history object that get_history should return
    mock_history = mocker.Mock()
    mock_history.thread_id = thread_id
    mock_history.chat_json = expected_chat_json
    mock_history.edx_module_id = edx_module_id or ""
    mocker.patch.object(TutorBot, "get_latest_history", return_value=mock_history)

    checkpointer = await AsyncDjangoSaver.create_with_session(
        thread_id=thread_id,
        message=final_message[1]["messages"][1].content,
        user=None,
        dj_session_key="anonymous",
        agent=TutorBotHttpConsumer.ROOM_NAME,
        object_id=edx_module_id,
    )

    chatbot = await sync_to_async(TutorBot)(
        "anonymous",
        checkpointer=checkpointer,
        edx_module_id=edx_module_id,
        block_siblings=block_siblings,
        problem_set_title=problem_set_title,
        run_readable_id=run_readable_id,
        thread_id=thread_id,
    )

    results = ""
    async for chunk in chatbot.get_completion(user_msg):
        results += str(chunk)
    assert "Let's start by thinking about the problem. " in results

    checkpoint = await database_sync_to_async(
        lambda: (
            DjangoCheckpoint.objects.select_related("session")
            .filter(thread_id=thread_id)
            .last()
        )
    )()
    history = await database_sync_to_async(
        lambda: TutorBotOutput.objects.filter(thread_id=thread_id).last()
    )()

    metadata = json.loads(re.search(r"<!-- (.*) -->", results, re.DOTALL).group(1))
    assert metadata["thread_id"] == thread_id
    assert metadata["checkpoint_pk"] == checkpoint.pk
    assert history.thread_id == thread_id
    assert history.chat_json == expected_chat_json
    assert history.edx_module_id == (edx_module_id or "")


@pytest.mark.asyncio
async def test_video_gpt_bot_create_agent_graph(mocker, mock_checkpointer):
    """Test that create_agent_graph function calls create_react_agent with expected arguments"""
    mock_create_agent = mocker.patch("ai_chatbots.chatbots.create_react_agent")
    chatbot = await sync_to_async(VideoGPTBot)(
        "anonymous", mock_checkpointer, thread_id="12345678-1234-5678-9abc-123456789abc"
    )
    mock_create_agent.assert_called_once_with(
        chatbot.llm,
        tools=chatbot.tools,
        checkpointer=chatbot.checkpointer,
        state_schema=VideoGPTAgentState,
        pre_model_hook=ANY,
        prompt=chatbot.instructions,
    )


@pytest.mark.asyncio
async def test_get_problem_from_edx_block(mock_httpx_async_client):
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
    mock_httpx_async_client(
        contentfile_api_results, patch_path="ai_chatbots.utils.get_async_http_client"
    )

    problem, problem_set = await get_problem_from_edx_block(
        edx_module_id, block_siblings
    )
    assert problem == "<problem>problem 1</problem>"
    assert problem_set == "<problem>problem 1</problem><problem>problem 2</problem>"


@pytest.mark.asyncio
async def test_get_canvas_problem_set(mock_httpx_async_client):
    """Test that the get_canvas_problem_set function returns the expected problem set and solution"""
    run_readable_id = "a_run_readable_id"
    problem_set_title = "A Problem Set Title"

    problem_api_results = {
        "problem_set_files": [
            {
                "file_name": "test_problem_set",
                "content": "test problem set",
            }
        ],
        "solution_set_files": [
            {
                "file_name": "test_solution",
                "content": "test solution",
            }
        ],
    }
    mock_httpx_async_client(
        problem_api_results, patch_path="ai_chatbots.utils.get_async_http_client"
    )

    problem_set = await get_canvas_problem_set(run_readable_id, problem_set_title)
    assert problem_set == problem_api_results


@pytest.mark.parametrize("default_model", ["gpt-3.5-turbo", "gpt-4", "gpt-4o"])
@pytest.mark.asyncio
async def test_video_gpt_bot_get_completion_state(
    mock_checkpointer, mock_openai_astream, default_model
):
    """Proper state should get passed along by get_completion"""
    settings.AI_DEFAULT_VIDEO_GPT_MODEL = default_model
    chatbot = await sync_to_async(VideoGPTBot)(
        "anonymous", mock_checkpointer, thread_id="12345678-1234-5678-9abc-123456789abc"
    )
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


@pytest.mark.asyncio
async def test_video_gpt_bot_tool(
    settings,
    mock_checkpointer,
    video_gpt_agent_state,
    video_transcript_content_chunk_results,
    mock_httpx_async_client,
):
    """The VideoGPTBot should call the correct tool"""
    settings.AI_MIT_TRANSCRIPT_SEARCH_LIMIT = 2
    settings.LEARN_ACCESS_TOKEN = "test_token"  # noqa: S105
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

    mock_client_patch = mock_httpx_async_client(
        video_transcript_content_chunk_results,
        patch_path="ai_chatbots.utils.get_async_http_client",
    )
    chatbot = await sync_to_async(VideoGPTBot)("anonymous", mock_checkpointer)

    search_parameters = {
        "q": "What is this video about?",
        "edx_module_id": video_gpt_agent_state["transcript_asset_id"][-1],
        "limit": 2,
    }
    expected_results["metadata"]["parameters"] = search_parameters
    expected_results["metadata"]["search_url"] = settings.AI_MIT_VIDEO_TRANSCRIPT_URL
    tool = chatbot.create_tools()[0]
    results = await tool.ainvoke(
        {"q": "What is this video about?", "state": video_gpt_agent_state}
    )
    mock_client_instance = mock_client_patch.return_value
    mock_client_instance.get.assert_called_once_with(
        settings.AI_MIT_VIDEO_TRANSCRIPT_URL,
        params=search_parameters,
        headers={"Authorization": f"Bearer {settings.LEARN_ACCESS_TOKEN}"},
        timeout=30,
    )
    assert_json_equal(json.loads(results), expected_results)


@pytest.mark.asyncio
async def test_bad_request(mocker, mock_checkpointer):
    """Test that the bad_request function logs the exception"""
    mock_log = mocker.patch("ai_chatbots.chatbots.log.exception")
    chatbot = await sync_to_async(VideoGPTBot)("anonymous", mock_checkpointer)
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


@pytest.mark.asyncio
async def test_get_completion_handles_value_error(mocker, mock_checkpointer):
    """Should call validate_and_clean_checkpoint, add system message, and retry on ValueError."""
    chatbot = await sync_to_async(ResourceRecommendationBot)("user", mock_checkpointer)

    call_count = 0
    captured_state = None

    async def mock_send_chunks(state):
        nonlocal call_count, captured_state
        call_count += 1
        if call_count == 1:
            raise ValueError
        captured_state = state
        yield "success"

    mocker.patch.object(chatbot, "send_chunks", side_effect=mock_send_chunks)
    mock_validate = mocker.patch.object(
        chatbot, "validate_and_clean_checkpoint", new_callable=AsyncMock
    )
    mock_log = mocker.patch("ai_chatbots.chatbots.log.exception")

    results = [chunk async for chunk in chatbot.get_completion("hello")]

    mock_validate.assert_called_once()
    mock_log.assert_called_with("Validation error, cleaning checkpoint")
    assert call_count == 2
    # Check that a SystemMessage was added to notify about lost context
    assert len(captured_state["messages"]) == 2
    assert isinstance(captured_state["messages"][1], SystemMessage)
    assert "success" in results


@pytest.mark.asyncio
@pytest.mark.django_db
@pytest.mark.parametrize("is_valid", [True, False])
async def test_validate_and_clean_checkpoint(mocker, mock_checkpointer, is_valid):
    """Should not modify checkpoint when messages are valid."""
    chatbot = await sync_to_async(ResourceRecommendationBot)("user", mock_checkpointer)
    valid_messages = [HumanMessageFactory.create(), AIMessage(content="response")]
    mocker.patch.object(
        chatbot.agent,
        "aget_state",
        return_value=mocker.Mock(values={"messages": valid_messages}),
    )
    mock_truncate = mocker.patch(
        "ai_chatbots.chatbots.save_truncated_checkpoint",
        new_callable=AsyncMock,
    )
    mocker.patch(
        "ai_chatbots.chatbots._validate_chat_history",
        side_effect=(ValueError if not is_valid else None),
    )

    await chatbot.validate_and_clean_checkpoint()
    assert mock_truncate.call_count == (0 if is_valid else 1)


@pytest.mark.asyncio
async def test_support_bot_has_no_tools(mock_checkpointer):
    """
    The v1 support bot takes in a problem; it does not try to answer it, and it
    does not file through the model either. Filing happens in get_completion, so
    a second tool-driven path could only disagree with it.
    """
    chatbot = await sync_to_async(SupportBot)("anonymous", mock_checkpointer)

    assert chatbot.create_tools() == []


async def _support_bot(mocker, mock_checkpointer, prior_messages, state=None):
    """Build a SupportBot whose thread already holds prior_messages, LLM stubbed."""
    thread_id = uuid4().hex
    await UserChatSession.objects.acreate(thread_id=thread_id, agent="SupportBot")
    chatbot = await sync_to_async(SupportBot)(
        "anonymous", mock_checkpointer, thread_id=thread_id
    )
    # Write the thread's starting point through the graph rather than stubbing
    # the read, so what one turn accumulates is what the next turn sees.
    await chatbot.agent.aupdate_state(
        chatbot.config,
        {"messages": list(prior_messages), **(state or {})},
        as_node="agent",
    )
    mocker.patch.object(
        SupportBot, "send_chunks", return_value=MockAsyncIterator(["from the llm"])
    )
    return chatbot


SUPPORT_PROBLEM = "The week 3 video just spins forever."


def _support_state(email=""):
    return {"page_url": ["https://learn.mit.edu/week-3"], "user_email": [email]}


@pytest.mark.asyncio
async def test_support_bot_reviews_before_it_files(mocker, mock_checkpointer):
    """
    Nothing reaches the support team until the learner has seen what is going.
    The review is built in code for the same reason the ticket is: under
    AI_MOCK_RESPONSE the model would paraphrase it into something untrue.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 912}),
    )
    chatbot = await _support_bot(mocker, mock_checkpointer, [])

    reply = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                SUPPORT_PROBLEM, extra_state=_support_state("signedin@mit.edu")
            )
        ]
    )

    mock_file.assert_not_called()
    assert SUPPORT_PROBLEM in reply
    assert "signedin@mit.edu" in reply
    assert "https://learn.mit.edu/week-3" in reply
    SupportBot.send_chunks.assert_not_called()


@pytest.mark.parametrize(
    "confirmation",
    ["send", "Send it.", "yes please", "ok", "no, that's everything"],
)
@pytest.mark.asyncio
async def test_support_bot_files_on_the_turn_after_the_review(
    mocker, mock_checkpointer, confirmation
):
    """
    The turn after the review always files, whatever the learner typed. We have
    no way to carry on a conversation and file later, so anything that does not
    file here strands the request.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={"user_email": ["signedin@mit.edu"], "awaiting_send": [True]},
    )

    reply = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                confirmation, extra_state=_support_state("signedin@mit.edu")
            )
        ]
    )

    mock_file.assert_called_once()
    assert "4821" in reply
    assert mock_file.call_args.kwargs["description"] == SUPPORT_PROBLEM


@pytest.mark.asyncio
async def test_support_bot_files_against_a_corrected_address(mocker, mock_checkpointer):
    """
    A learner who reads the review and spots a typo in their own address has no
    way to fix it except to type it again, so the later one has to win.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM), HumanMessage("learner@exmaple.com")],
        state={"user_email": ["learner@exmaple.com"], "awaiting_send": [True]},
    )

    reply = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                "learner@example.com", extra_state=_support_state()
            )
        ]
    )

    assert mock_file.call_args.kwargs["email"] == "learner@example.com"
    assert "learner@example.com" in reply


@pytest.mark.asyncio
async def test_support_bot_retries_a_failed_file_with_the_corrected_address(
    mocker, mock_checkpointer
):
    """
    A Zendesk failure leaves awaiting_send set so the next turn retries. The
    retry has to read the corrected address, not the one state started with.
    """
    mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"error": "boom"}),
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM), HumanMessage("learner@exmaple.com")],
        state={"user_email": ["learner@exmaple.com"], "awaiting_send": [True]},
    )

    async for _ in chatbot.get_completion(
        "learner@example.com", extra_state=_support_state()
    ):
        pass

    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    async for _ in chatbot.get_completion("send", extra_state=_support_state()):
        pass

    assert mock_file.call_args.kwargs["email"] == "learner@example.com"


@pytest.mark.asyncio
async def test_support_bot_files_under_the_learners_name(mocker, mock_checkpointer):
    """
    The widget sends the name with the opening message, not with the turn that
    files, so it has to come back out of accumulated state like the URL does.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={
            "user_email": ["signedin@mit.edu"],
            "user_name": ["Ada Lovelace"],
            "awaiting_send": [True],
        },
    )

    async for _ in chatbot.get_completion(
        "send", extra_state=_support_state("signedin@mit.edu")
    ):
        pass

    assert mock_file.call_args.kwargs["name"] == "Ada Lovelace"


@pytest.mark.asyncio
async def test_support_bot_files_under_the_learners_username(mocker, mock_checkpointer):
    """
    The username rides alongside user_name the same way: accumulated in graph
    state from the opening message, read back out on the turn that files.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={
            "user_email": ["signedin@mit.edu"],
            "user_name": ["Ada Lovelace"],
            "user_username": ["ada"],
            "awaiting_send": [True],
        },
    )

    async for _ in chatbot.get_completion(
        "send", extra_state=_support_state("signedin@mit.edu")
    ):
        pass

    assert mock_file.call_args.kwargs["username"] == "ada"


@pytest.mark.asyncio
async def test_support_bot_adds_late_detail_to_the_description(
    mocker, mock_checkpointer
):
    """
    "Anything else to add?" is only worth asking if the answer reaches the
    support team.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={"user_email": ["signedin@mit.edu"], "awaiting_send": [True]},
    )

    async for _ in chatbot.get_completion(
        "It happens on my phone too.", extra_state=_support_state("signedin@mit.edu")
    ):
        pass

    description = mock_file.call_args.kwargs["description"]
    assert SUPPORT_PROBLEM in description
    assert "It happens on my phone too." in description


@pytest.mark.asyncio
async def test_support_bot_describes_every_turn_the_learner_typed(
    mocker, mock_checkpointer
):
    """
    Filing only the opening line throws away everything the learner added while
    TIM was asking for their address.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [
            HumanMessage(SUPPORT_PROBLEM),
            AIMessage("What's your email address?"),
            HumanMessage("learner@example.com"),
            AIMessage("Here's what I'll send..."),
        ],
        state={"user_email": ["learner@example.com"], "awaiting_send": [True]},
    )

    async for _ in chatbot.get_completion(
        "Chrome on Windows.", extra_state=_support_state()
    ):
        pass

    description = mock_file.call_args.kwargs["description"]
    assert SUPPORT_PROBLEM in description
    assert "Chrome on Windows." in description
    # The address is the reply-to, not part of the problem.
    assert "learner@example.com" not in description


@pytest.mark.asyncio
async def test_support_bot_can_retry_a_failed_send(mocker, mock_checkpointer):
    """
    A Zendesk outage must not cost the learner the request. Their next message
    tries again rather than starting the review over.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"error": "Could not file the support request."}),
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={"user_email": ["signedin@mit.edu"], "awaiting_send": [True]},
    )
    async for _ in chatbot.get_completion(
        "send", extra_state=_support_state("signedin@mit.edu")
    ):
        pass

    mock_file.return_value = {"reference": 4821}
    reply = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                "try again", extra_state=_support_state("signedin@mit.edu")
            )
        ]
    )

    assert "4821" in reply


@pytest.mark.asyncio
async def test_support_bot_files_ticket_when_learner_gives_email(
    mocker, mock_checkpointer
):
    """
    The whole point of v1: the learner answers TIM's email question, confirms,
    and a ticket exists. The LLM emits no tool calls under AI_MOCK_RESPONSE, so
    leaving the filing to the model means no ticket is ever filed.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    chatbot = await _support_bot(
        mocker, mock_checkpointer, [HumanMessage(SUPPORT_PROBLEM)]
    )
    async for _ in chatbot.get_completion(
        "learner@example.com", extra_state=_support_state()
    ):
        pass

    reply = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                "send", extra_state=_support_state()
            )
        ]
    )

    assert "4821" in reply
    mock_file.assert_called_once()
    assert mock_file.call_args.kwargs["email"] == "learner@example.com"
    assert mock_file.call_args.kwargs["description"] == SUPPORT_PROBLEM
    assert mock_file.call_args.kwargs["page_url"] == "https://learn.mit.edu/week-3"
    SupportBot.send_chunks.assert_not_called()
    session = await UserChatSession.objects.aget(thread_id=chatbot.thread_id)
    assert session.support_ticket_reference == "4821"


@pytest.mark.asyncio
async def test_support_bot_does_not_file_on_the_first_message(
    mocker, mock_checkpointer
):
    """
    A learner whose problem mentions an address ("can't log in as x@y.com") has
    not been asked for a reply-to address yet. Filing there wastes the one ticket
    on a one-line description.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 1}),
    )
    chatbot = await _support_bot(mocker, mock_checkpointer, [])

    reply = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                "I cannot log in as learner@example.com",
                extra_state=_support_state(),
            )
        ]
    )

    mock_file.assert_not_called()
    assert reply.startswith("from the llm")


@pytest.mark.asyncio
async def test_support_bot_confirmation_names_the_email(mocker, mock_checkpointer):
    """
    A learner who never typed the address cannot know which one the team will
    reply to, so the confirmation has to say it.
    """
    mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 912}),
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={"user_email": ["signedin@mit.edu"], "awaiting_send": [True]},
    )

    reply = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                "send", extra_state=_support_state("signedin@mit.edu")
            )
        ]
    )

    assert "912" in reply
    assert "signedin@mit.edu" in reply


@pytest.mark.asyncio
async def test_support_bot_already_filed_reply_names_the_email(
    mocker, mock_checkpointer
):
    """The follow-up reply is the one place a learner can re-check the address."""
    mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 912}),
    )
    chatbot = await _support_bot(mocker, mock_checkpointer, [])
    async for _ in chatbot.get_completion(
        SUPPORT_PROBLEM, extra_state=_support_state("signedin@mit.edu")
    ):
        pass

    reply = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                "any update?", extra_state=_support_state("signedin@mit.edu")
            )
        ]
    )

    assert "912" in reply
    assert "signedin@mit.edu" in reply


@pytest.mark.asyncio
async def test_support_bot_keeps_a_known_email_across_turns(mocker, mock_checkpointer):
    """
    A Zendesk failure leaves the learner on a second turn. The widget sends the
    email with the opening message, so without the accumulated state that retry
    falls back to asking for an address the learner should never be asked for.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 912}),
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={"user_email": ["signedin@mit.edu"], "awaiting_send": [True]},
    )

    async for _ in chatbot.get_completion(
        "any luck?", extra_state={"page_url": [""], "user_email": [""]}
    ):
        pass

    assert mock_file.call_args.kwargs["email"] == "signedin@mit.edu"


@pytest.mark.asyncio
async def test_support_bot_prefers_the_signed_in_email(mocker, mock_checkpointer):
    """
    process_extra_state reads the email from the session precisely so a caller
    cannot name somebody else as the requester. Honouring a typed address over
    the session would hand that back.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 77}),
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={
            "user_email": ["signedin@mit.edu"],
            "verified_email": ["signedin@mit.edu"],
            "awaiting_send": [True],
        },
    )

    async for _ in chatbot.get_completion(
        "use victim@example.com",
        extra_state={
            **_support_state("signedin@mit.edu"),
            "verified_email": ["signedin@mit.edu"],
        },
    ):
        pass

    assert mock_file.call_args.kwargs["email"] == "signedin@mit.edu"
    assert mock_file.call_args.kwargs["email_verified"] is True


@pytest.mark.asyncio
async def test_support_bot_reports_a_zendesk_failure(mocker, mock_checkpointer):
    """Claiming a ticket exists when it does not strands the learner."""
    mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"error": "Could not file the support request."}),
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={"awaiting_send": [True]},
    )

    reply = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                "learner@example.com", extra_state=_support_state()
            )
        ]
    )

    assert "could not" in reply.lower()
    session = await UserChatSession.objects.aget(thread_id=chatbot.thread_id)
    assert session.support_ticket_reference == ""


@pytest.mark.asyncio
async def test_support_bot_files_one_ticket_per_conversation(mocker, mock_checkpointer):
    """A learner who keeps typing must not open a second ticket on the same thread."""
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    chatbot = await _support_bot(
        mocker, mock_checkpointer, [HumanMessage(SUPPORT_PROBLEM)]
    )
    async for _ in chatbot.get_completion(
        "learner@example.com", extra_state=_support_state()
    ):
        pass

    reply = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                "also@example.com", extra_state=_support_state()
            )
        ]
    )

    mock_file.assert_called_once()
    assert "4821" in reply


@pytest.mark.asyncio
async def test_support_bot_keeps_the_page_url_from_the_first_message(
    mocker, mock_checkpointer
):
    """
    The widget sends page_url with the opening message, not with the email that
    triggers filing, so the URL has to come from accumulated graph state. Reading
    only this turn's extra_state files the ticket without the Page: line, losing
    the one piece of context the learner never has to type.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={
            "page_url": ["https://learn.mit.edu/week-3"],
            "awaiting_send": [True],
        },
    )

    async for _ in chatbot.get_completion(
        "learner@example.com",
        extra_state={"page_url": [""], "user_email": [""]},
    ):
        pass

    assert mock_file.call_args.kwargs["page_url"] == "https://learn.mit.edu/week-3"


@pytest.mark.parametrize(
    "message",
    [
        "the error page told me to contact noreply@edx.org for help",
        "I emailed support@mit.edu and admin@mit.edu but nobody replied",
        f"my address is {'a' * 300}@example.com",
    ],
)
@pytest.mark.asyncio
async def test_support_bot_does_not_treat_prose_addresses_as_the_reply_to(
    mocker, mock_checkpointer, message
):
    """
    An address quoted inside a sentence belongs to the problem, not to the
    learner. Filing against it makes Zendesk email a stranger, and the session
    guard then locks that in for good. An over-long address is worse: the ticket
    is created and the save afterwards blows up on a 254-char column.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    chatbot = await _support_bot(
        mocker, mock_checkpointer, [HumanMessage(SUPPORT_PROBLEM)]
    )

    async for _ in chatbot.get_completion(message, extra_state=_support_state()):
        pass

    mock_file.assert_not_called()


@pytest.mark.parametrize(
    "message",
    [
        "learner@example.com",
        "my email is learner@example.com",
        "it's learner@example.com, thanks!",
    ],
)
@pytest.mark.asyncio
async def test_support_bot_accepts_an_address_offered_as_an_answer(
    mocker, mock_checkpointer, message
):
    """The learner answering TIM's email question still has to work."""
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    chatbot = await _support_bot(
        mocker, mock_checkpointer, [HumanMessage(SUPPORT_PROBLEM)]
    )

    async for _ in chatbot.get_completion(message, extra_state=_support_state()):
        pass
    async for _ in chatbot.get_completion("send", extra_state=_support_state()):
        pass

    assert mock_file.call_args.kwargs["email"] == "learner@example.com"


@pytest.mark.asyncio
async def test_support_bot_marks_a_session_email_as_verified(mocker, mock_checkpointer):
    """An address from the authenticated session is the one support can trust."""
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    chatbot = await _support_bot(mocker, mock_checkpointer, [])
    signed_in = {
        "page_url": ["https://learn.mit.edu/week-3"],
        "user_email": ["signedin@mit.edu"],
        "verified_email": ["signedin@mit.edu"],
    }

    async for _ in chatbot.get_completion(SUPPORT_PROBLEM, extra_state=signed_in):
        pass
    async for _ in chatbot.get_completion("send", extra_state=signed_in):
        pass

    assert mock_file.call_args.kwargs["email_verified"] is True


@pytest.mark.parametrize(
    ("extra_state", "message"),
    [
        # An edX MFE learner: the host asserts the address, learn-ai has no
        # session to check it against.
        (
            {
                "page_url": ["https://learn.mit.edu/week-3"],
                "user_email": ["asserted@example.com"],
                "verified_email": [""],
            },
            SUPPORT_PROBLEM,
        ),
        # An anonymous learner typing an address into the chat.
        (
            {"page_url": ["https://learn.mit.edu/week-3"], "verified_email": [""]},
            "learner@example.com",
        ),
    ],
)
@pytest.mark.asyncio
async def test_support_bot_marks_an_asserted_email_as_unverified(
    mocker, mock_checkpointer, extra_state, message
):
    """Anyone can name anyone; the ticket has to say so rather than imply trust."""
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={"awaiting_send": [True]},
    )

    async for _ in chatbot.get_completion(message, extra_state=extra_state):
        pass

    assert mock_file.call_args.kwargs["email_verified"] is False


async def _checkpointed_support_bot(mock_checkpointer, prior_messages=()):
    """Build a SupportBot on a real checkpointer, so state can be read back."""
    thread_id = uuid4().hex
    await UserChatSession.objects.acreate(thread_id=thread_id, agent="SupportBot")
    chatbot = await sync_to_async(SupportBot)(
        "anonymous", mock_checkpointer, thread_id=thread_id
    )
    if prior_messages:
        await chatbot.agent.aupdate_state(
            chatbot.config, {"messages": list(prior_messages)}
        )
    return chatbot


@pytest.mark.asyncio
async def test_support_bot_checkpoints_the_filing_turn(mocker, mock_checkpointer):
    """
    The deterministic path returns before super().get_completion, so without
    this the learner's last message and the reference they were given are not in
    the thread at all - and the trailing metadata comment every other bot ends
    with, which is how the caller learns the thread_id, is never sent.
    """
    mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    chatbot = await _checkpointed_support_bot(
        mock_checkpointer, [HumanMessage(SUPPORT_PROBLEM)]
    )
    async for _ in chatbot.get_completion(
        "learner@example.com", extra_state=_support_state()
    ):
        pass

    reply = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                "send", extra_state=_support_state()
            )
        ]
    )

    assert f'"thread_id": "{chatbot.thread_id}"' in reply
    state = await chatbot.agent.aget_state(chatbot.config)
    contents = [message.content for message in state.values["messages"]]
    assert "learner@example.com" in contents
    assert any("4821" in content for content in contents)


@pytest.mark.asyncio
async def test_support_bot_checkpoints_the_already_filed_turn(
    mocker, mock_checkpointer
):
    """The same holds for the turn that reads an existing reference back."""
    mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    chatbot = await _checkpointed_support_bot(
        mock_checkpointer, [HumanMessage(SUPPORT_PROBLEM)]
    )
    await UserChatSession.objects.filter(thread_id=chatbot.thread_id).aupdate(
        support_ticket_reference="4821", support_ticket_email="learner@example.com"
    )

    reply = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                "any update?", extra_state=_support_state()
            )
        ]
    )

    assert f'"thread_id": "{chatbot.thread_id}"' in reply
    state = await chatbot.agent.aget_state(chatbot.config)
    contents = [message.content for message in state.values["messages"]]
    assert "any update?" in contents
