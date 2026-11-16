"""Tests for AI chatbots."""

import asyncio
import inspect
import json
import os
import re
import threading
import time
import warnings
from operator import add
from typing import Annotated
from unittest.mock import ANY, AsyncMock
from uuid import uuid4

import ol_llm
import opik
import pytest
from asgiref.sync import sync_to_async
from channels.db import database_sync_to_async
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import OperationalError
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import InjectedState
from open_learning_ai_tutor.constants import Intent
from open_learning_ai_tutor.utils import (
    filter_out_system_messages,
    tutor_output_to_json,
)
from openai import BadRequestError
from pydantic.warnings import PydanticDeprecatedSince20

from ai_chatbots import tools
from ai_chatbots.chatbots import (
    AFFIRMATIVE_PHRASES,
    PAUSE_PHRASES,
    SUPPORT_ANSWER_CLOSED,
    SUPPORT_ASK_EMAIL,
    SUPPORT_ASK_EMAIL_AGAIN,
    SUPPORT_ASK_MORE,
    SUPPORT_ASK_PROBLEM,
    SUPPORT_FILING_LOCK_PREFIX,
    SUPPORT_NOTHING_TO_SEND,
    SUPPORT_OFFER_TICKET,
    SUPPORT_SEND_UNCLEAR,
    SUPPORT_TICKET_BAD_EMAIL,
    SUPPORT_TICKET_CANCELLED,
    SUPPORT_TICKET_FAILED,
    SUPPORT_TICKET_MAYBE_FILED,
    SUPPORT_TICKET_REJECTED,
    SUPPORT_TICKET_REVIEW_HEADER,
    SUPPORT_TICKET_SENDING,
    ResourceRecommendationBot,
    SearchSummaryBot,
    SupportAgentState,
    SupportBot,
    SyllabusAgentState,
    SyllabusBot,
    TutorBot,
    VideoGPTAgentState,
    VideoGPTBot,
    _closes_the_answer,
    _is_cancellation,
    _support_description,
    _support_review,
    _the_answer_in,
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
from ai_chatbots.prompts import (
    PROMPT_ACCOUNT_TOOLS_CORE,
    PROMPT_SEARCH_SUMMARY_QUERY,
    PROMPT_SYLLABUS_ACCOUNT_TOOLS,
    SYSTEM_PROMPT_MAPPING,
)
from ai_chatbots.proxies import LiteLLMProxy
from ai_chatbots.tools import SearchToolSchema
from ai_chatbots.utils import get_django_cache
from ai_chatbots.zendesk import (
    HTTPX_TIMEOUT_PHASES,
    TICKET_BAD_EMAIL,
    TICKET_REJECTED,
    WORST_PATH_REQUESTS,
    file_support_ticket,
    max_filing_seconds,
)
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


# Captured before the autouse fixture below replaces it. ainvoke runs through
# astream, so a test that drives a real graph has to put the real one back.
_REAL_ASTREAM = CompiledStateGraph.astream


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
        "state": {},
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
        return_value=MockAsyncIterator(
            [
                (
                    AIMessageChunkFactory.create(content=val),
                    {"langgraph_node": "agent"},
                )
                for val in expected_return_value
            ]
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
    extra_state = {"search_filters": [{"topic": ["Physics"]}]}

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


async def test_search_summary_bot_closes_the_completion_it_wraps(
    mocker, mock_checkpointer
):
    """Closing the summary early closes the recommendation completion, which
    holds the trace, in the same task.
    """
    closed = []

    async def parent_completion(self, message, **kwargs):
        try:
            yield "summary"
            yield "more"
        finally:
            closed.append(asyncio.current_task())

    mocker.patch.object(ResourceRecommendationBot, "get_completion", parent_completion)
    chatbot = await sync_to_async(SearchSummaryBot)("anonymous", mock_checkpointer)
    mocker.patch.object(
        chatbot.agent, "aget_state", return_value=mocker.Mock(values={"messages": []})
    )

    completion = chatbot.get_completion("physics")
    assert await anext(completion) == "summary"
    await completion.aclose()

    assert closed == [asyncio.current_task()]


async def test_get_completion_closes_the_graph_stream(mocker, mock_checkpointer):
    """Closing a completion early closes the agent graph's stream in the same
    task, while the trace is still open.
    """
    closed = []

    async def astream(*args, **kwargs):
        try:
            yield AIMessageChunkFactory.create(content="Here "), {}
            yield AIMessageChunkFactory.create(content="more"), {}
        finally:
            closed.append(asyncio.current_task())

    chatbot = await sync_to_async(ResourceRecommendationBot)(
        "anonymous", mock_checkpointer
    )
    mocker.patch.object(chatbot.agent, "astream", astream)

    completion = chatbot.get_completion("physics")
    assert await anext(completion) == "Here "
    await completion.aclose()

    assert closed == [asyncio.current_task()]


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


@pytest.fixture
def opik_traces():
    """Trace through ol-llm into local storage instead of sending to Opik."""
    ol_llm.tracing.reset()
    ol_llm.configure_tracing(ol_llm.TracingConfig())
    # finally: record_traces_locally can raise in setup, and tracing left on
    # breaks every later test.
    try:
        with opik.record_traces_locally() as storage:
            yield storage
    finally:
        ol_llm.tracing.reset()
        ol_llm.configure_tracing(ol_llm.TracingConfig(enabled=False))


def opik_spans(span):
    """Return a span and every span under it, depth first."""
    return [span, *(child for sub in span.spans for child in opik_spans(sub))]


@pytest.mark.asyncio
async def test_get_completion_traces_through_ol_llm(
    mocker, mock_checkpointer, mock_openai_astream, opik_traces
):
    """A chat turn is one ol-llm root trace, with the model's span priced inside."""
    mocker.stop(mock_openai_astream)
    mocker.patch.object(settings, "POSTHOG_API_HOST", None)
    mocker.patch(
        "ai_chatbots.chatbots.ResourceRecommendationBot.create_tools",
        return_value=[],
    )
    chatbot = await sync_to_async(ResourceRecommendationBot)(
        "user-1",
        mock_checkpointer,
        model="openai/gpt-4o-mini",
        thread_id=uuid4().hex,
    )
    chatbot.llm = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="Here are some courses",
                usage_metadata={
                    "input_tokens": 12,
                    "output_tokens": 5,
                    "total_tokens": 17,
                },
            )
        ]
    )
    chatbot.agent = chatbot.create_agent_graph()

    with warnings.catch_warnings():
        # Opik's LangChain tracer reads runs with the deprecated Run.dict().
        warnings.simplefilter("ignore", category=PydanticDeprecatedSince20)
        reply = "".join([chunk async for chunk in chatbot.get_completion("courses?")])

    assert reply.startswith("Here are some courses")
    [trace] = opik_traces.trace_trees
    assert trace.name == "recommendation"
    assert trace.project_name == "learn-ai"
    assert trace.thread_id == chatbot.thread_id
    assert sorted(trace.tags) == [
        "app:learn-ai",
        f"env:{settings.OL_LLM_ENVIRONMENT}",
        "use_case:recommendation",
    ]
    assert trace.metadata["user_id"] == "user-1"
    assert trace.metadata["_opik_graph_definition"]["format"] == "mermaid"
    assert trace.input == {"user_prompt": "courses?"}
    assert trace.output == {"output": "Here are some courses"}
    [model_span] = [
        span for root in trace.spans for span in opik_spans(root) if span.type == "llm"
    ]
    assert (model_span.provider, model_span.model) == ("openai", "gpt-4o-mini")
    assert model_span.usage["prompt_tokens"] == 12
    assert model_span.usage["completion_tokens"] == 5


@pytest.mark.parametrize("tracing", [True, False])
@pytest.mark.asyncio
async def test_get_completion_passes_the_trace_callbacks(
    mocker, mock_checkpointer, mock_openai_astream, tracing
):
    """The PostHog handler and ol-llm's tracer both reach the graph.

    The agent graph is drawn for the trace only when ol-llm traces.
    """
    mocker.patch.object(ol_llm.tracing, "enabled", return_value=tracing)
    traced_chat = mocker.patch.object(ol_llm, "traced_chat")
    chat = traced_chat.return_value.__enter__.return_value
    chat.callbacks = [mocker.sentinel.tracer]
    mocker.patch.object(
        ResourceRecommendationBot,
        "set_callbacks",
        return_value=[mocker.sentinel.posthog],
    )
    chatbot = await sync_to_async(ResourceRecommendationBot)(
        "user-1", mock_checkpointer, thread_id="12345678-1234-5678-9abc-123456789abc"
    )
    graph = mocker.Mock()
    mocker.patch.object(chatbot.agent, "get_graph", return_value=graph)

    async for _ in chatbot.get_completion("hello"):
        pass

    traced_chat.assert_called_once_with(
        chatbot.use_case,
        "hello",
        attribution=ol_llm.Attribution(
            user_id="user-1", thread_id="12345678-1234-5678-9abc-123456789abc"
        ),
        graph=graph if tracing else None,
    )
    if tracing:
        chatbot.agent.get_graph.assert_called_once_with(xray=True)
    else:
        chatbot.agent.get_graph.assert_not_called()
    assert chatbot.config["callbacks"] == [
        mocker.sentinel.posthog,
        mocker.sentinel.tracer,
    ]


@pytest.mark.parametrize(
    ("model", "catalog_id"),
    [
        ("openai/gpt-4o-mini", "openai:gpt-4o-mini"),
        (
            "bedrock/us.anthropic.claude-sonnet-5",
            "bedrock:us.anthropic.claude-sonnet-5",
        ),
        ("gpt-4o", "openai:gpt-4o"),
        ("gemini/gemini-3.1-pro-preview", "google:gemini-3.1-pro-preview"),
        ("vertex_ai/gemini-2.5-pro", "google-cloud:gemini-2.5-pro"),
        ("together_ai/meta-llama/Llama-3-8b", "together:meta-llama/Llama-3-8b"),
        # litellm resolves no provider for a name only the proxy knows.
        ("proxy-only-alias", "unresolved:proxy-only-alias"),
    ],
)
@pytest.mark.asyncio
async def test_bot_use_case_names_the_catalog_model(
    mock_checkpointer, model, catalog_id
):
    """Each bot's ol-llm use case names its model as the catalog does."""
    chatbot = await sync_to_async(SyllabusBot)(
        "anonymous", mock_checkpointer, model=model
    )

    assert chatbot.use_case == ol_llm.UseCase(
        app="learn-ai", name="syllabus", profile="request", model=catalog_id
    )


@pytest.mark.asyncio
async def test_bot_with_a_retired_gemini_model_is_refused(mock_checkpointer):
    """The catalog names litellm's gemini/ provider google:, and retires this model."""
    with pytest.raises(ValueError, match="retired"):
        await sync_to_async(SyllabusBot)(
            "anonymous", mock_checkpointer, model="gemini/gemini-3-pro-preview"
        )


@pytest.mark.asyncio
async def test_bot_with_an_unresolvable_model_is_built_uncataloged(mock_checkpointer):
    """A model id litellm resolves to no provider does not fail the bot's build.

    The catalog does not list it, and the completion goes to litellm as before.
    """
    chatbot = await sync_to_async(SyllabusBot)(
        "anonymous", mock_checkpointer, model="proxy-only-alias"
    )

    assert ol_llm.check_chat_model(chatbot.use_case) == ("model_status:uncataloged",)
    assert chatbot.llm.model == "proxy-only-alias"


@pytest.mark.asyncio
async def test_bot_that_never_calls_the_model_skips_the_catalog_check(
    mocker, mock_checkpointer
):
    """SupportBot scripts every reply, so a retired model does not stop intake."""
    check = mocker.patch.object(
        ol_llm, "check_chat_model", side_effect=ValueError("retired")
    )

    await sync_to_async(SupportBot)("anonymous", mock_checkpointer)

    check.assert_not_called()


@pytest.mark.asyncio
async def test_an_answering_bot_with_a_retired_model_still_builds(
    mocker, mock_checkpointer
):
    """A retired model costs the answer, not the scripted intake."""
    check = mocker.patch.object(
        ol_llm, "check_chat_model", side_effect=ValueError("retired")
    )

    chatbot = await sync_to_async(SupportBot)(
        "learner", mock_checkpointer, enable_account_tools=True
    )

    assert chatbot.CALLS_THE_MODEL is True
    check.assert_not_called()


@pytest.mark.asyncio
async def test_syllabus_bot_trace_properties(mock_checkpointer):
    """The syllabus bot should tag traces with the resource it was asked about."""
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


@pytest.mark.asyncio
async def test_base_bot_trace_properties_default(mock_checkpointer):
    """Bots without TRACE_STATE_KEYS should add no extra trace properties."""
    chatbot = await sync_to_async(ResourceRecommendationBot)(
        "anonymous", mock_checkpointer, thread_id="12345678-1234-5678-9abc-123456789abc"
    )
    assert chatbot.get_trace_properties({"search_filters": [{}]}) == {}


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
async def test_edx_tutor_bot_trace_metadata_has_derived_run(mocker, mock_checkpointer):
    """
    The derived run must reach the Opik trace's spans too, not just the
    LangSmith wrapper the consumer opens.  TutorBot.get_completion opens its
    own trace, bypassing BaseChatbot.get_completion.
    """
    traced_chat = mocker.patch.object(ol_llm, "traced_chat")
    mocker.patch(
        "ai_chatbots.chatbots.get_problem_from_edx_block",
        new_callable=AsyncMock,
        return_value=("problem_xml", "problem_set_xml"),
    )
    mocker.patch.object(TutorBot, "get_latest_history", return_value=None)
    mocker.patch("ai_chatbots.chatbots.message_tutor", side_effect=RuntimeError)
    chatbot = await sync_to_async(TutorBot)(
        "anonymous",
        mock_checkpointer,
        edx_module_id="block-v1:MITxT+3.012Sx+3T2024+type@problem+block@abc123",
        block_siblings=["block1", "block2"],
    )

    async for _ in chatbot.get_completion("hello"):
        pass

    metadata = traced_chat.call_args.kwargs["metadata"]
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


@pytest.mark.parametrize("enable_account_tools", [True, False])
@pytest.mark.asyncio
async def test_syllabus_bot_account_tools(
    mocker, mock_checkpointer, enable_account_tools
):
    """SyllabusBot gets the account tools and their instructions only if enabled."""
    mocker.patch("ai_chatbots.chatbots.create_react_agent")
    chatbot = await sync_to_async(SyllabusBot)(
        "anonymous",
        mock_checkpointer,
        thread_id="12345678-1234-5678-9abc-123456789abc",
        enable_account_tools=enable_account_tools,
    )
    account_tools = [
        "get_enrollments",
        "get_course_progress",
        "get_certificate_status",
        "get_upcoming_assignments",
    ]
    tool_names = [tool.name for tool in chatbot.create_tools()]
    assert (tool_names[-4:] == account_tools) is enable_account_tools
    for prompt in (PROMPT_ACCOUNT_TOOLS_CORE, PROMPT_SYLLABUS_ACCOUNT_TOOLS):
        assert (prompt in chatbot.instructions) is enable_account_tools


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
        litellm_id="openai/test-reasoning-model",
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
        "user", mock_checkpointer, model="openai/test-reasoning-model"
    )

    assert chatbot.llm.model_kwargs == {"reasoning_effort": "high"}


@pytest.mark.asyncio
async def test_get_llm_without_reasoning_effort(mocker, mock_checkpointer):
    """Test that get_llm passes empty model_kwargs when model_spec has no reasoning_effort."""
    from ai_chatbots.models import LLMModel

    # Create a model without reasoning_effort
    await sync_to_async(LLMModel.objects.update_or_create)(
        litellm_id="openai/test-no-reasoning-model",
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
        "user", mock_checkpointer, model="openai/test-no-reasoning-model"
    )

    assert chatbot.llm.model_kwargs == {}


@pytest.mark.asyncio
async def test_get_llm_model_not_found(mocker, mock_checkpointer):
    """Test that get_llm handles missing model_spec gracefully."""
    mocker.patch(
        "ai_chatbots.chatbots.ResourceRecommendationBot.create_tools",
        return_value=[],
    )

    # Use a model that doesn't exist in the database
    chatbot = await sync_to_async(ResourceRecommendationBot)(
        "user", mock_checkpointer, model="openai/nonexistent-model"
    )

    assert chatbot.llm.temperature is None
    assert chatbot.llm.model_kwargs == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("temp_value", [0.25, None])
async def test_get_llm_temperature_supported(mocker, mock_checkpointer, temp_value):
    """Test that temperature is set when model_spec.temperature is not null."""
    from ai_chatbots.models import LLMModel

    await sync_to_async(LLMModel.objects.update_or_create)(
        litellm_id="openai/temp-supported-model",
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
        "user",
        mock_checkpointer,
        model="openai/temp-supported-model",
        temperature=temp_value,
    )

    assert chatbot.llm.temperature == temp_value


@pytest.mark.asyncio
async def test_get_llm_temperature_not_supported(mocker, mock_checkpointer):
    """Test that temperature is NOT set when model_spec.temperature is None."""
    from ai_chatbots.models import LLMModel

    await sync_to_async(LLMModel.objects.update_or_create)(
        litellm_id="openai/temp-not-supported-model",
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
        "user",
        mock_checkpointer,
        model="openai/temp-not-supported-model",
        temperature=0.8,
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


@pytest.mark.asyncio
async def test_get_llm_azure(settings, mocker, mock_checkpointer):
    """An azure/ model gets the Azure endpoint, API version, and token provider."""
    from ai_chatbots.models import LLMModel

    settings.AZURE_OPENAI_ENDPOINT = "https://example.openai.azure.com/"
    settings.AZURE_OPENAI_API_VERSION = "v1"
    token_provider = mocker.Mock(return_value="entra-token")
    mocker.patch(
        "ai_chatbots.utils.get_azure_ad_token_provider", return_value=token_provider
    )
    mocker.patch(
        "ai_chatbots.chatbots.ResourceRecommendationBot.create_tools",
        return_value=[],
    )
    await sync_to_async(LLMModel.objects.update_or_create)(
        litellm_id="azure/gpt-5.2",
        defaults={"provider": "azure", "name": "gpt-5.2", "reasoning_effort": "none"},
    )

    chatbot = await sync_to_async(ResourceRecommendationBot)(
        "user", mock_checkpointer, model="azure/gpt-5.2"
    )

    assert chatbot.llm.model == "azure/gpt-5.2"
    assert chatbot.llm.api_base == settings.AZURE_OPENAI_ENDPOINT
    assert chatbot.llm.model_kwargs == {
        "reasoning_effort": "none",
        "api_version": "v1",
        "azure_ad_token_provider": token_provider,
    }


@pytest.mark.asyncio
async def test_get_llm_azure_streams_with_token_provider(
    settings, mocker, mock_checkpointer
):
    """The streaming path hands the token provider and API version to litellm."""
    settings.AZURE_OPENAI_ENDPOINT = "https://example.openai.azure.com/"
    settings.AZURE_OPENAI_API_VERSION = "v1"
    token_provider = mocker.Mock(return_value="entra-token")
    mocker.patch(
        "ai_chatbots.utils.get_azure_ad_token_provider", return_value=token_provider
    )
    mocker.patch(
        "ai_chatbots.chatbots.ResourceRecommendationBot.create_tools",
        return_value=[],
    )
    mock_acompletion = mocker.patch(
        "litellm.acompletion",
        new_callable=AsyncMock,
        return_value=MockAsyncIterator(
            [{"choices": [{"delta": {"role": "assistant", "content": "hi"}}]}]
        ),
    )
    chatbot = await sync_to_async(ResourceRecommendationBot)(
        "user", mock_checkpointer, model="azure/gpt-4o"
    )

    chunks = [chunk async for chunk in chatbot.llm.astream("hello")]

    assert "".join(chunk.content for chunk in chunks) == "hi"
    call_kwargs = mock_acompletion.call_args.kwargs
    assert call_kwargs["model"] == "azure/gpt-4o"
    assert call_kwargs["stream"] is True
    assert call_kwargs["api_base"] == settings.AZURE_OPENAI_ENDPOINT
    assert call_kwargs["api_version"] == "v1"
    assert call_kwargs["azure_ad_token_provider"] is token_provider


@pytest.mark.asyncio
async def test_llm_with_tools_streams_through_litellm(mock_checkpointer):
    """A real litellm call with tools bound returns a reply (litellm is not mocked)."""
    chatbot = await sync_to_async(ResourceRecommendationBot)("user", mock_checkpointer)
    assert isinstance(chatbot.llm, RunnableBinding)

    chunks = [chunk async for chunk in chatbot.llm.astream("hello", mock_response="ok")]

    assert "".join(chunk.content for chunk in chunks) == "ok"


@pytest.mark.asyncio
async def test_get_llm_azure_no_endpoint(settings, mocker, mock_checkpointer):
    """An azure/ model without AZURE_OPENAI_ENDPOINT fails loudly."""
    settings.AZURE_OPENAI_ENDPOINT = ""
    mocker.patch(
        "ai_chatbots.chatbots.ResourceRecommendationBot.create_tools",
        return_value=[],
    )
    with pytest.raises(ImproperlyConfigured, match="AZURE_OPENAI_ENDPOINT"):
        await sync_to_async(ResourceRecommendationBot)(
            "user", mock_checkpointer, model="azure/gpt-4o"
        )


@pytest.mark.parametrize(
    ("use_proxy", "model_name"),
    [(True, "azure/gpt-4o"), (False, "openai/gpt-4o")],
)
@pytest.mark.asyncio
async def test_get_llm_azure_not_applied(
    settings, mocker, mock_checkpointer, use_proxy, model_name
):
    """Azure kwargs are skipped for non-azure models and when a proxy is set."""
    mocker.patch("ai_chatbots.proxies.LiteLLMProxy.create_proxy_user")
    mock_token_provider = mocker.patch("ai_chatbots.utils.get_azure_ad_token_provider")
    mock_llm = mocker.patch("ai_chatbots.chatbots.ChatLiteLLM")
    mocker.patch(
        "ai_chatbots.chatbots.ResourceRecommendationBot.create_tools",
        return_value=[],
    )
    settings.AZURE_OPENAI_ENDPOINT = "https://example.openai.azure.com/"
    settings.AI_PROXY_CLASS = "LiteLLMProxy" if use_proxy else None
    settings.AI_PROXY_URL = "http://proxy.url"
    chatbot = await sync_to_async(ResourceRecommendationBot)(
        "user", mock_checkpointer, model=model_name
    )

    mock_token_provider.assert_not_called()
    call_kwargs = mock_llm.call_args.kwargs
    assert call_kwargs["model"] == f"{chatbot.proxy_prefix}{model_name}"
    assert call_kwargs.get("api_base") != settings.AZURE_OPENAI_ENDPOINT
    assert "api_version" not in call_kwargs["model_kwargs"]
    assert "azure_ad_token_provider" not in call_kwargs["model_kwargs"]


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

    mock_stream = MockAsyncIterator(generator_return_values)
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


@pytest.mark.parametrize("enable_account_tools", [False, True])
@pytest.mark.asyncio
async def test_support_bot_answers_only_with_the_account_tools(
    mock_checkpointer, enable_account_tools
):
    """Matt's account tools, as they are, and only those."""
    chatbot = await sync_to_async(SupportBot)(
        "learner", mock_checkpointer, enable_account_tools=enable_account_tools
    )

    assert chatbot.create_tools() == (
        list(tools.ACCOUNT_TOOLS) if enable_account_tools else []
    )
    assert (chatbot._answering_agent is not None) is enable_account_tools  # noqa: SLF001


@pytest.mark.asyncio
async def test_support_bot_loads_its_prompt_only_when_answering(
    mocker, mock_checkpointer
):
    mock_get_prompt = mocker.patch(
        "ai_chatbots.chatbots.get_system_prompt", side_effect=lambda name, *_: name
    )

    scripted = await sync_to_async(SupportBot)("anonymous", mock_checkpointer)
    assert scripted.instructions is None
    mock_get_prompt.assert_not_called()

    answering = await sync_to_async(SupportBot)(
        "learner", mock_checkpointer, enable_account_tools=True
    )

    assert [call[0][0] for call in mock_get_prompt.call_args_list] == [
        "support",
        "account_tools_core",
        "support_account_tools",
    ]
    assert (
        answering.instructions
        == "support\n\naccount_tools_core\n\nsupport_account_tools"
    )


@pytest.mark.parametrize("answering", [False, True])
@pytest.mark.asyncio
async def test_support_bot_provisions_a_budget_only_when_it_answers(
    settings, mocker, answering, mock_checkpointer
):
    """create_proxy_user is blocking HTTP, and a scripted bot spends no tokens."""
    _with_a_proxy(settings)
    proxy = mocker.patch("ai_chatbots.proxies.LiteLLMProxy.create_proxy_user")
    thread_id = uuid4().hex
    await UserChatSession.objects.acreate(thread_id=thread_id, agent="SupportBot")
    chatbot = await sync_to_async(SupportBot)(
        "learner",
        mock_checkpointer,
        thread_id=thread_id,
        enable_account_tools=answering,
    )
    assert proxy.call_count == 0
    if answering:
        mocker.patch.object(
            chatbot._answering_agent,  # noqa: SLF001
            "ainvoke",
            AsyncMock(return_value=_answer_result(_tool_message(), AIMessage("ok"))),
        )

    await _say(chatbot, "my grade?")

    assert proxy.called is answering
    # Either way get_llm needs the proxy's prefix and kwargs.
    assert isinstance(chatbot.proxy, LiteLLMProxy)
    assert chatbot.proxy_prefix == LiteLLMProxy.PROXY_MODEL_PREFIX


@pytest.mark.parametrize(
    ("message", "closes"),
    [
        ("thanks", True),
        ("thank you", True),
        ("that helped", True),
        ("got it", True),
        ("no thanks", True),
        ("no", True),
        ("nothing else", True),
        ("that's all", True),
        ("no that's it", True),
        ("that is all", True),
        ("that is everything", True),
        ("that is it", True),
        # Phone keyboards type curly apostrophes.
        ("that\u2019s all", True),
        ("that\u2019s helpful", True),
        ("no that\u2019s everything", True),
        # Curly quotes around a word still come off.
        ("\u2018thanks\u2019", True),
        # Said to the offer, affirmatives mean pass it on.
        ("send it", False),
        ("file it", False),
        ("submit", False),
        ("yes please", False),
        ("go ahead", False),
        ("ok", False),
        ("wait", False),
        ("hold on", False),
        ("my certificate still isn't showing", False),
        ("what about 18.02?", False),
        ("", False),
        # Padded with words from the phrases, or a few fillers.
        ("ok thanks!", True),
        ("thanks so much", True),
        ("thats it", True),
        ("ok, got it", True),
        ("thanks for the info", True),
        ("no that's all, thanks", True),
        # One word outside the set makes it a follow-up.
        ("thanks, but that's not right", False),
        ("that's not right, I submitted all my assignments last week", False),
        ("ok that's it", True),
        # A plain no or a "nothing" only counts as the whole reply.
        ("no help", False),
        ("nothing helped", False),
        ("nothing helps", False),
        ("got nothing", False),
        ("that helped nothing", False),
        ("is it done", False),
        ("is that all?", False),
        ("is that all", False),
    ],
)
def test_closes_the_answer(message, closes):
    assert _closes_the_answer(message) is closes


def test_close_out_phrases_exclude_every_affirmative():
    """Said to the offer, an affirmative means pass it on."""
    assert not [phrase for phrase in AFFIRMATIVE_PHRASES if _closes_the_answer(phrase)]
    assert not [phrase for phrase in PAUSE_PHRASES if _closes_the_answer(phrase)]


def test_support_state_tracks_hand_off_and_answers():
    channels = SupportAgentState.__annotations__

    assert channels["handed_off"] == Annotated[list[bool], add]
    assert channels["answered"] == Annotated[list[bool], add]


def _answer_result(*messages):
    """Shape an ainvoke return the way create_react_agent does."""
    return {"messages": list(messages)}


def _tool_message(content='{"enrollments": []}'):
    return ToolMessage(content=content, tool_call_id="call-1", name="get_enrollments")


class _FakeToolCallingModel(GenericFakeChatModel):
    """A fake model the react agent will accept: bind_tools is a no-op."""

    def bind_tools(self, *args, **kwargs):  # noqa: ARG002
        return self


async def _answering_bot(mocker, mock_checkpointer, result=None):
    """
    Build a SupportBot with answering on, whose answering graph returns result:
    by default a lookup followed by "ok". Not a fixture, since the proxy tests
    change settings before the bot is built.
    """
    if result is None:
        result = _answer_result(_tool_message(), AIMessage("ok"))
    chatbot = await sync_to_async(SupportBot)(
        "learner", mock_checkpointer, enable_account_tools=True
    )
    mocker.patch.object(
        chatbot._answering_agent,  # noqa: SLF001
        "ainvoke",
        AsyncMock(side_effect=result)
        if isinstance(result, Exception)
        else AsyncMock(return_value=result),
    )
    return chatbot


def _with_a_proxy(settings):
    """Without a proxy class, proxy assertions would pass vacuously."""
    settings.AI_PROXY_CLASS = "LiteLLMProxy"
    settings.AI_PROXY_URL = "http://proxy.url"
    settings.AI_PROXY_AUTH_TOKEN = "test"  # noqa: S105


@pytest.mark.asyncio
async def test_a_dead_proxy_hands_the_turn_to_intake(
    settings, mocker, mock_checkpointer
):
    """The real create_proxy_user logs its own errors; this covers one that raises."""
    _with_a_proxy(settings)
    mocker.patch(
        "ai_chatbots.proxies.LiteLLMProxy.create_proxy_user",
        side_effect=RuntimeError("proxy is down"),
    )
    chatbot = await _answering_bot(mocker, mock_checkpointer)

    assert await chatbot._try_to_answer("my grade?", {}, {}) == ""  # noqa: SLF001


@pytest.mark.asyncio
async def test_a_slow_proxy_does_not_hold_the_learner(
    settings, mocker, mock_checkpointer
):
    """Provisioning runs inside the answer timeout."""
    _with_a_proxy(settings)
    settings.AI_SUPPORT_ANSWER_TIMEOUT = 0.05
    mocker.patch(
        "ai_chatbots.proxies.LiteLLMProxy.create_proxy_user",
        side_effect=lambda *args, **kwargs: time.sleep(0.5),  # noqa: ARG005
    )
    chatbot = await _answering_bot(mocker, mock_checkpointer)

    started = time.monotonic()
    answer = await chatbot._try_to_answer("my grade?", {}, {})  # noqa: SLF001

    assert answer == ""
    assert time.monotonic() - started < 0.4
    chatbot._answering_agent.ainvoke.assert_not_called()  # noqa: SLF001


@pytest.mark.asyncio
async def test_provisioning_stays_off_the_shared_executor(
    settings, mocker, mock_checkpointer
):
    _with_a_proxy(settings)
    threads = {}
    mocker.patch(
        "ai_chatbots.proxies.LiteLLMProxy.create_proxy_user",
        side_effect=lambda *args, **kwargs: threads.__setitem__(  # noqa: ARG005
            "proxy", threading.current_thread()
        ),
    )
    chatbot = await _answering_bot(mocker, mock_checkpointer)

    await chatbot._try_to_answer("my grade?", {}, {})  # noqa: SLF001
    await sync_to_async(
        lambda: threads.__setitem__("shared", threading.current_thread())
    )()

    assert threads["proxy"] is not threads["shared"]


LOOKALIKE_HOST_PAGE = (
    "https://courses.learn.test.evil.example"
    "/learn/course/course-v1:MITx+18.01x+2T2025/home"
)


@pytest.mark.parametrize(
    ("base_url", "page_url", "expected"),
    [
        (
            "https://courses.learn.test/",
            "https://courses.learn.test/learn/course/course-v1:MITx+18.01x+2T2025/home",
            "course-v1:MITx+18.01x+2T2025",
        ),
        (
            "https://courses.learn.test/",
            "https://evil.example/course-v1:MITx+18.01x+2T2025",
            None,
        ),
        ("https://courses.learn.test/", LOOKALIKE_HOST_PAGE, None),
        ("https://courses.learn.test", LOOKALIKE_HOST_PAGE, None),
        ("https://courses.learn.test/", "https://learn.mit.edu/dashboard", None),
        ("https://courses.learn.test/", "", None),
    ],
)
@pytest.mark.asyncio
async def test_the_answer_attempt_gets_this_turns_page_course(  # noqa: PLR0913
    settings, mocker, mock_checkpointer, base_url, page_url, expected
):
    """prior_state still holds last turn's page, which must not leak through."""
    settings.LEARNER_API_OPENEDX_BASE_URL = base_url
    chatbot = await _answering_bot(mocker, mock_checkpointer)

    await chatbot._try_to_answer(  # noqa: SLF001
        "what's my grade?",
        {"page_url": ["https://courses.learn.test/course-v1:MITx+6.00x+1T2025"]},
        {"page_url": [page_url]},
    )

    state = chatbot._answering_agent.ainvoke.call_args[0][0]  # noqa: SLF001
    assert state["course_id"] == [expected]
    assert state["platform"] == ["mitxonline" if expected else None]


@pytest.mark.asyncio
async def test_the_answer_attempt_is_traced(settings, mocker, mock_checkpointer):
    settings.LEARNER_API_OPENEDX_BASE_URL = "https://courses.learn.test/"
    chatbot = await _answering_bot(mocker, mock_checkpointer)
    handler = object()
    callbacks = mocker.patch.object(
        SupportBot, "set_callbacks", AsyncMock(return_value=[handler])
    )
    page = "https://courses.learn.test/course-v1:MITx+18.01x+2T2025"

    await chatbot._try_to_answer("my grade?", {}, {"page_url": [page]})  # noqa: SLF001

    config = chatbot._answering_agent.ainvoke.call_args.kwargs["config"]  # noqa: SLF001
    # In, not ==: traced_chat may add Opik's tracer.
    assert handler in config["callbacks"]
    assert callbacks.await_args[0][0] == {
        "page_url": page,
        "course_id": "course-v1:MITx+18.01x+2T2025",
        "platform": "mitxonline",
    }
    assert "callbacks" not in chatbot.config


@pytest.mark.parametrize("tracing", [True, False])
@pytest.mark.asyncio
async def test_the_answer_attempt_runs_inside_an_ol_llm_trace(
    settings, mocker, mock_checkpointer, tracing
):
    """set_callbacks only carries PostHog; the Opik span comes from traced_chat."""
    settings.LEARNER_API_OPENEDX_BASE_URL = "https://courses.learn.test/"
    mocker.patch.object(ol_llm.tracing, "enabled", return_value=tracing)
    traced_chat = mocker.patch.object(ol_llm, "traced_chat")
    chat = traced_chat.return_value.__enter__.return_value
    chat.callbacks = [mocker.sentinel.tracer]
    mocker.patch.object(
        SupportBot, "set_callbacks", AsyncMock(return_value=[mocker.sentinel.posthog])
    )
    chatbot = await _answering_bot(
        mocker,
        mock_checkpointer,
        _answer_result(_tool_message(), AIMessage("You're enrolled in 18.01x.")),
    )
    graph = mocker.Mock()
    mocker.patch.object(chatbot._answering_agent, "get_graph", return_value=graph)  # noqa: SLF001
    page = "https://courses.learn.test/course-v1:MITx+18.01x+2T2025"

    answer = await chatbot._try_to_answer(  # noqa: SLF001
        "what am I enrolled in?", {}, {"page_url": [page]}
    )

    traced_chat.assert_called_once_with(
        chatbot.use_case,
        "what am I enrolled in?",
        attribution=chatbot.attribution,
        metadata={
            "page_url": page,
            "course_id": "course-v1:MITx+18.01x+2T2025",
            "platform": "mitxonline",
        },
        graph=graph if tracing else None,
    )
    config = chatbot._answering_agent.ainvoke.call_args.kwargs["config"]  # noqa: SLF001
    assert config["callbacks"] == [mocker.sentinel.posthog, mocker.sentinel.tracer]
    assert chatbot._answering_agent.get_graph.called is tracing  # noqa: SLF001
    chat.record.assert_called_once_with(answer)


@pytest.mark.asyncio
async def test_a_failed_answer_attempt_still_closes_its_trace(
    mocker, mock_checkpointer
):
    traced_chat = mocker.patch.object(ol_llm, "traced_chat")
    chat = traced_chat.return_value.__enter__.return_value
    chat.callbacks = []
    chatbot = await _answering_bot(
        mocker, mock_checkpointer, RuntimeError("the model is down")
    )

    assert await chatbot._try_to_answer("what am I enrolled in?", {}) == ""  # noqa: SLF001

    traced_chat.return_value.__exit__.assert_called_once()
    chat.record.assert_not_called()


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        pytest.param(
            [
                HumanMessage("what am I enrolled in?"),
                _tool_message(),
                AIMessage("You're enrolled in 18.01x."),
            ],
            "You're enrolled in 18.01x.",
            id="a tool reached the records",
        ),
        pytest.param(
            [
                HumanMessage("why was I charged twice?"),
                AIMessage("I'm sorry to hear that."),
            ],
            "",
            id="no tool fired",
        ),
        pytest.param(
            [
                HumanMessage("what's my grade?"),
                _tool_message(
                    '{"enrollments": [], "note": "There is no course on this page."}'
                ),
                AIMessage("Which course did you mean?"),
            ],
            "Which course did you mean?",
            id="a lookup that found nothing still reached the records",
        ),
        pytest.param(
            [
                HumanMessage("what's my grade?"),
                _tool_message(json.dumps({"error": tools.COULD_NOT_CHECK})),
                AIMessage("I couldn't check your records right now."),
            ],
            "",
            id="every tool failed",
        ),
        pytest.param(
            [
                # Plain text, not JSON, so only the status catches it.
                ToolMessage(
                    content="Error: 1 validation error for get_certificate_status\n"
                    " Please fix your mistakes.",
                    tool_call_id="call-1",
                    name="get_certificate_status",
                    status="error",
                ),
                AIMessage("You haven't earned a certificate in that course."),
            ],
            "",
            id="langgraph's error ToolMessage",
        ),
        pytest.param(
            [
                HumanMessage("what's my grade?"),
                _tool_message(json.dumps({"error": tools.COULD_NOT_CHECK})),
                _tool_message('{"enrollments": [{"course_id": "18.01x"}]}'),
                AIMessage("You're in 18.01x; I couldn't read your grade."),
            ],
            "You're in 18.01x; I couldn't read your grade.",
            id="only some tools failed",
        ),
        pytest.param([_tool_message(), AIMessage("")], "", id="empty content"),
        pytest.param(
            [_tool_message(), AIMessage([{"type": "text", "text": "hi"}])],
            "",
            id="content blocks",
        ),
        pytest.param(
            [
                # The result includes history, which must not stand in for an
                # unusable reply.
                HumanMessage("did I pass 18.01x?"),
                AIMessage("You scored 94% in 18.01x."),
                HumanMessage("when is my next assignment due?"),
                _tool_message(),
                AIMessage([{"type": "text", "text": "due 3 May"}]),
            ],
            "",
            id="never an earlier turn's reply",
        ),
    ],
)
def test_the_answer_in(messages, expected):
    assert _the_answer_in(_answer_result(*messages)) == expected


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(TimeoutError(), id="timeout"),
        pytest.param(RuntimeError("proxy down"), id="raises"),
    ],
)
@pytest.mark.asyncio
async def test_try_to_answer_falls_through_on_failure(
    mocker, mock_checkpointer, failure
):
    chatbot = await _answering_bot(mocker, mock_checkpointer, failure)

    assert await chatbot._try_to_answer("what's my grade?", {}) == ""  # noqa: SLF001


@pytest.mark.asyncio
async def test_try_to_answer_gives_up_on_a_slow_model(
    settings, mocker, mock_checkpointer
):
    settings.AI_SUPPORT_ANSWER_TIMEOUT = 0.01
    chatbot = await sync_to_async(SupportBot)(
        "learner", mock_checkpointer, enable_account_tools=True
    )

    # Short enough that a missing timeout fails instead of hanging the suite.
    async def never_returns(*args, **kwargs):
        await asyncio.sleep(5)

    mocker.patch.object(chatbot._answering_agent, "ainvoke", never_returns)  # noqa: SLF001
    # A spy, not a mock: an unawaited coroutine warns in a later test.
    spy_wait_for = mocker.spy(asyncio, "wait_for")

    assert await chatbot._try_to_answer("what's my grade?", {}) == ""  # noqa: SLF001
    assert spy_wait_for.call_args.kwargs["timeout"] == 0.01


@pytest.mark.asyncio
async def test_the_answering_graph_is_separate_and_unpersisted(mock_checkpointer):
    chatbot = await sync_to_async(SupportBot)(
        "learner", mock_checkpointer, enable_account_tools=True
    )

    assert chatbot._answering_agent is not chatbot.agent  # noqa: SLF001
    assert chatbot._answering_agent.checkpointer is None  # noqa: SLF001
    assert chatbot.agent.checkpointer is mock_checkpointer


@pytest.mark.asyncio
async def test_a_failed_answer_attempt_logs_no_exception_detail(
    mocker, mock_checkpointer
):
    """Frame locals hold the learner's cookies and records."""
    mock_log = mocker.patch("ai_chatbots.chatbots.log.warning")
    chatbot = await _answering_bot(
        mocker, mock_checkpointer, RuntimeError("cookie=secret")
    )

    await chatbot._try_to_answer("what's my grade?", {})  # noqa: SLF001

    assert mock_log.call_args.kwargs.get("exc_info") is None
    assert "cookie=secret" not in str(mock_log.call_args)
    assert "RuntimeError" in str(mock_log.call_args)


@pytest.mark.asyncio
async def test_try_to_answer_sends_a_bounded_slice_of_the_thread(
    mocker, mock_checkpointer
):
    chatbot = await _answering_bot(mocker, mock_checkpointer)
    history = [HumanMessage(f"message {index}") for index in range(30)]

    await chatbot._try_to_answer("and now?", {"messages": history})  # noqa: SLF001

    sent = chatbot._answering_agent.ainvoke.call_args[0][0]["messages"]  # noqa: SLF001
    assert len(sent) == SupportBot.ANSWER_HISTORY_MESSAGES + 1
    assert sent[0].content == "message 20"
    assert sent[-1].content == "and now?"


@pytest.mark.asyncio
async def test_try_to_answer_leaves_withdrawn_messages_out_of_the_attempt(
    mocker, mock_checkpointer
):
    chatbot = await _answering_bot(mocker, mock_checkpointer)
    history = [
        HumanMessage("withdrawn problem"),
        AIMessage("cancelled"),
        HumanMessage("a fresh start"),
    ]

    await chatbot._try_to_answer(  # noqa: SLF001
        "and my grade?", {"messages": history, "cancelled_after": [2]}
    )

    sent = chatbot._answering_agent.ainvoke.call_args[0][0]["messages"]  # noqa: SLF001
    assert [msg.content for msg in sent] == ["a fresh start", "and my grade?"]


@pytest.mark.asyncio
async def test_try_to_answer_keeps_the_review_block_out_of_the_attempt(
    mocker, mock_checkpointer
):
    """The review quotes the reply-to. Called directly: no path reaches it today."""
    chatbot = await _answering_bot(mocker, mock_checkpointer)
    history = [
        HumanMessage("I can't log in"),
        AIMessage(_support_review("I can't log in", "", "learner@example.com")),
    ]

    await chatbot._try_to_answer("and my grade?", {"messages": history})  # noqa: SLF001

    sent = chatbot._answering_agent.ainvoke.call_args[0][0]["messages"]  # noqa: SLF001
    assert not any("learner@example.com" in str(msg.content) for msg in sent)


async def _support_bot(
    mocker, mock_checkpointer, prior_messages, state=None, user_agent=""
):
    """Build a SupportBot whose thread already holds prior_messages, LLM stubbed."""
    thread_id = uuid4().hex
    await UserChatSession.objects.acreate(thread_id=thread_id, agent="SupportBot")
    chatbot = await sync_to_async(SupportBot)(
        "anonymous", mock_checkpointer, thread_id=thread_id, user_agent=user_agent
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


async def _answering_support_bot(mocker, mock_checkpointer, answer, state=None):
    """Build a SupportBot whose attempt returns answer, or each list item in turn."""
    thread_id = uuid4().hex
    await UserChatSession.objects.acreate(thread_id=thread_id, agent="SupportBot")
    chatbot = await sync_to_async(SupportBot)(
        "learner", mock_checkpointer, thread_id=thread_id, enable_account_tools=True
    )
    if state:
        await chatbot.agent.aupdate_state(
            chatbot.config,
            # Not empty: with tools, the agent node's edge reads messages[-1].
            {
                "messages": [
                    HumanMessage("my certificate is missing"),
                    AIMessage("Here's what I'll send to the support team:"),
                ],
                **state,
            },
            as_node="agent",
        )
    mocker.patch.object(
        SupportBot,
        "_try_to_answer",
        AsyncMock(side_effect=answer)
        if isinstance(answer, list)
        else AsyncMock(return_value=answer),
    )
    return chatbot


async def _say(chatbot, message, **extra):
    return "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                message, extra_state={**_support_state("learner@mit.edu"), **extra}
            )
        ]
    )


async def _state_of(chatbot):
    return (await chatbot.agent.aget_state(chatbot.config)).values


async def _awaiting_send(chatbot):
    return ((await _state_of(chatbot)).get("awaiting_send") or [False])[-1]


@pytest.mark.asyncio
async def test_an_intake_in_progress_is_not_interrupted_by_an_answer(
    mocker, mock_checkpointer
):
    """Once a turn fell to intake, the next one owes the review, not an answer."""
    chatbot = await _answering_support_bot(mocker, mock_checkpointer, ["", "an answer"])

    first = await _say(chatbot, SUPPORT_PROBLEM)
    second = await _say(chatbot, "it also happens on week 4")

    assert first.startswith(SUPPORT_ASK_MORE)
    assert SupportBot._try_to_answer.await_count == 1  # noqa: SLF001
    assert SUPPORT_PROBLEM in second
    assert "an answer" not in second


@pytest.mark.asyncio
async def test_a_close_out_reopens_answering(mocker, mock_checkpointer):
    chatbot = await _answering_support_bot(
        mocker, mock_checkpointer, ["first answer", "second answer"]
    )

    await _say(chatbot, "did my certificate come through?")
    closed = await _say(chatbot, "thanks")
    again = await _say(chatbot, "when does 18.01x start?")

    # get_completion appends the checkpoint comment, so this is "in", not "=="
    assert SUPPORT_ANSWER_CLOSED in closed
    assert "second answer" in again


@pytest.mark.asyncio
async def test_support_bot_answers_and_still_offers_the_ticket(
    mocker, mock_checkpointer
):
    chatbot = await _answering_support_bot(
        mocker, mock_checkpointer, "You're enrolled in 18.01x."
    )

    reply = await _say(chatbot, "what am I enrolled in?")

    assert "You're enrolled in 18.01x." in reply
    assert SUPPORT_OFFER_TICKET in reply
    state = await _state_of(chatbot)
    # An answer is not a request, so nothing is queued to send.
    assert (state.get("awaiting_send") or [False])[-1] is False
    assert state["answered"][-1] is True


@pytest.mark.asyncio
async def test_an_answer_reaches_the_learner_from_a_real_tool_call(
    mocker, mock_checkpointer
):
    """End to end through _the_answer_in; the other tests mock _try_to_answer."""
    thread_id = uuid4().hex
    await UserChatSession.objects.acreate(thread_id=thread_id, agent="SupportBot")
    chatbot = await sync_to_async(SupportBot)(
        "learner", mock_checkpointer, thread_id=thread_id, enable_account_tools=True
    )
    mocker.patch.object(
        chatbot._answering_agent,  # noqa: SLF001
        "ainvoke",
        AsyncMock(
            return_value=_answer_result(
                _tool_message(
                    '{"enrollments": [], "note": "There is no course on this page."}'
                ),
                AIMessage("Which course did you mean?"),
            )
        ),
    )

    reply = await _say(chatbot, "what's my grade?")

    assert "Which course did you mean?" in reply
    assert SUPPORT_ASK_MORE not in reply


@pytest.mark.asyncio
async def test_support_bot_collects_when_it_cannot_answer(mocker, mock_checkpointer):
    chatbot = await _answering_support_bot(mocker, mock_checkpointer, "")

    reply = await _say(chatbot, "why was I charged twice?")

    assert SUPPORT_ASK_MORE in reply
    assert SUPPORT_OFFER_TICKET not in reply


@pytest.mark.asyncio
@pytest.mark.parametrize("close_out", ["thanks", "ok thanks!", "thanks so much"])
@pytest.mark.parametrize("answering_goes_off", [False, True])
async def test_support_bot_closes_out_an_answer_without_filing(
    mocker, mock_checkpointer, close_out, answering_goes_off
):
    """
    A second attempt would find nothing and show a review, so only the close-out
    keeps this from filing. answering_goes_off is a flag rollback or an expired
    cookie between turns.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 912}),
    )
    chatbot = await _answering_support_bot(
        mocker, mock_checkpointer, ["You're enrolled in 18.01x.", ""]
    )
    await _say(chatbot, "what am I enrolled in?")
    if answering_goes_off:
        chatbot._answering_agent = None  # noqa: SLF001

    reply = await _say(chatbot, close_out)

    assert SUPPORT_ANSWER_CLOSED in reply
    mock_file.assert_not_called()
    state = await _state_of(chatbot)
    assert (state.get("awaiting_send") or [False])[-1] is False
    assert state["answered"][-1] is False
    assert state["cancelled_after"][-1] == len(state["messages"])


@pytest.mark.asyncio
async def test_send_after_disagreeing_with_an_answer_files_both(
    mocker, mock_checkpointer
):
    """
    Disagreeing is a follow-up, so it gets another answer; the offer's "send"
    then takes the learner's question and their disagreement to the review.
    """
    chatbot = await _answering_support_bot(
        mocker,
        mock_checkpointer,
        ["Your grade in AT.101x is 90.0%.", "Your grade in AT.101x is still 90.0%."],
    )
    await _say(chatbot, "What's my grade in AT.101x?")
    disagreement = "That's not right, I submitted all my assignments last week."

    again = await _say(chatbot, disagreement)
    review = await _say(chatbot, "send")

    assert "still 90.0%" in again
    assert SUPPORT_OFFER_TICKET in again
    assert SUPPORT_TICKET_REVIEW_HEADER in review
    assert "What's my grade in AT.101x?" in review
    assert disagreement in review
    assert (await _state_of(chatbot))["awaiting_send"][-1] is True


@pytest.mark.asyncio
async def test_send_it_after_an_answer_is_not_a_close_out(mocker, mock_checkpointer):
    chatbot = await _answering_support_bot(
        mocker, mock_checkpointer, "You're enrolled in 18.01x."
    )
    await _say(chatbot, "my certificate is missing")

    reply = await _say(chatbot, "send it")

    assert SUPPORT_ANSWER_CLOSED not in reply


@pytest.mark.asyncio
async def test_a_filler_after_a_close_out_asks_rather_than_showing_an_empty_review(
    mocker, mock_checkpointer
):
    """The second attempt returns nothing: a filler isn't a records question."""
    chatbot = await _answering_support_bot(
        mocker, mock_checkpointer, ["You're enrolled in 18.01x.", ""]
    )
    await _say(chatbot, "what am I enrolled in?")
    await _say(chatbot, "thanks")

    reply = await _say(chatbot, "ok")

    assert SUPPORT_ASK_PROBLEM in reply
    assert "**Your message:**" not in reply
    assert await _awaiting_send(chatbot) is False


@pytest.mark.parametrize(
    ("reply_to_the_ask", "expected"),
    [
        ("no", SUPPORT_NOTHING_TO_SEND),
        ("no thanks", SUPPORT_NOTHING_TO_SEND),
        ("nothing else", SUPPORT_NOTHING_TO_SEND),
        ("that's all", SUPPORT_NOTHING_TO_SEND),
        ("cancel", SUPPORT_NOTHING_TO_SEND),
        # ANSWER_CLOSING_PHRASES, which FILLER_PHRASES doesn't contain.
        ("thanks", SUPPORT_NOTHING_TO_SEND),
        ("thank you", SUPPORT_NOTHING_TO_SEND),
        ("got it", SUPPORT_NOTHING_TO_SEND),
        ("perfect", SUPPORT_NOTHING_TO_SEND),
        ("cheers", SUPPORT_NOTHING_TO_SEND),
        ("that helped", SUPPORT_NOTHING_TO_SEND),
        # Calling off a cancel isn't a request either.
        ("don't cancel", SUPPORT_NOTHING_TO_SEND),
        ("wait, don't cancel", SUPPORT_NOTHING_TO_SEND),
        ("i don't want to cancel", SUPPORT_NOTHING_TO_SEND),
        # A "yes" means more is coming.
        ("yes", SUPPORT_ASK_PROBLEM),
    ],
)
@pytest.mark.asyncio
async def test_a_reply_to_the_empty_ask(
    mocker, mock_checkpointer, reply_to_the_ask, expected
):
    chatbot = await _answering_support_bot(
        mocker, mock_checkpointer, ["You're enrolled in 18.01x.", "", ""]
    )
    await _say(chatbot, "what am I enrolled in?")
    await _say(chatbot, "thanks")
    await _say(chatbot, "ok")

    reply = await _say(chatbot, reply_to_the_ask)

    assert expected in reply
    assert (
        {SUPPORT_ASK_PROBLEM, SUPPORT_NOTHING_TO_SEND} - {expected}
    ).pop() not in reply
    assert "**Your message:**" not in reply
    assert await _awaiting_send(chatbot) is False


@pytest.mark.asyncio
async def test_that_decline_reopens_answering_like_the_other_two_exits(
    mocker, mock_checkpointer
):
    # Three, not four: the decline turn never reaches an attempt.
    chatbot = await _answering_support_bot(
        mocker,
        mock_checkpointer,
        ["You're enrolled in 18.01x.", "", "Your certificate is ready."],
    )
    await _say(chatbot, "what am I enrolled in?")
    await _say(chatbot, "thanks")
    await _say(chatbot, "ok")
    await _say(chatbot, "no")

    reply = await _say(chatbot, "where's my certificate?")

    assert "Your certificate is ready." in reply
    assert "**Your message:**" not in reply


@pytest.mark.asyncio
async def test_yes_after_an_answer_takes_up_the_offer(mocker, mock_checkpointer):
    chatbot = await _answering_support_bot(
        mocker, mock_checkpointer, ["You're enrolled in 18.01x.", "another answer"]
    )
    await _say(chatbot, "my certificate is missing")

    reply = await _say(chatbot, "yes")

    assert SupportBot._try_to_answer.await_count == 1  # noqa: SLF001
    assert "another answer" not in reply
    assert SUPPORT_OFFER_TICKET not in reply
    assert SUPPORT_ANSWER_CLOSED not in reply
    assert "my certificate is missing" in reply
    assert (await _state_of(chatbot))["answered"][-1] is False


@pytest.mark.asyncio
async def test_support_bot_escalates_to_the_intake_script(mocker, mock_checkpointer):
    """The button means a person, so the bot stops trying to be one."""
    chatbot = await _answering_support_bot(
        mocker, mock_checkpointer, "You're enrolled in 18.01x."
    )

    reply = await _say(chatbot, "Contact the support team", escalate=[True])

    assert SUPPORT_ASK_PROBLEM in reply
    assert SUPPORT_OFFER_TICKET not in reply
    SupportBot._try_to_answer.assert_not_called()  # noqa: SLF001
    state = await _state_of(chatbot)
    assert state["handed_off"][-1] is True
    # A checkpointed escalate would re-escalate every later turn.
    assert "escalate" not in state


@pytest.mark.asyncio
async def test_escalate_sticks_even_under_the_review(mocker, mock_checkpointer):
    """A later cancel mustn't resume answering someone who asked for a person."""
    chatbot = await _answering_support_bot(
        mocker, mock_checkpointer, "answer", state={"awaiting_send": [True]}
    )

    await _say(chatbot, "Contact the support team", escalate=[True])

    assert (await _state_of(chatbot))["handed_off"][-1] is True


@pytest.mark.asyncio
async def test_support_bot_never_answers_again_after_a_hand_off(
    mocker, mock_checkpointer
):
    chatbot = await _answering_support_bot(
        mocker, mock_checkpointer, "answer", state={"handed_off": [True]}
    )

    await _say(chatbot, "what's my grade?")

    SupportBot._try_to_answer.assert_not_called()  # noqa: SLF001


@pytest.mark.asyncio
async def test_an_escalate_under_the_review_leaves_the_request_alone(
    mocker, mock_checkpointer
):
    """Under the review the button's label would otherwise read as an edit."""
    label = "Contact the support team"
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM), AIMessage("Here's what I'll send...")],
        state={"user_email": ["learner@mit.edu"], "awaiting_send": [True]},
    )

    review = await _say(chatbot, label, escalate=[True])

    assert SUPPORT_PROBLEM in review
    assert label not in review


@pytest.mark.asyncio
async def test_support_bot_keeps_the_escalate_buttons_words_out_of_the_ticket(
    mocker, mock_checkpointer
):
    """A press carries the button's label, which isn't the learner's problem."""
    label = "Contact the support team"
    chatbot = await _answering_support_bot(mocker, mock_checkpointer, "an answer")
    await _say(chatbot, SUPPORT_PROBLEM)

    review = await _say(chatbot, label, escalate=[True])
    # The edit reads the checkpointed message, not just this turn's description.
    edited = await _say(chatbot, "It also happens on week 4.")

    assert SUPPORT_PROBLEM in review
    assert label not in review
    assert label not in edited
    assert "It also happens on week 4." in edited


@pytest.mark.asyncio
async def test_an_escalate_after_a_close_out_asks_instead_of_filing_nothing(
    mocker, mock_checkpointer
):
    chatbot = await _answering_support_bot(
        mocker, mock_checkpointer, ["You're enrolled in 18.01x.", ""]
    )
    await _say(chatbot, "what am I enrolled in?")
    await _say(chatbot, "thanks")

    reply = await _say(chatbot, "Contact the support team", escalate=[True])

    assert SUPPORT_ASK_PROBLEM in reply
    assert "**Your message:**" not in reply
    assert await _awaiting_send(chatbot) is False


@pytest.mark.asyncio
async def test_an_escalate_on_a_new_thread_does_not_make_filler_the_request(
    mocker, mock_checkpointer
):
    """Answering is off: escalate is read before the gate."""
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 912}),
    )
    chatbot = await _support_bot(mocker, mock_checkpointer, [])

    asked = await _say(chatbot, "Contact the support team", escalate=[True])
    reply = await _say(chatbot, "thanks")

    assert SUPPORT_ASK_PROBLEM in asked
    assert SUPPORT_NOTHING_TO_SEND in reply
    assert "**Your message:**" not in reply
    assert await _awaiting_send(chatbot) is False

    await _say(chatbot, "yes")

    mock_file.assert_not_called()


@pytest.mark.asyncio
async def test_a_scripted_support_bot_never_attempts_an_answer(
    mocker, mock_checkpointer
):
    """Flag off or anonymous learner: the feature must be invisible."""
    mocker.patch.object(SupportBot, "_try_to_answer", AsyncMock(return_value="hi"))
    chatbot = await _support_bot(mocker, mock_checkpointer, [])

    reply = await _say(chatbot, SUPPORT_PROBLEM)

    assert SUPPORT_ASK_MORE in reply
    SupportBot._try_to_answer.assert_not_called()  # noqa: SLF001
    assert chatbot._answering_agent is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_support_bot_reviews_before_it_files(mocker, mock_checkpointer):
    """
    Nothing reaches the support team until the learner has seen what is going.
    The review is built in code for the same reason the ticket is: a model
    asked to repeat it back would paraphrase it into something untrue.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 912}),
    )
    chatbot = await _support_bot(mocker, mock_checkpointer, [])

    # Turn 1: we already know the authenticated learner's address, so the bot
    # invites more detail rather than filing or reviewing straight away.
    first = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                SUPPORT_PROBLEM, extra_state=_support_state("signedin@mit.edu")
            )
        ]
    )
    assert SUPPORT_ASK_MORE in first

    # Turn 2: now the review is shown, still without filing.
    reply = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                "that's everything", extra_state=_support_state("signedin@mit.edu")
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
    ["send", "Send it.", "yes please", "ok", "go ahead", "Submit"],
)
@pytest.mark.asyncio
async def test_support_bot_files_on_a_go_ahead_after_the_review(
    mocker, mock_checkpointer, confirmation
):
    """
    A learner who has read the review and said yes has to get their request
    filed on that turn - there is no later turn that would do it for them.
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
    assert "learner@example.com" in reply

    async for _ in chatbot.get_completion("send", extra_state=_support_state()):
        pass

    assert mock_file.call_args.kwargs["email"] == "learner@example.com"


@pytest.mark.asyncio
async def test_support_bot_keeps_a_correction_the_widget_keeps_overwriting(
    mocker, mock_checkpointer
):
    """
    A host that puts the learner's address in every request body would otherwise
    undo the correction on the turn that files: the body value accumulates after
    the saved one, and a bare "send" has no typed address to prefer over it.
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

    async for _ in chatbot.get_completion(
        "learner@example.com", extra_state=_support_state("learner@exmaple.com")
    ):
        pass
    async for _ in chatbot.get_completion(
        "send", extra_state=_support_state("learner@exmaple.com")
    ):
        pass

    assert mock_file.call_args.kwargs["email"] == "learner@example.com"


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

    # The correction is an edit, so it is shown back; the send after it fails.
    async for _ in chatbot.get_completion(
        "learner@example.com", extra_state=_support_state()
    ):
        pass
    async for _ in chatbot.get_completion("send", extra_state=_support_state()):
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
async def test_support_bot_files_with_the_requests_browser(mocker, mock_checkpointer):
    """
    The browser comes from the request that files, not from the thread: the
    approval turn is itself an HTTP request carrying its own User-Agent, so
    there is nothing to accumulate and nothing to write into every checkpoint.
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
        user_agent="Mozilla/5.0 (iPhone)",
    )

    async for _ in chatbot.get_completion(
        "send", extra_state=_support_state("signedin@mit.edu")
    ):
        pass

    assert mock_file.call_args.kwargs["user_agent"] == "Mozilla/5.0 (iPhone)"


@pytest.mark.asyncio
async def test_support_bot_files_without_a_browser(mocker, mock_checkpointer):
    """A caller that sends no User-Agent still files."""
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
        "send", extra_state=_support_state("signedin@mit.edu")
    ):
        pass

    assert mock_file.call_args.kwargs["user_agent"] == ""


@pytest.mark.asyncio
async def test_support_bot_ticket_matches_the_filing_signature(
    mocker, mock_checkpointer
):
    """
    Every other test here mocks file_support_ticket, so a ticket key the real
    function does not take would pass the suite and blow up on the ** splat in
    production. Bind what was built against the real signature.
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
        user_agent="Mozilla/5.0 (iPhone)",
    )

    async for _ in chatbot.get_completion(
        "send", extra_state=_support_state("signedin@mit.edu")
    ):
        pass

    inspect.signature(file_support_ticket).bind(**mock_file.call_args.kwargs)


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
    async for _ in chatbot.get_completion("send", extra_state=_support_state()):
        pass

    description = mock_file.call_args.kwargs["description"]
    assert SUPPORT_PROBLEM in description
    assert "Chrome on Windows." in description
    # The address is the reply-to, not part of the problem.
    assert "learner@example.com" not in description


@pytest.mark.asyncio
async def test_support_bot_keeps_an_opening_message_that_quotes_an_address(
    mocker, mock_checkpointer
):
    """
    "I cannot log in as x@y.com" is the single most likely way to open a
    support request, and it reads to _reply_to_email exactly like an answer to
    "what's your email?". Dropping it filed an empty ticket under the subject
    "Support request". The opening message is the problem, whatever it looks
    like.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    opening = "I cannot log in as learner@example.com"
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [
            HumanMessage(opening),
            AIMessage("What's your email address?"),
            HumanMessage("learner@example.com"),
            AIMessage("Here's what I'll send..."),
        ],
        state={"user_email": ["learner@example.com"], "awaiting_send": [True]},
    )

    async for _ in chatbot.get_completion("send", extra_state=_support_state()):
        pass

    assert mock_file.call_args.kwargs["description"] == opening
    assert mock_file.call_args.kwargs["subject"] == opening


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
    and a ticket exists. Filing happens in code, so it does not depend on the
    model choosing to emit a tool call.
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
    assert SUPPORT_ASK_EMAIL in reply
    SupportBot.send_chunks.assert_not_called()


@pytest.mark.asyncio
async def test_support_bot_runs_the_whole_intake_without_the_model(
    mocker, mock_checkpointer
):
    """
    An anonymous learner is walked from problem to filed ticket entirely in
    code: ask for an address, show the review, file on the confirmation. The
    model is never called on any turn.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    chatbot = await _support_bot(mocker, mock_checkpointer, [])

    ask = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                SUPPORT_PROBLEM, extra_state=_support_state()
            )
        ]
    )
    assert SUPPORT_ASK_EMAIL in ask

    review = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                "learner@example.com", extra_state=_support_state()
            )
        ]
    )
    assert SUPPORT_PROBLEM in review
    assert "learner@example.com" in review

    filed = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                "send", extra_state=_support_state()
            )
        ]
    )
    assert "4821" in filed

    mock_file.assert_called_once()
    SupportBot.send_chunks.assert_not_called()


@pytest.mark.asyncio
async def test_support_bot_re_asks_when_no_email_is_given(mocker, mock_checkpointer):
    """
    A learner who answers the email question with more detail instead of an
    address is asked again rather than dropped or filed.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 1}),
    )
    chatbot = await _support_bot(mocker, mock_checkpointer, [])

    first = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                SUPPORT_PROBLEM, extra_state=_support_state()
            )
        ]
    )
    assert SUPPORT_ASK_EMAIL in first

    again = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                "it also happens on mobile", extra_state=_support_state()
            )
        ]
    )
    assert SUPPORT_ASK_EMAIL_AGAIN in again
    mock_file.assert_not_called()
    SupportBot.send_chunks.assert_not_called()


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
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={"user_email": ["signedin@mit.edu"], "awaiting_send": [True]},
    )
    # File the ticket, so the next turn reads an existing reference back.
    async for _ in chatbot.get_completion(
        "send", extra_state=_support_state("signedin@mit.edu")
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
    async for _ in chatbot.get_completion(
        "send", extra_state={"page_url": [""], "user_email": [""]}
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

    signed_in = {
        **_support_state("signedin@mit.edu"),
        "verified_email": ["signedin@mit.edu"],
    }
    async for _ in chatbot.get_completion(
        "use victim@example.com", extra_state=signed_in
    ):
        pass
    async for _ in chatbot.get_completion("send", extra_state=signed_in):
        pass

    assert mock_file.call_args.kwargs["email"] == "signedin@mit.edu"
    assert mock_file.call_args.kwargs["email_verified"] is True


@pytest.mark.parametrize(
    ("reply_text", "expected"),
    [
        ("the page says contact admin@example.org", "learner@mit.edu"),
        ("that error names ops@example.org", "learner@mit.edu"),
        ("sorry, it's new@mit.edu", "new@mit.edu"),
        ("new@mit.edu", "new@mit.edu"),
    ],
)
@pytest.mark.asyncio
async def test_support_bot_only_takes_a_bare_address_as_a_correction(
    mocker, mock_checkpointer, reply_text, expected
):
    """
    An address quoted while describing the problem is not the learner asking
    us to write back to it, and filing against one makes Zendesk raise a
    ticket in a stranger's name. Retyping the address still has to work.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 331}),
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={"user_email": ["learner@mit.edu"]},
    )
    # No signed-in session, so the widget sends no address of its own and the
    # one the learner typed earlier comes back out of the thread.
    state = _support_state()

    async for _ in chatbot.get_completion(reply_text, extra_state=state):
        pass
    async for _ in chatbot.get_completion("send", extra_state=state):
        pass

    assert mock_file.call_args.kwargs["email"] == expected


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
    async for _ in chatbot.get_completion("send", extra_state=_support_state()):
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
    async for _ in chatbot.get_completion(
        "send", extra_state={"page_url": [""], "user_email": [""]}
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
async def test_support_bot_files_a_short_message_that_mentions_an_address(
    mocker, mock_checkpointer
):
    """
    A short message naming an address is read as the reply-to, and the learner
    sees that in the review. What it says about the problem is theirs either
    way, so it cannot be dropped on the way to the ticket.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    chatbot = await _support_bot(
        mocker, mock_checkpointer, [HumanMessage(SUPPORT_PROBLEM)]
    )
    detail = "the refund email bursar@mit.edu bounced"

    review = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                detail, extra_state=_support_state()
            )
        ]
    )
    async for _ in chatbot.get_completion("send", extra_state=_support_state()):
        pass

    assert detail in review
    assert mock_file.call_args.kwargs["description"] == f"{SUPPORT_PROBLEM}\n\n{detail}"


@pytest.mark.parametrize("detail", ["no 18.01", "ok 7.2", "yes 好"])
@pytest.mark.asyncio
async def test_support_bot_files_a_reply_carrying_a_number(
    mocker, mock_checkpointer, detail
):
    """
    A course number or an accented word is the only thing support can act on,
    so a reply that opens like an answer to us still has to reach the ticket.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={"user_email": ["signedin@mit.edu"]},
    )
    state = _support_state("signedin@mit.edu")

    async for _ in chatbot.get_completion(detail, extra_state=state):
        pass
    async for _ in chatbot.get_completion("send", extra_state=state):
        pass

    assert mock_file.call_args.kwargs["description"] == f"{SUPPORT_PROBLEM}\n\n{detail}"


@pytest.mark.asyncio
async def test_support_bot_marks_a_session_email_as_verified(mocker, mock_checkpointer):
    """An address from the authenticated session is the one support can trust."""
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    signed_in = {
        "page_url": ["https://learn.mit.edu/week-3"],
        "user_email": ["signedin@mit.edu"],
        "verified_email": ["signedin@mit.edu"],
    }
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={**signed_in, "awaiting_send": [True]},
    )

    async for _ in chatbot.get_completion("send", extra_state=signed_in):
        pass

    assert mock_file.call_args.kwargs["email_verified"] is True


@pytest.mark.asyncio
async def test_support_bot_does_not_inherit_verification_from_an_earlier_turn(
    mocker, mock_checkpointer
):
    """
    Verification describes the request in hand, not the thread. A turn that
    arrives with no session - however it reached the thread - must not borrow
    the sign-in an earlier turn had, or the ticket tells support an address was
    checked when nobody checked it.
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
            "user_email": ["signedin@mit.edu"],
            "verified_email": ["signedin@mit.edu"],
            "awaiting_send": [True],
        },
    )

    anonymous = {"page_url": [""], "user_email": [""], "verified_email": [""]}
    async for _ in chatbot.get_completion("send", extra_state=anonymous):
        pass

    assert mock_file.call_args.kwargs["email_verified"] is False


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
    async for _ in chatbot.get_completion("send", extra_state=extra_state):
        pass

    assert mock_file.call_args.kwargs["email_verified"] is False


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
    chatbot = await _support_bot(
        mocker, mock_checkpointer, [HumanMessage(SUPPORT_PROBLEM)]
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


@pytest.mark.parametrize(
    ("message", "cancels"),
    [
        # A cancellation word on its own, or with the send negated.
        ("cancel", True),
        ("stop", True),
        ("actually, never mind", True),
        ("wait, don't send it", True),
        ("Cancel, don't send this", True),
        # "no" before the word is agreement with the offer to cancel.
        ("no, cancel it", True),
        # Negated: the learner wants the opposite.
        ("i don't want to cancel", False),
        ("don't cancel", False),
        ("please don't cancel it", False),
        ("not cancelling", False),
        # Carrying detail, so it is an edit rather than an answer either way.
        ("cancel the enrollment I never asked for", False),
    ],
)
def test_is_cancellation(message, cancels):
    """A negated trigger reads as the opposite; a plain "no" in front agrees."""
    assert _is_cancellation(message) is cancels


@pytest.mark.parametrize(
    "cancellation",
    [
        "Cancel, don't send this",
        "cancel",
        "wait, don't send it",
        # The same words with the apostrophe a phone keyboard produces.
        "wait, don\u2019t send it",
        # The review's own word, quoted back, with the quotes a phone makes.
        "\u2018cancel\u2019",
        "actually, never mind",
        "stop",
    ],
)
@pytest.mark.asyncio
async def test_support_bot_does_not_file_when_the_learner_cancels(
    mocker, mock_checkpointer, cancellation
):
    """
    A learner who reads the review and changes their mind must be able to stop
    it. Filing anyway sends the support team a request the learner withdrew.
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
                cancellation, extra_state=_support_state("signedin@mit.edu")
            )
        ]
    )

    mock_file.assert_not_called()
    assert SUPPORT_TICKET_CANCELLED in reply


@pytest.mark.asyncio
async def test_support_bot_does_not_file_after_a_cancellation(
    mocker, mock_checkpointer
):
    """
    Cancelling has to clear the pending send, not just skip one turn. Otherwise
    the learner's next message - whatever it is - files the request they just
    withdrew.
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
        "cancel", extra_state=_support_state("signedin@mit.edu")
    ):
        pass
    async for _ in chatbot.get_completion(
        "The week 4 video is broken too.",
        extra_state=_support_state("signedin@mit.edu"),
    ):
        pass

    mock_file.assert_not_called()


@pytest.mark.asyncio
async def test_support_bot_leaves_a_cancelled_request_out_of_the_next_one(
    mocker, mock_checkpointer
):
    """
    Cancelling withdraws the request, not just the send. Carrying the withdrawn
    words into the next ticket files what the learner took back - and on a
    shared device, puts one person's problem in another person's ticket.
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

    async def say(message):
        return "".join(
            [
                chunk
                async for chunk in chatbot.get_completion(
                    message, extra_state=_support_state("signedin@mit.edu")
                )
            ]
        )

    await say("cancel")
    review = await say("My certificate name is spelled wrong.")
    await say("send")

    # The review for the new request must not show the cancelled one either -
    # the learner approves what they are shown.
    assert SUPPORT_PROBLEM not in review
    description = mock_file.call_args.kwargs["description"]
    assert "certificate name" in description
    assert SUPPORT_PROBLEM not in description


@pytest.mark.asyncio
async def test_support_bot_shows_an_edit_back_before_filing(mocker, mock_checkpointer):
    """
    Late detail is an edit to the ticket, so the learner has to see the edited
    version before it goes. Filing on it sends something they never read.
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
                "It happens on my phone too.",
                extra_state=_support_state("signedin@mit.edu"),
            )
        ]
    )

    mock_file.assert_not_called()
    assert "It happens on my phone too." in reply
    assert SUPPORT_PROBLEM in reply

    async for _ in chatbot.get_completion(
        "send", extra_state=_support_state("signedin@mit.edu")
    ):
        pass

    assert (
        mock_file.call_args.kwargs["description"]
        == f"{SUPPORT_PROBLEM}\n\nIt happens on my phone too."
    )


@pytest.mark.asyncio
async def test_support_bot_treats_no_as_nothing_more_to_add(mocker, mock_checkpointer):
    """
    Before the review, "no" answers "anything else to add?", so it moves on to
    the review. Reading it as a cancellation there would strand every learner
    who answers the question we asked.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={"user_email": ["signedin@mit.edu"]},
    )

    reply = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                "no", extra_state=_support_state("signedin@mit.edu")
            )
        ]
    )

    mock_file.assert_not_called()
    assert SUPPORT_PROBLEM in reply
    assert SUPPORT_TICKET_CANCELLED not in reply

    async for _ in chatbot.get_completion(
        "send", extra_state=_support_state("signedin@mit.edu")
    ):
        pass

    mock_file.assert_called_once()
    assert mock_file.call_args.kwargs["description"] == SUPPORT_PROBLEM


@pytest.mark.parametrize("decline", ["no", "No.", "nope", "no thanks"])
@pytest.mark.asyncio
async def test_support_bot_does_not_file_when_the_learner_declines(
    mocker, mock_checkpointer, decline
):
    """
    Under the review the question on screen is whether to send, so "no" there
    means do not send. Filing on it was the reported bug: the learner says no
    and the support team gets the request anyway.
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
                decline, extra_state=_support_state("signedin@mit.edu")
            )
        ]
    )

    mock_file.assert_not_called()
    assert SUPPORT_TICKET_CANCELLED in reply


@pytest.mark.parametrize(
    "reply_text",
    [
        "yes please send it to support asap",
        "yes, and the course is 18.01",
        "yes, 18.01",
        "ok 7.2",
        "ok 好",
        "that's all",
        "nothing else",
        "done",
    ],
)
@pytest.mark.asyncio
async def test_support_bot_asks_when_the_go_ahead_is_unclear(
    mocker, mock_checkpointer, reply_text
):
    """
    "yes please send it to support asap" adds nothing to the ticket while
    "yes, and the course is 18.01" does, and nothing here can tell them apart.
    Guessing either way is wrong: one pastes chatter into the ticket, the other
    files without the detail the learner just gave.
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
                reply_text, extra_state=_support_state("signedin@mit.edu")
            )
        ]
    )

    mock_file.assert_not_called()
    assert SUPPORT_SEND_UNCLEAR in reply
    assert reply_text not in reply


@pytest.mark.asyncio
async def test_support_bot_keeps_the_send_pending_after_an_unclear_reply(
    mocker, mock_checkpointer
):
    """
    Asking must not drop the learner back into intake, and the words we could
    not read must not reach the ticket.
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
        "yes please send it to support asap",
        extra_state=_support_state("signedin@mit.edu"),
    ):
        pass
    async for _ in chatbot.get_completion(
        "send", extra_state=_support_state("signedin@mit.edu")
    ):
        pass

    mock_file.assert_called_once()
    assert mock_file.call_args.kwargs["description"] == SUPPORT_PROBLEM


@pytest.mark.asyncio
async def test_support_bot_treats_a_long_reply_as_an_edit(mocker, mock_checkpointer):
    """
    A reply long enough to carry real detail is an edit even when it opens with
    "yes" - short chatter is what we cannot read, not a sentence.
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
    detail = "yes and the week 4 video spins on Firefox and Chrome as well"

    reply = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                detail, extra_state=_support_state("signedin@mit.edu")
            )
        ]
    )

    mock_file.assert_not_called()
    assert detail in reply


@pytest.mark.asyncio
async def test_support_bot_files_on_an_explicit_confirm_flag(mocker, mock_checkpointer):
    """
    A widget with a Send button sends confirm=True rather than a word we have
    to recognise, so the flag alone has to be enough to file.
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
        "looks right to me",
        extra_state={**_support_state("signedin@mit.edu"), "confirm": [True]},
    ):
        pass

    mock_file.assert_called_once()
    assert mock_file.call_args.kwargs["description"] == SUPPORT_PROBLEM


@pytest.fixture
def support_filing_cache(settings):
    """
    Give the filing lock a cache that remembers, so the test exercises it.

    get_django_cache returns the redis cache whenever a broker is configured,
    and conftest makes that one a DummyCache: aadd there always succeeds, so the
    lock would be a no-op and the test would pass without it.
    """
    settings.CELERY_BROKER_URL = ""
    settings.CACHES = {
        **settings.CACHES,
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "support-filing-test",
        },
    }
    cache = get_django_cache()
    cache.clear()
    return cache


@pytest.mark.asyncio
async def test_support_bot_files_one_ticket_when_two_turns_land_together(
    mocker, mock_checkpointer, support_filing_cache
):
    """
    Two sends on one thread must produce one ticket. The already-filed check
    reads the reference the first turn has not written yet, so without a lock
    around the call both turns get past it.
    """
    in_flight = asyncio.Event()
    finish = asyncio.Event()
    calls = 0

    async def slow_file(**_kwargs):
        # Only the first call hangs around. A second one getting this far is the
        # bug, and it has to return so the assertion below reports it.
        nonlocal calls
        calls += 1
        if calls == 1:
            in_flight.set()
            await finish.wait()
        return {"reference": 4821}

    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket", AsyncMock(side_effect=slow_file)
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={"user_email": ["signedin@mit.edu"], "awaiting_send": [True]},
    )
    # A second request on the same thread is a second bot, not this one again.
    other = await sync_to_async(SupportBot)(
        "anonymous", mock_checkpointer, thread_id=chatbot.thread_id
    )

    async def drain(bot):
        return "".join(
            [
                chunk
                async for chunk in bot.get_completion(
                    "send", extra_state=_support_state("signedin@mit.edu")
                )
            ]
        )

    first = asyncio.create_task(drain(chatbot))
    await in_flight.wait()
    second_reply = await drain(other)
    finish.set()
    first_reply = await first

    assert mock_file.call_count == 1
    assert "4821" in first_reply
    assert SUPPORT_TICKET_SENDING in second_reply


@pytest.mark.asyncio
async def test_support_bot_leaves_a_lock_it_no_longer_owns(
    mocker, mock_checkpointer, support_filing_cache
):
    """
    Httpx bounds each read and write, not the whole call, so the lock TTL can
    lapse while a filing is still running. The slow caller must not come back
    and free the lock whoever took over is relying on.
    """
    cache = support_filing_cache
    lock_key = ""
    successor = uuid4().hex

    async def slow_file(**_kwargs):
        # The TTL runs out mid-call and the next turn takes the lock.
        await cache.adelete(lock_key)
        await cache.aadd(lock_key, value=successor, timeout=300)
        return {"reference": 4821}

    mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket", AsyncMock(side_effect=slow_file)
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={"user_email": ["signedin@mit.edu"], "awaiting_send": [True]},
    )
    lock_key = f"{SUPPORT_FILING_LOCK_PREFIX}{chatbot.thread_id}"

    async for _ in chatbot.get_completion(
        "send", extra_state=_support_state("signedin@mit.edu")
    ):
        pass

    assert await cache.aget(lock_key) == successor


@pytest.mark.asyncio
async def test_support_bot_holds_the_filing_lock_for_the_whole_zendesk_call(
    mocker, mock_checkpointer
):
    """
    The lock is the only thing between two concurrent sends and two tickets. One
    that expires while the first call is still in flight hands the second turn a
    free pass at exactly the moment it must not have one.
    """
    held = {}
    cache = mocker.Mock()

    async def _aadd(key, value=None, timeout=None):
        held[key] = value
        return True

    cache.aadd = AsyncMock(side_effect=_aadd)
    cache.aget = AsyncMock(side_effect=held.get)
    cache.adelete = AsyncMock(side_effect=lambda key: held.pop(key, None))
    mocker.patch("ai_chatbots.chatbots.get_django_cache", return_value=cache)
    mocker.patch(
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
        "send", extra_state=_support_state("signedin@mit.edu")
    ):
        pass

    assert cache.aadd.call_args.kwargs["timeout"] > max_filing_seconds()


@pytest.mark.asyncio
async def test_support_bot_does_not_file_when_the_reference_lands_mid_turn(
    mocker, mock_checkpointer, support_filing_cache
):
    """
    The already-filed check runs at the top of the turn, so an overlapping turn
    can file after it and before this one calls Zendesk. Re-reading the row
    before the call is what closes that window.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 9999}),
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={"user_email": ["signedin@mit.edu"], "awaiting_send": [True]},
    )
    read_state = SupportBot._prior_state  # noqa: SLF001

    async def file_in_another_turn(self):
        await UserChatSession.objects.filter(thread_id=self.thread_id).aupdate(
            support_ticket_reference="4821",
            support_ticket_email="signedin@mit.edu",
        )
        return await read_state(self)

    mocker.patch.object(SupportBot, "_prior_state", file_in_another_turn)

    reply = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                "send", extra_state=_support_state("signedin@mit.edu")
            )
        ]
    )

    mock_file.assert_not_called()
    assert "4821" in reply


@pytest.mark.asyncio
async def test_support_bot_releases_the_filing_lock_after_a_failure(
    mocker, mock_checkpointer, support_filing_cache
):
    """
    A failed send has to leave the thread free to retry. A lock held until its
    TTL expires would answer the retry with "still sending" instead.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(side_effect=[{"error": "nope"}, {"reference": 4821}]),
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
    reply = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                "try again", extra_state=_support_state("signedin@mit.edu")
            )
        ]
    )

    assert mock_file.call_count == 2
    assert "4821" in reply


@pytest.mark.asyncio
async def test_support_bot_does_not_file_again_after_an_unclear_outcome(
    mocker, mock_checkpointer
):
    """
    Zendesk can create the ticket and still fail to answer. Calling that a
    failure invites the retry that duplicates it, so the thread stops instead
    and every later message gets the same answer.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"error": "Could not file.", "uncertain": True}),
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={"user_email": ["signedin@mit.edu"], "awaiting_send": [True]},
    )

    async def say(message):
        return "".join(
            [
                chunk
                async for chunk in chatbot.get_completion(
                    message, extra_state=_support_state("signedin@mit.edu")
                )
            ]
        )

    first = await say("send")
    # Three turns, because only clearing awaiting_send would drop the learner
    # back into intake, re-show the review, and file on the send after that.
    replies = [first, await say("still nothing"), await say("send")]

    assert mock_file.call_count == 1
    assert all(SUPPORT_TICKET_MAYBE_FILED in reply for reply in replies)
    # The wording has to stop short of the claim the failure reply makes.
    assert "could not" not in first.lower()
    session = await UserChatSession.objects.aget(thread_id=chatbot.thread_id)
    assert session.support_ticket_reference == ""


@pytest.mark.asyncio
async def test_support_bot_records_an_unclear_outcome_before_replying(
    mocker, mock_checkpointer
):
    """
    An uncertain filing lives only in the thread state, so a learner whose
    connection drops as the reply goes out must not come back to a thread that
    will happily file a second ticket.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"error": "Could not file.", "uncertain": True}),
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={"user_email": ["signedin@mit.edu"], "awaiting_send": [True]},
    )
    state = _support_state("signedin@mit.edu")

    # Take the reply and walk away, the way a closed tab does.
    stream = chatbot.get_completion("send", extra_state=state)
    assert SUPPORT_TICKET_MAYBE_FILED in await anext(stream)
    await stream.aclose()

    retry = "".join(
        [chunk async for chunk in chatbot.get_completion("send", extra_state=state)]
    )

    assert mock_file.call_count == 1
    assert SUPPORT_TICKET_MAYBE_FILED in retry


@pytest.mark.asyncio
async def test_support_bot_stops_asking_after_zendesk_turns_the_ticket_down(
    mocker, mock_checkpointer
):
    """
    Zendesk answers a rejected ticket the same way every time, so inviting
    another "send" walks the learner around a loop that cannot end.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(
            return_value={
                "error": "Could not file.",
                "uncertain": False,
                "reason": TICKET_REJECTED,
            }
        ),
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={"user_email": ["signedin@mit.edu"], "awaiting_send": [True]},
    )

    async def say(message):
        return "".join(
            [
                chunk
                async for chunk in chatbot.get_completion(
                    message, extra_state=_support_state("signedin@mit.edu")
                )
            ]
        )

    replies = [await say("send"), await say("send"), await say("try it again")]

    assert mock_file.call_count == 1
    assert all(SUPPORT_TICKET_REJECTED in reply for reply in replies)
    assert SUPPORT_TICKET_FAILED not in replies[0]


@pytest.mark.asyncio
async def test_support_bot_asks_for_an_address_zendesk_will_accept(
    mocker, mock_checkpointer
):
    """
    An address Zendesk will not take is the one rejection the learner can fix,
    so the thread goes back to collecting instead of stopping.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(
            side_effect=[
                {
                    "error": "Bad address.",
                    "uncertain": False,
                    "reason": TICKET_BAD_EMAIL,
                },
                {"reference": 4821},
            ]
        ),
    )
    chatbot = await _support_bot(
        mocker,
        mock_checkpointer,
        [HumanMessage(SUPPORT_PROBLEM)],
        state={"user_email": ["typo@mit"], "awaiting_send": [True]},
    )

    async def say(message):
        return "".join(
            [
                chunk
                async for chunk in chatbot.get_completion(
                    message, extra_state={"page_url": ["https://learn.mit.edu/week-3"]}
                )
            ]
        )

    rejected = await say("send")
    await say("sorry, it's me@mit.edu")
    filed = await say("send")

    assert SUPPORT_TICKET_BAD_EMAIL in rejected
    assert mock_file.call_count == 2
    assert mock_file.call_args.kwargs["email"] == "me@mit.edu"
    assert mock_file.call_args.kwargs["description"] == SUPPORT_PROBLEM
    assert "4821" in filed


@pytest.mark.asyncio
async def test_support_bot_keeps_confirm_out_of_the_thread_state(
    mocker, mock_checkpointer
):
    """
    The confirm flag is this turn's Send button, not a fact about the thread.
    Storing it files the learner's one approval against every turn after it.
    """
    mocker.patch(
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
        "looks right to me",
        extra_state={**_support_state("signedin@mit.edu"), "confirm": [True]},
    ):
        pass

    state = await chatbot.agent.aget_state(chatbot.config)
    assert "confirm" not in state.values


def test_max_filing_seconds_covers_every_phase_of_every_request(settings):
    """
    A scalar httpx timeout bounds connect, write, read and pool separately, so
    one request can burn four of them. A budget that counts each request once
    expires while the call it is sizing is still running.
    """
    settings.REQUESTS_TIMEOUT = 30

    assert max_filing_seconds() >= 30 * HTTPX_TIMEOUT_PHASES * WORST_PATH_REQUESTS


@pytest.mark.asyncio
async def test_support_bot_does_not_file_again_when_the_session_write_fails(
    mocker, mock_checkpointer
):
    """
    Zendesk has the ticket by the time the reference is written down. Letting
    that write's failure escape loses the only record the next turn reads, and
    the next turn then sends the same request a second time.
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
    mocker.patch.object(
        UserChatSession,
        "asave",
        AsyncMock(side_effect=OperationalError("connection lost")),
    )

    reply = "".join(
        [
            chunk
            async for chunk in chatbot.get_completion(
                "send", extra_state=_support_state("signedin@mit.edu")
            )
        ]
    )
    async for _ in chatbot.get_completion(
        "any update?", extra_state=_support_state("signedin@mit.edu")
    ):
        pass

    assert "4821" in reply
    assert mock_file.await_count == 1


@pytest.mark.asyncio
async def test_support_bot_clears_the_pending_send_once_it_has_filed(
    mocker, mock_checkpointer
):
    """
    A send that succeeded is no longer pending. Leaving the flag set keeps the
    thread one affirmative away from filing again, with only the session row
    standing in the way.
    """
    mocker.patch(
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
        "send", extra_state=_support_state("signedin@mit.edu")
    ):
        pass

    state = await chatbot.agent.aget_state(chatbot.config)
    assert state.values["awaiting_send"][-1] is False


@pytest.mark.asyncio
async def test_support_bot_leaves_a_cancelled_filler_out_of_the_next_request(
    mocker, mock_checkpointer
):
    """
    The opening message is kept whatever it looks like because nobody has been
    asked anything yet. After a cancellation that no longer holds, and keeping
    the first thing typed regardless puts a stray "ok" at the top of the ticket.
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

    for message in ("cancel", "ok", "The week 4 video is broken too.", "send"):
        async for _ in chatbot.get_completion(
            message, extra_state=_support_state("signedin@mit.edu")
        ):
            pass

    description = mock_file.await_args.kwargs["description"]
    assert description == "The week 4 video is broken too."


@pytest.mark.asyncio
async def test_support_bot_asks_rather_than_dropping_the_request_on_wait(
    mocker, mock_checkpointer
):
    """
    "wait" is a learner asking for a moment, not for their request to be thrown
    away. Reading it as a cancellation discards everything they typed, and the
    reply tells them it is gone.
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
                "wait", extra_state=_support_state("signedin@mit.edu")
            )
        ]
    )
    async for _ in chatbot.get_completion(
        "send", extra_state=_support_state("signedin@mit.edu")
    ):
        pass

    assert SUPPORT_SEND_UNCLEAR in reply
    assert SUPPORT_TICKET_CANCELLED not in reply
    assert mock_file.await_args.kwargs["description"] == SUPPORT_PROBLEM


@pytest.mark.asyncio
async def test_support_bot_keeps_a_pause_out_of_the_description(
    mocker, mock_checkpointer
):
    """
    "wait" is addressed to us, so it has nothing in it for support to read. It
    reaches the description through intake, where no review has been shown yet
    and so nothing has been asked about it.
    """
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 4821}),
    )
    chatbot = await _support_bot(
        mocker, mock_checkpointer, [HumanMessage(SUPPORT_PROBLEM)]
    )

    for message in ("wait", "It started on Monday.", "send"):
        async for _ in chatbot.get_completion(
            message, extra_state=_support_state("signedin@mit.edu")
        ):
            pass

    description = mock_file.await_args.kwargs["description"]
    assert description == f"{SUPPORT_PROBLEM}\n\nIt started on Monday."


@pytest.mark.asyncio
async def test_a_failed_answer_leaves_only_the_scripted_exchange(
    mocker, mock_checkpointer
):
    """Runs the real method against a raising ainvoke and reads the checkpoint."""
    thread_id = uuid4().hex
    await UserChatSession.objects.acreate(thread_id=thread_id, agent="SupportBot")
    chatbot = await sync_to_async(SupportBot)(
        "learner", mock_checkpointer, thread_id=thread_id, enable_account_tools=True
    )
    mocker.patch.object(
        chatbot._answering_agent,  # noqa: SLF001
        "ainvoke",
        AsyncMock(side_effect=RuntimeError("the model is down")),
    )

    reply = await _say(chatbot, SUPPORT_PROBLEM)

    assert reply.startswith(SUPPORT_ASK_MORE)
    messages = (await _state_of(chatbot))["messages"]
    assert [type(msg) for msg in messages] == [HumanMessage, AIMessage]
    assert messages[0].content == SUPPORT_PROBLEM


@pytest.mark.asyncio
async def test_a_bot_that_cannot_answer_runs_todays_intake_unchanged(
    mocker, mock_checkpointer
):
    """The exact key set catches a new channel leaking into a non-answering thread."""
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 77}),
    )
    chatbot = await _support_bot(mocker, mock_checkpointer, [])

    first = await _say(chatbot, SUPPORT_PROBLEM)
    review = await _say(chatbot, "nothing else")
    filed = await _say(chatbot, "yes")

    # _say carries an address, so the opening turn asks for more detail.
    assert first.startswith(SUPPORT_ASK_MORE)
    assert SUPPORT_PROBLEM in review
    assert "77" in filed
    assert mock_file.call_args.kwargs["description"] == SUPPORT_PROBLEM
    state = await _state_of(chatbot)
    assert state.get("answered", []) == []
    assert state.get("handed_off", []) == []


def test_support_description_leaves_the_bots_own_words_out():
    """An answer can quote the learner's records, which mustn't reach Zendesk."""
    prior = [
        HumanMessage("did I pass 18.01x?"),
        AIMessage("You scored 94% and your certificate was issued on 3 May."),
    ]

    assert _support_description(prior, "and my refund?") == (
        "did I pass 18.01x?\n\nand my refund?"
    )


@pytest.mark.parametrize(
    "closing", ["thanks", "thank you", "got it", "perfect", "cheers", "that helped"]
)
def test_support_description_drops_a_closing_phrase(closing):
    prior = [
        HumanMessage("did I pass 18.01x?"),
        AIMessage("You scored 94%."),
    ]

    assert _support_description(prior, closing, since=2) == ""


@pytest.mark.asyncio
async def test_a_ticket_filed_after_an_answer_carries_none_of_the_answer(
    mocker, mock_checkpointer
):
    """Privacy: the description filter alone keeps records out of the ticket."""
    mock_file = mocker.patch(
        "ai_chatbots.chatbots.file_support_ticket",
        AsyncMock(return_value={"reference": 91}),
    )
    answer = "You scored 94% in 18.01x and your certificate was issued on 3 May."
    chatbot = await _answering_support_bot(mocker, mock_checkpointer, [answer])

    answered = await _say(chatbot, "did I pass 18.01x?")
    await _say(chatbot, "yes")
    await _say(chatbot, "send")

    # The learner did see their records; it is the ticket they stay out of.
    assert "94%" in answered
    description = mock_file.await_args.kwargs["description"]
    assert description == "did I pass 18.01x?"


@pytest.mark.asyncio
async def test_the_answering_graph_hands_the_tools_the_page_course(
    settings, mocker, mock_checkpointer
):
    """Real graph and a probe tool: a mocked ainvoke hid the dropped course_id."""
    settings.LEARNER_API_OPENEDX_BASE_URL = "https://courses.learn.test/"
    mocker.patch.object(CompiledStateGraph, "astream", _REAL_ASTREAM)
    seen = {}

    @tool
    def probe(state: Annotated[dict, InjectedState]) -> str:
        """Record the course the tools can see."""
        seen["course_id"] = (state or {}).get("course_id")
        seen["platform"] = (state or {}).get("platform")
        return '{"enrollments": []}'

    chatbot = await sync_to_async(SupportBot)(
        "learner", mock_checkpointer, enable_account_tools=True
    )
    chatbot.tools = [probe]
    chatbot.llm = _FakeToolCallingModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "probe", "args": {}, "id": "call-1"}],
                ),
                AIMessage(content="Your grade is 88%."),
            ]
        )
    )
    chatbot._build_answering_agent()  # noqa: SLF001

    await chatbot._try_to_answer(  # noqa: SLF001
        "what's my grade?",
        {},
        {
            "page_url": [
                (
                    "https://courses.learn.test/learn/course/"
                    "course-v1:MITx+18.01x+2T2025/home"
                )
            ]
        },
    )

    assert seen["course_id"] == ["course-v1:MITx+18.01x+2T2025"]
    assert seen["platform"] == ["mitxonline"]
