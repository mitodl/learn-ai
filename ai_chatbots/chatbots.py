"""Agent service classes for the AI chatbots"""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from operator import add
from typing import Annotated, Any
from uuid import uuid4

import posthog
from django.conf import settings
from django.utils.module_loading import import_string
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.tools.base import BaseTool
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from langgraph.prebuilt.chat_agent_executor import AgentState, _validate_chat_history
from open_learning_ai_tutor.message_tutor import message_tutor
from open_learning_ai_tutor.prompts import get_system_prompt
from open_learning_ai_tutor.tools import tutor_tools
from open_learning_ai_tutor.utils import (
    filter_out_system_messages,
    json_to_intent_list,
    json_to_messages,
    tutor_output_to_json,
)
from openai import BadRequestError

from ai_chatbots import tools
from ai_chatbots.admin import LLMModel
from ai_chatbots.api import (
    DjangoCheckpoint,
    MessageTruncationNode,
    create_tutorbot_output_and_checkpoints,
    get_resource_facts,
    get_search_tool_metadata,
    query_tutorbot_output,
)
from ai_chatbots.constants import AZURE_MODEL_PREFIX
from ai_chatbots.models import UserChatSession
from ai_chatbots.posthog import TokenTrackingCallbackHandler
from ai_chatbots.prompts import (
    CONTEXT_LOST_PROMPT,
    PROMPT_SEARCH_SUMMARY_QUERY,
    SYSTEM_PROMPT_MAPPING,
)
from ai_chatbots.utils import (
    async_request,
    comment_safe_json,
    get_azure_openai_kwargs,
    get_django_cache,
    get_run_readable_id_from_block_id,
    save_truncated_checkpoint,
    truncate_to_latest_human_message,
)
from ai_chatbots.zendesk import (
    TICKET_BAD_EMAIL,
    TICKET_REJECTED,
    file_support_ticket,
    max_filing_seconds,
)
from main.opik_keycloak_auth import is_opik_configured

log = logging.getLogger(__name__)


class BaseChatbot(ABC):
    """
    Base AI chatbot class
    """

    PROMPT_TEMPLATE = "base"

    # For LiteLLM tracking purposes
    TASK_NAME = "BASE_TASK"
    JOB_ID = "BASECHAT_JOB"
    STATE_CLASS = AgentState

    TRACE_STATE_KEYS: tuple[str, ...] = ()

    # A bot that scripts every reply spends no tokens, so it has no budget to
    # provision. Setting this False skips that call, which is a blocking
    # request that would otherwise make the proxy a dependency of a bot that
    # never reaches it.
    CALLS_THE_MODEL = True

    def __init__(  # noqa: PLR0913
        self,
        user_id: str,
        checkpointer: BaseCheckpointSaver,
        *,
        name: str = "MIT Open Learning Chatbot",
        model: str | None = None,
        temperature: float | None = None,
        instructions: str | None = None,
        thread_id: str | None = None,
    ):
        """Initialize the AI chat agent service"""
        self.bot_name = name
        self.model = model or settings.AI_DEFAULT_MODEL
        self.temperature = temperature or settings.AI_DEFAULT_TEMPERATURE
        self.instructions = (
            instructions
            or get_system_prompt(
                self.PROMPT_TEMPLATE, SYSTEM_PROMPT_MAPPING, get_django_cache
            )
            if self.PROMPT_TEMPLATE
            else None
        )
        self.user_id = user_id
        self.thread_id = thread_id or uuid4().hex
        self.config = {"configurable": {"thread_id": self.thread_id}}
        self.checkpointer = checkpointer
        if settings.AI_PROXY_CLASS:
            self.proxy = import_string(
                f"ai_chatbots.proxies.{settings.AI_PROXY_CLASS}"
            )()
            if self.CALLS_THE_MODEL:
                self.proxy.create_proxy_user(self.user_id)
            self.proxy_prefix = self.proxy.PROXY_MODEL_PREFIX
        else:
            self.proxy = None
            self.proxy_prefix = ""
        self.tools = self.create_tools()
        self.llm = self.get_llm()
        self.agent = None

    def create_tools(self):
        """Create any tools required by the agent"""
        return []

    def get_llm(self, **kwargs) -> BaseChatModel:
        """
        Return the LLM instance for the chatbot.
        Set it up to use a proxy, with required proxy kwargs, if applicable.
        Bind the LLM to any tools if they are present.
        """
        model_spec = LLMModel.objects.filter(litellm_id=self.model).first()
        # Set reasoning effort if specified for the model
        model_kwargs = (
            {"reasoning_effort": model_spec.reasoning_effort}
            if model_spec and model_spec.reasoning_effort
            else {}
        )
        # A configured proxy handles azure/ models itself
        azure_kwargs = (
            get_azure_openai_kwargs()
            if not self.proxy and self.model.startswith(AZURE_MODEL_PREFIX)
            else {}
        )
        model_kwargs.update(azure_kwargs.pop("model_kwargs", {}))
        llm = ChatLiteLLM(
            model=f"{self.proxy_prefix}{self.model}",
            streaming=True,
            # Explicitly request token usage on the final streaming chunk.
            # (langchain_litellm already defaults this on; we set it so the
            # behavior survives a library default change.)
            stream_options={"include_usage": True},
            model_kwargs=model_kwargs,
            **azure_kwargs,
            **(self.proxy.get_api_kwargs() if self.proxy else {}),
            **(self.proxy.get_additional_kwargs(self) if self.proxy else {}),
            **kwargs,
        )
        # Set the temperature if appropriate
        llm.temperature = (
            self.temperature or model_spec.temperature
            if (model_spec and model_spec.temperature is not None)
            else None
        )
        # Bind tools to the LLM if any
        if self.tools:
            llm = llm.bind_tools(self.tools)
        return llm

    def create_agent_graph(self) -> CompiledStateGraph:
        """
        Return a graph for the relevant LLM and tools.

        An easy way to create a graph is to use the prebuilt create_react_agent:

            from langgraph.prebuilt import create_react_agent

            return create_react_agent(
                self.llm,
                tools=self.tools,
                checkpointer=self.checkpointer,
                prompt=self.instructions,
            )

        The base implementation here accomplishes the same thing but a
        bit more explicitly, to give a better idea of what's happening
        and how it can be customized.
        """

        # Names of nodes in the graph
        agent_node = "agent"
        tools_node = "tools"

        def tool_calling_llm(state: AgentState) -> AgentState:
            """Call the LLM, injecting system prompt"""
            if len(state["messages"]) == 1:
                # New chat, so inject the system prompt
                state["messages"].insert(0, SystemMessage(self.instructions))
            return self.STATE_CLASS(messages=[self.llm.invoke(state["messages"])])

        agent_graph = StateGraph(MessagesState)
        # Add the agent node that first calls the LLM
        agent_graph.add_node(agent_node, tool_calling_llm)
        if self.tools:
            # Add the tools node
            agent_graph.add_node(tools_node, ToolNode(tools=self.tools))
            # Add a conditional edge that determines when to run the tools.
            # If no tool call is requested, the edge is not taken and the
            # agent node will end its response.
            agent_graph.add_conditional_edges(agent_node, tools_condition)
            # Send the tool node output back to the agent node
            agent_graph.add_edge(tools_node, agent_node)
        # Set the entry point to the agent node
        agent_graph.set_entry_point(agent_node)

        # compile and return the agent graph
        return agent_graph.compile(checkpointer=self.checkpointer)

    async def get_latest_history(self) -> dict:
        """Get the most recent state history"""
        async for state in self.agent.aget_state_history(self.config):
            if state:
                return state
        return None

    async def validate_and_clean_checkpoint(self) -> None:
        """
        Clean the current checkpoint if it has validation errors.
        """
        try:
            # Get the current state
            current_state = await self.agent.aget_state(self.config)
            if not current_state or not current_state.values:
                return

            messages = current_state.values.get("messages", [])
            if not messages:
                return

            # Validate using LangChain's built-in validation
            # This is done to make sure it's actually a problem
            # with the chat history, and not something else
            try:
                _validate_chat_history(messages)
            except ValueError:
                # if validation failed, truncate to latest HumanMessage
                cleaned_messages = truncate_to_latest_human_message(messages)
                keep_ids = {msg.id for msg in cleaned_messages}
                await save_truncated_checkpoint(self.thread_id, keep_ids)

        except Exception:
            log.exception("Error while cleaning checkpoint")

    async def _get_latest_checkpoint_id(self) -> str | None:
        """Get the most recent assistant response checkpoint"""
        checkpoint = (
            await DjangoCheckpoint.objects.prefetch_related("session", "session__user")
            .filter(
                thread_id=self.thread_id,
            )
            .order_by("-id")
            .afirst()
        )
        return checkpoint.id if checkpoint else None

    def get_trace_properties(
        self, extra_state: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Return extra properties to tag traces with, pulled from the state
        keys named in TRACE_STATE_KEYS.  State values arrive as lists (they
        are reducer-appended fields), so single-item lists are unwrapped.
        """
        state = extra_state or {}
        properties = {}
        for key in self.TRACE_STATE_KEYS:
            value = state.get(key)
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
            if value:
                properties[key] = value
        return properties

    async def set_callbacks(
        self, properties: dict | None = None
    ) -> list[BaseCallbackHandler]:
        """Set callbacks for the agent LLM"""
        callbacks = []
        extra_props = properties or {}
        if settings.POSTHOG_PROJECT_API_KEY and settings.POSTHOG_API_HOST:
            hog_client = posthog.Posthog(
                settings.POSTHOG_PROJECT_API_KEY, host=settings.POSTHOG_API_HOST
            )
            model_parts = self.model.rsplit("/", 1)
            callback_handler = TokenTrackingCallbackHandler(
                model_name=self.model,
                client=hog_client,
                bot=self,
                properties={
                    "$ai_trace_id": self.thread_id,
                    "$ai_span_name": self.JOB_ID,
                    "$ai_model": model_parts[-1],
                    "$ai_provider": model_parts[0],
                    "distinct_id": self.user_id,
                    "botName": self.JOB_ID,
                    "user": self.user_id,
                    **extra_props,
                },
            )
            callbacks.append(callback_handler)
        if is_opik_configured():
            from ai_chatbots.opik_tracing import CostTrackingOpikTracer

            callbacks.append(
                CostTrackingOpikTracer(
                    # Log the agent graph so Opik can render it in the trace
                    # sidebar ("Show Agent Graph"). xray=True expands nested
                    # subgraphs (e.g. ToolNode internals) for full visibility.
                    graph=self.agent.get_graph(xray=True) if self.agent else None,
                    project_name=settings.OPIK_PROJECT_NAME,
                    thread_id=self.thread_id,
                    tags=[self.JOB_ID],
                    metadata={
                        "botName": self.JOB_ID,
                        "model": self.model,
                        "user": self.user_id,
                        **extra_props,
                    },
                )
            )
        return callbacks

    async def send_chunks(self, state) -> AsyncGenerator[str, None]:
        """Yield the response in chunks"""
        response_generator = self.agent.astream(
            state,
            self.config,
            stream_mode="messages",
        )
        async for chunk in response_generator:
            if (
                isinstance(chunk[0], AIMessage | AIMessageChunk)
                and chunk[1].get("langgraph_node") != "pre_model_hook"
            ):
                yield chunk[0].content

    async def get_completion(
        self,
        message: str,
        *,
        extra_state: dict[str, Any] | None = None,
        debug: bool = settings.AI_DEBUG,
    ) -> AsyncGenerator[str, None]:
        """
        Send the user message to the agent and yield the response as
        it comes in.

        Append the response with debugging metadata and/or errors.
        """
        full_response = ""
        if not self.agent:
            error = "Create agent before running"
            raise ValueError(error)
        try:
            trace_properties = self.get_trace_properties(extra_state)

            self.config["metadata"] = {
                **self.config.get("metadata", {}),
                **trace_properties,
            }
            self.config["callbacks"] = await self.set_callbacks(
                properties=trace_properties
            )
            state = {
                "messages": [HumanMessage(message)],
                **(extra_state or {}),
            }
            try:
                async for chunk in self.send_chunks(state):
                    full_response += chunk
                    yield chunk
            except ValueError:
                # Validate and clean checkpoint, then try again
                log.exception("Validation error, cleaning checkpoint")
                await self.validate_and_clean_checkpoint()
                # Add a system message to notify the user about lost context
                state["messages"].append(
                    SystemMessage(
                        content=CONTEXT_LOST_PROMPT,
                        id=str(uuid4()),
                    )
                )
                # Try to answer the user's last question anyway
                async for chunk in self.send_chunks(state):
                    full_response += chunk
                    yield chunk
        except BadRequestError as error:
            log.exception("Bad request error")
            # Format and yield an error message inside a hidden comment
            if hasattr(error, "response"):
                error = error.response.json()
            else:
                error = {
                    "error": {"message": "An error has occurred, please try again"}
                }
            if (
                error["error"]["message"].startswith("Budget has been exceeded")
                and not debug
            ):  # Friendlier message for end user
                error["error"]["message"] = (
                    "You have exceeded your AI usage limit. Please try again later."
                )
            yield f"<!-- {comment_safe_json(error)} -->"
        except Exception:
            yield '<!-- {"error":{"message":"An error occurred, please try again"}} -->'
            log.exception("Error running AI agent")
        metadata = await self.get_metadata(debug=debug)
        yield f"\n\n<!-- {metadata} -->\n\n"

    async def get_metadata(self, *, debug: bool = False) -> str:
        """
        Return metadata JSON about the response
        """
        # After streaming completes, get the latest checkpoint ID
        metadata = {} if not debug else await self.get_tool_metadata()
        metadata["checkpoint_pk"] = await self._get_latest_checkpoint_id()
        metadata["thread_id"] = self.thread_id
        if metadata:
            return comment_safe_json(metadata)
        return ""

    @abstractmethod
    async def get_tool_metadata(self) -> str:
        """
        Return metadata JSON about the response
        """
        raise NotImplementedError


class SummaryState(AgentState):
    """
    AgentState with context field to keep track of previous summary information
    """

    context: dict[str, Any]


class TruncatingChatbot(BaseChatbot):
    """
    Chatbot that truncates chat history to keep only recent messages.

    This prevents context window errors by limiting messages sent to the LLM,
    while preserving all messages in the database. Counts only human messages
    to determine truncation point.
    """

    STATE_CLASS = AgentState

    def create_agent_graph(self) -> CompiledStateGraph:
        """
        Generate a standard react agent graph with message truncation.

        Truncates to last N human messages (and their responses) before
        sending to LLM.
        """
        log.debug("Instructions: \n%s\n\n", self.instructions)

        return create_react_agent(
            self.llm,
            tools=self.tools,
            checkpointer=self.checkpointer,
            pre_model_hook=MessageTruncationNode(),
            state_schema=self.STATE_CLASS,
            prompt=self.instructions,
        )


class RecommendationAgentState(SummaryState):
    """
    State for the recommendation bot. Passes search url
    to the associated tool function.
    """

    search_url: Annotated[list[str], add]


class ResourceRecommendationBot(TruncatingChatbot):
    """
    Chatbot that searches for learning resources in the MIT Learn catalog,
    then recommends the best results to the user based on their query.
    """

    PROMPT_TEMPLATE = "recommendation"
    TASK_NAME = "RECOMMENDATION_TASK"
    JOB_ID = "RECOMMENDATION_JOB"
    STATE_CLASS = RecommendationAgentState

    def __init__(  # noqa: PLR0913
        self,
        user_id: str,
        checkpointer: BaseCheckpointSaver | None = None,
        *,
        name: str = "MIT Open Learning Chatbot",
        model: str | None = None,
        temperature: float | None = None,
        instructions: str | None = None,
        thread_id: str | None = None,
    ):
        """Initialize the AI search agent service"""
        super().__init__(
            user_id,
            name=name,
            checkpointer=checkpointer,
            model=model or settings.AI_DEFAULT_RECOMMENDATION_MODEL,
            temperature=temperature,
            instructions=instructions,
            thread_id=thread_id,
        )
        self.agent = self.create_agent_graph()

    def create_tools(self) -> list[BaseTool]:
        """Create tools required by the agent"""
        return [tools.search_courses, tools.search_content_files]

    async def get_tool_metadata(self) -> str:
        """Return the metadata for the search tool"""
        thread_id = self.config["configurable"]["thread_id"]
        latest_state = await self.get_latest_history()
        return get_search_tool_metadata(thread_id, latest_state)


class SearchSummaryBot(ResourceRecommendationBot):
    """
    Recommendation bot that summarizes relevant courses for a search query
    on the MIT Learn search page.  The first message of a thread is the search
    query, which gets wrapped in the summary instructions; follow-up messages
    are handled like any other recommendation bot conversation.
    """

    TASK_NAME = "SEARCH_SUMMARY_TASK"
    JOB_ID = "SEARCH_SUMMARY_JOB"

    async def get_completion(
        self,
        message: str,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Wrap the search query in the summary instructions for a new thread"""
        state = await self.agent.aget_state(self.config)
        if not (state and state.values.get("messages")):
            # The query comes from a URL, so flatten it, keep it short, and strip
            # anything that could close the <search_query> tag in the prompt.
            query = " ".join(message.split())[:200].replace("<", "").replace(">", "")
            message = PROMPT_SEARCH_SUMMARY_QUERY.format(query=query)
        async for chunk in super().get_completion(message, **kwargs):
            yield chunk


class SyllabusAgentState(SummaryState):
    """
    State for the syllabus bot. Passes course_id and
    collection_name to the associated tool function.
    """

    course_id: Annotated[list[str], add]
    platform: Annotated[list[str] | None, add]
    collection_name: Annotated[list[str], add]
    related_courses: Annotated[list[str], add]
    # str representation of a boolean value, because the
    # langgraph JsonPlusSerializer can't handle booleans
    exclude_canvas: Annotated[list[str] | None, add]


class SyllabusBot(TruncatingChatbot):
    """Service class for the AI syllabus agent"""

    PROMPT_TEMPLATE = "syllabus"
    TASK_NAME = "SYLLABUS_TASK"
    JOB_ID = "SYLLABUS_JOB"
    STATE_CLASS = SyllabusAgentState
    TRACE_STATE_KEYS = ("course_id", "collection_name", "related_courses", "platform")

    def __init__(  # noqa: PLR0913
        self,
        user_id: str,
        checkpointer: BaseCheckpointSaver,
        *,
        name: str = "MIT Open Learning Syllabus Chatbot",
        model: str | None = None,
        temperature: float | None = None,
        instructions: str | None = None,
        thread_id: str | None = None,
        enable_related_courses: bool | None = False,
        course_id: str | None = None,
        platform: str | None = None,
    ):
        self.enable_related_courses = enable_related_courses
        super().__init__(
            user_id,
            name=name,
            checkpointer=checkpointer,
            model=model or settings.AI_DEFAULT_SYLLABUS_MODEL,
            temperature=temperature,
            instructions=instructions,
            thread_id=thread_id,
        )
        # The graph captures the instructions, so the facts have to be in
        # place before create_agent_graph below.
        resource_facts = get_resource_facts(course_id, platform)
        if resource_facts and self.instructions:
            self.instructions = f"{self.instructions.rstrip()}\n\n{resource_facts}"
        if self.enable_related_courses and self.instructions:
            self.instructions += (
                "\n\nIMPORTANT: You have two content search tools. Always "
                "call BOTH search_content_files AND "
                "search_related_course_content_files for questions about the "
                "course material. search_content_files searches the current "
                "course, while search_related_course_content_files searches "
                "other courses in the same program. Combine results from both "
                "tools to provide the most comprehensive answer. Neither of "
                "them replaces search_support_articles, which is still the "
                "tool for questions about how MIT platforms work rather than "
                "about these courses."
            )
        self.agent = self.create_agent_graph()

    def create_tools(self):
        """Create tools required by the agent"""
        bot_tools = [tools.search_content_files]
        if self.enable_related_courses:
            bot_tools.append(tools.search_related_course_content_files)
        bot_tools.append(tools.search_support_articles)
        return bot_tools

    async def get_tool_metadata(self) -> str:
        """Return the metadata for the search tool"""
        thread_id = self.config["configurable"]["thread_id"]
        latest_state = await self.get_latest_history()
        return get_search_tool_metadata(thread_id, latest_state)


class CanvasSyllabusBot(SyllabusBot):
    """Service class for the Canvas syllabus agent"""

    PROMPT_TEMPLATE = "syllabus_canvas"
    TASK_NAME = "CANVAS_SYLLABUS_TASK"
    JOB_ID = "CANVAS_SYLLABUS_JOB"
    STATE_CLASS = SyllabusAgentState


class TutorBot(BaseChatbot):
    """
    Chatbot that assists with problem sets
    """

    PROMPT_TEMPLATE = None
    TASK_NAME = "TUTOR_TASK"
    JOB_ID = "TUTOR_JOB"

    def __init__(  # noqa: PLR0913
        self,
        user_id: str,
        checkpointer: BaseCheckpointSaver | None = BaseCheckpointSaver,
        *,
        name: str = "MIT Open Learning Tutor Chatbot",
        model: str | None = None,
        temperature: float | None = None,
        thread_id: str | None = None,
        block_siblings: list[str] | None = None,
        edx_module_id: str | None = None,
        run_readable_id: str | None = None,
        problem_set_title: str | None = None,
    ):
        super().__init__(
            user_id,
            name=name,
            checkpointer=checkpointer,
            temperature=temperature,
            thread_id=thread_id,
            model=model or settings.AI_DEFAULT_TUTOR_MODEL,
        )

        self.edx_module_id = edx_module_id
        self.block_siblings = block_siblings
        self.run_readable_id = run_readable_id or get_run_readable_id_from_block_id(
            edx_module_id
        )
        self.problem_set_title = problem_set_title

        if not self.edx_module_id:
            self.variant = "canvas"
            self.problem = ""
        else:
            self.variant = "edx"
            self.problem = None

        self.problem_set = None
        self.problem_data_loaded = False

    def get_trace_properties(
        self,
        extra_state: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        """
        Return the identifiers for the problem the tutor was asked about.
        They arrive as init kwargs rather than graph state, so they are read
        off the instance instead of via TRACE_STATE_KEYS.

        """
        return {
            key: value
            for key, value in (
                ("run_readable_id", self.run_readable_id),
                ("problem_set_title", self.problem_set_title),
                ("edx_module_id", self.edx_module_id),
            )
            if value
        }

    async def get_tool_metadata(self) -> str:
        """Return the metadata for the  tool"""
        await self.load_problem_data()
        return {
            "edx_module_id": self.edx_module_id,
            "block_siblings": self.block_siblings,
            "problem": self.problem,
            "problem_set": self.problem_set,
            "problem_set_title": self.problem_set_title,
            "run_readable_id": self.run_readable_id,
        }

    async def get_latest_history(self):
        return await query_tutorbot_output(self.thread_id)

    async def get_completion(
        self,
        message: str,
        *,
        extra_state: dict[str, Any] | None = None,  # noqa: ARG002
        debug: bool = settings.AI_DEBUG,
    ) -> AsyncGenerator[str, None]:
        """Call message_tutor with the user query and return the response"""

        await self.load_problem_data()

        history = await self.get_latest_history()
        message_id = str(uuid4())
        if history:
            json_history = json.loads(history.chat_json)
            chat_history = json_to_messages(  # noqa: RUF005
                json_history.get("chat_history", [])
            ) + [HumanMessage(content=message, id=message_id)]

            intent_history = json_to_intent_list(json_history["intent_history"])
            assessment_history = json_to_messages(json_history["assessment_history"])
        else:
            chat_history = [HumanMessage(content=message, id=message_id)]
            intent_history = []
            assessment_history = []
        self.llm.callbacks = await self.set_callbacks(
            properties=await self.get_tool_metadata()
        )

        full_response = ""
        new_history = []
        try:
            (
                generator,
                new_intent_history,
                new_assessment_history,
            ) = await asyncio.to_thread(
                message_tutor,
                self.problem,
                self.problem_set,
                self.llm,
                [HumanMessage(content=message, id=message_id)],
                chat_history,
                assessment_history,
                intent_history,
                tools=tutor_tools,
                variant=self.variant,
            )

            async for chunk in generator:
                # the generator yields message chuncks for a streaming resopnse
                # then finally yields the full response as the last chunk
                if (
                    chunk[0] == "messages"
                    and chunk[1]
                    and isinstance(chunk[1][0], AIMessage | AIMessageChunk)
                ):
                    full_response += chunk[1][0].content
                    yield chunk[1][0].content

                elif chunk[0] == "values":
                    new_history = filter_out_system_messages(chunk[1]["messages"])
            metadata = {
                "edx_module_id": self.edx_module_id,
                "tutor_model": self.model,
                "problem_set_title": self.problem_set_title,
                "run_readable_id": self.run_readable_id,
            }
            json_output = tutor_output_to_json(
                new_history,
                new_intent_history,
                new_assessment_history,
                metadata,
            )
            # Save both TutorBotOutput and DjangoCheckpoint objects atomically
            if self.checkpointer:
                await create_tutorbot_output_and_checkpoints(
                    self.thread_id, json_output, self.edx_module_id
                )
            yield (f"<!-- {await self.get_metadata(debug=debug)} -->")

        except Exception:
            yield '<!-- {"error":{"message":"An error occurred, please try again"}} -->'
            log.exception("Error running AI agent")

    async def load_problem_data(self) -> None:
        """Fetch problem content if it has not already been loaded."""
        if self.problem_data_loaded:
            return

        if self.variant == "canvas":
            if not self.run_readable_id or not self.problem_set_title:
                msg = "Canvas tutor requires run_readable_id and problem_set_title"
                raise ValueError(msg)
            self.problem_set = await get_canvas_problem_set(
                self.run_readable_id,
                self.problem_set_title,
            )
        else:
            if not self.edx_module_id or not self.block_siblings:
                msg = "Edx tutor requires edx_module_id and block_siblings"
                raise ValueError(msg)
            problem, problem_set = await get_problem_from_edx_block(
                self.edx_module_id,
                self.block_siblings,
            )
            self.problem = problem
            self.problem_set = problem_set

        self.problem_data_loaded = True


async def get_problem_from_edx_block(
    edx_module_id: str, block_siblings: list[str]
) -> tuple[str, str]:
    """
    Make an call to the learn contentfiles api to get the problem xml and problem
    set xml using the block id

    Args:
        edx_module_id: The edx_module_id of the problem
        block_siblings: The edx_module_id of block siblings of the problem, including
            the problem itself

    Returns:
        problem: The problem xml
        problem_set: The problem set xml
    """

    api_url = settings.AI_MIT_CONTENTFILE_URL
    params = {"edx_module_id": block_siblings}

    response = await async_request(
        api_url, params, timeout=10, include_learn_token=True
    )

    response = response.json()

    problem = get_matching_content(response, edx_module_id)

    problem_set = ""

    for sibling_module_id in block_siblings:
        problem_set += get_matching_content(response, sibling_module_id)
    return problem, problem_set


async def get_canvas_problem_set(
    run_readable_id: str, problem_set_title: str
) -> dict[str, list[dict[str, str]]]:
    """
    Make an call to the learn tutor probalem api to get the problem set and solution
    using run_readable_id and problem_set_title

    Args:
        run_readable_id: The readable id of the run
        problem_set_title: The title of the problem set

    Returns:
        problem_set: The problem set xml
    """

    api_url = f"{settings.PROBLEM_SET_URL}{run_readable_id}/{problem_set_title}/"

    response = await async_request(api_url, {}, timeout=10, include_learn_token=True)
    response = response.json()

    return {
        key: value
        for key, value in response.items()
        if key in ["problem_set_files", "solution_set_files"]
    }


def get_matching_content(api_results: json, edx_module_id: str):
    """
    Get the matching content from the api results

    Args:
        api_results: The api results
        edx_module_id: The edx_module_id of a specific contentfile

    Returns:
        The content of the contentfile
    """

    for result in api_results["results"]:
        if result["edx_module_id"] == edx_module_id:
            return result["content"]

    return ""


class VideoGPTAgentState(SummaryState):
    """
    State for the video GPT bot. Passes transcript_asset_id
    to the associated tool function.
    """

    transcript_asset_id: Annotated[list[str], add]


class VideoGPTBot(TruncatingChatbot):
    """Service class for the AI video chat agent"""

    PROMPT_TEMPLATE = "video_gpt"
    TASK_NAME = "VIDEO_GPT_TASK"
    JOB_ID = "VIDEO_GPT_JOB"
    STATE_CLASS = VideoGPTAgentState
    TRACE_STATE_KEYS = ("transcript_asset_id",)

    def get_trace_properties(
        self, extra_state: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Return the traced video identifiers, adding the course run that the
        transcript asset id embeds.
        """
        properties = super().get_trace_properties(extra_state)
        run_readable_id = get_run_readable_id_from_block_id(
            properties.get("transcript_asset_id")
        )
        if run_readable_id:
            properties["run_readable_id"] = run_readable_id
        return properties

    def __init__(  # noqa: PLR0913
        self,
        user_id: str,
        checkpointer: BaseCheckpointSaver,
        *,
        name: str = "MIT Open Learning VideoGPT Chatbot",
        model: str | None = None,
        temperature: float | None = None,
        instructions: str | None = None,
        thread_id: str | None = None,
    ):
        super().__init__(
            user_id,
            name=name,
            checkpointer=checkpointer,
            model=model or settings.AI_DEFAULT_VIDEO_GPT_MODEL,
            temperature=temperature,
            instructions=instructions,
            thread_id=thread_id,
        )
        self.agent = self.create_agent_graph()

    def create_tools(self):
        """Create tools required for the agent"""
        return [tools.get_video_transcript_chunk]

    async def get_tool_metadata(self) -> str:
        """Return the metadata for the search tool"""
        thread_id = self.config["configurable"]["thread_id"]
        latest_state = await self.get_latest_history()
        return get_search_tool_metadata(thread_id, latest_state)


class SupportAgentState(SummaryState):
    """
    State for the support bot: the page the learner was on, who to reach them at
    and under what name, and the subset of those addresses an authenticated
    session vouched for.
    """

    page_url: Annotated[list[str], add]
    user_email: Annotated[list[str], add]
    verified_email: Annotated[list[str], add]
    user_name: Annotated[list[str], add]
    user_username: Annotated[list[str], add]
    awaiting_send: Annotated[list[bool], add]
    filing_uncertain: Annotated[list[bool], add]
    filing_blocked: Annotated[list[bool], add]
    cancelled_after: Annotated[list[int], add]


SUPPORT_TICKET_REVIEW_HEADER = "Here's what I'll send to the support team:"
SUPPORT_TICKET_REVIEW_FOOTER = (
    'Reply "send" and I\'ll pass this on, or "cancel" to drop it. Anything else '
    "you type I'll add to the request and show you again."
)
SUPPORT_TICKET_FILED = (
    "Thanks - I've passed that on to the MIT Open Learning support team. Your "
    "reference number is {reference}, and they'll follow up at {email}."
)
SUPPORT_TICKET_ALREADY_FILED = (
    "I've already filed this one for you - your reference number is {reference}, "
    "and the support team will follow up at {email}."
)
SUPPORT_TICKET_FAILED = (
    'Sorry, I could not file your support request just now. Reply "send" to try '
    "again, or email support directly and they'll pick it up."
)
SUPPORT_TICKET_REJECTED = (
    "Sorry - the support system turned this request down, and sending it again "
    "won't change that. Please email the MIT Open Learning support team directly "
    "and they'll pick it up. I've flagged this so our team can look into it."
)
SUPPORT_TICKET_BAD_EMAIL = (
    "That address isn't one the support team can reply to. What's another email "
    "I can give them?"
)
SUPPORT_TICKET_MAYBE_FILED = (
    "I didn't get an answer back from the support system, so I can't tell you "
    "whether that went through. It may well have, so I won't send it a second "
    "time and risk a duplicate. If you haven't heard from the MIT Open Learning "
    "support team in a day or two, email them directly."
)
SUPPORT_TICKET_CANCELLED = (
    "No problem - I haven't sent anything, and I've put aside what you'd written "
    "so far. Tell me about it again whenever you're ready and we'll start fresh."
)
SUPPORT_SEND_UNCLEAR = (
    "I'm not sure whether that was a go-ahead or something to add, and I don't "
    'want to guess. Reply "send" on its own to pass on the request exactly as '
    'shown, "cancel" to drop it, or tell me what to add and I\'ll show you the '
    "updated version."
)
SUPPORT_TICKET_SENDING = (
    "I'm still sending that one over - give me a moment and I'll come back with "
    "the reference number."
)
SUPPORT_ASK_EMAIL = (
    "Thanks for letting us know. What's the best email address for the support "
    "team to reply to?"
)
SUPPORT_ASK_EMAIL_AGAIN = (
    "I still need an email address so the support team can reply. What's the best "
    "one to reach you at?"
)
SUPPORT_ASK_MORE = (
    "Thanks for letting us know. Is there anything else you'd like to add before I "
    "pass this on to the MIT Open Learning support team?"
)

EMAIL_IN_MESSAGE = re.compile(r"[^@\s,;<>()\[\]]+@[^@\s,;<>()\[\]]+\.[A-Za-z]{2,}")

SUPPORT_FILING_LOCK_PREFIX = "support_filing:"
# Covers the session read and write either side of the Zendesk call, and keeps
# the lock short enough that a worker killed mid-call frees the thread instead
# of stranding the learner on "still sending".
SUPPORT_FILING_LOCK_MARGIN = 30


def _filing_lock_ttl() -> int:
    """
    Hold the lock for longer than the call it guards can take. A shorter TTL
    lapses while the first turn is still in flight, which is the one moment a
    second turn must not get through.
    """
    return max_filing_seconds() + SUPPORT_FILING_LOCK_MARGIN


ZENDESK_SUBJECT_MAX_LENGTH = 150
# UserChatSession.support_ticket_email is an EmailField, so anything longer
# would file the ticket and then blow up on save.
EMAIL_MAX_LENGTH = 254
REPLY_TO_MAX_OTHER_WORDS = 6
# Everything a message can be made of and still be only an address being handed
# over. A word from outside this set means the message also says something, and
# what it says is support's to read.
REPLY_TO_ONLY_WORDS = frozenset(
    {
        "a",
        "actually",
        "address",
        "an",
        "and",
        "at",
        "best",
        "can",
        "contact",
        "email",
        "here",
        "here's",
        "i",
        "is",
        "it",
        "it's",
        "its",
        "me",
        "my",
        "no",
        "ok",
        "okay",
        "please",
        "reach",
        "sorry",
        "sure",
        "thanks",
        "the",
        "to",
        "use",
        "yes",
        "you",
    }
)


def _reply_to_email(message: str) -> str:
    """
    Return the address only when the message reads as an answer to "what's your
    email?". Addresses quoted inside prose belong to the problem, and filing
    against one makes Zendesk email a stranger.
    """
    matches = EMAIL_IN_MESSAGE.findall(message or "")
    if len(matches) != 1:
        return ""
    address = matches[0].rstrip(".")
    if len(address) > EMAIL_MAX_LENGTH:
        return ""
    if len(EMAIL_IN_MESSAGE.sub(" ", message).split()) > REPLY_TO_MAX_OTHER_WORDS:
        return ""
    return address


# A whole message from this set is the learner saying "go ahead".
AFFIRMATIVE_PHRASES = frozenset(
    {
        "send",
        "send it",
        "send it please",
        "send again",
        "yes",
        "yes please",
        "y",
        "yep",
        "yeah",
        "ok",
        "okay",
        "sure",
        "go",
        "go ahead",
        "do it",
        "confirm",
        "confirmed",
        "submit",
        "file it",
        "try again",
        "retry",
        "resend",
    }
)
# The word an affirmative can open with. Derived so the two cannot drift.
AFFIRMATIVE_OPENERS = frozenset(phrase.split()[0] for phrase in AFFIRMATIVE_PHRASES)

# A plain no. Under the review, where the question is whether to send, these
# mean do not send.
DECLINE_PHRASES = frozenset({"no", "nope", "no thanks", "no thank you"})

# Answers to "anything else to add?" that add nothing. Before the review they
# move things along; under it they are ambiguous - "that's all" may mean "go
# ahead" or "I have nothing more" - so they are asked about, not acted on.
NOTHING_MORE_PHRASES = frozenset(
    {
        "nothing",
        "nothing else",
        "no that's everything",
        "no that's all",
        "that's everything",
        "that's all",
        "that's it",
        "done",
    }
)

# None of these are the problem the learner wants help with, so none of them
# belong in what support reads.
FILLER_PHRASES = AFFIRMATIVE_PHRASES | DECLINE_PHRASES | NOTHING_MORE_PHRASES


# A word here on its own calls the send off.
CANCELLATION_TRIGGERS = frozenset(
    {
        "abort",
        "cancel",
        "canceled",
        "cancelled",
        "canceling",
        "cancelling",
        "discard",
        "forget",
        "nevermind",
        "scrap",
        "stop",
        "wait",
    }
)
# Everything a cancellation is allowed to be made of. A message containing one
# word from outside this set is an edit, because re-showing the review costs a
# turn while a wrong cancellation throws the learner's request away.
CANCELLATION_WORDS = CANCELLATION_TRIGGERS | {
    "actually",
    "dont",
    "don't",
    "do",
    "i",
    "it",
    "mind",
    "my",
    "never",
    "no",
    "not",
    "please",
    "request",
    "send",
    "sending",
    "sorry",
    "that",
    "the",
    "this",
    "ticket",
    "to",
    "want",
}
NEGATIONS = frozenset({"dont", "don't", "no", "not"})
# Past this a reply is carrying real detail, not answering the question we
# asked, so it is read as an edit rather than a yes or a no.
SHORT_REPLY_MAX_WORDS = 8


def _normalised(message: str) -> str:
    """
    Reduce a message to lower-case words, so phrases can be matched.

    Digits and non-ASCII letters are kept rather than stripped: dropping them
    leaves "yes, 18.01" as a bare "yes", which every caller here reads as
    permission to file the draft - discarding what the learner just typed.
    """
    return " ".join(re.sub(r"[^\w' ]", " ", (message or "").lower()).split())


def _is_cancellation(message: str) -> bool:
    """Report whether the learner is calling the send off."""
    words = _normalised(message).replace("never mind", "nevermind").split()
    if not words or len(words) > SHORT_REPLY_MAX_WORDS:
        return False
    if not set(words) <= CANCELLATION_WORDS:
        return False
    if set(words) & CANCELLATION_TRIGGERS:
        return True
    return "send" in words and bool(set(words) & NEGATIONS)


def _first(values: list[str]) -> str:
    """Return the first non-empty value, or an empty string."""
    return next((value for value in values if value), "")


def _latest(values: list[str]) -> str:
    """Return the last non-empty value, or an empty string."""
    return _first(list(reversed(values)))


def _is_bare_reply_to(message: str) -> bool:
    """
    Report whether the message is nothing but the address and the words used to
    hand it over. "I emailed prof@mit.edu already" is read as the reply-to, but
    it also tells support something, so it is not dropped from the ticket.
    """
    if not _reply_to_email(message):
        return False
    remainder = _normalised(EMAIL_IN_MESSAGE.sub(" ", message)).split()
    return set(remainder) <= REPLY_TO_ONLY_WORDS


def _is_filler(message: str) -> bool:
    """Report whether the message is an answer to us, with nothing for support."""
    return _normalised(message) in FILLER_PHRASES


def _is_affirmative(message: str) -> bool:
    """Report whether the learner said, and said only, to go ahead."""
    return _normalised(message) in AFFIRMATIVE_PHRASES


def _is_decline(message: str) -> bool:
    """Report whether the learner said no, plainly and nothing else."""
    return _normalised(message) in DECLINE_PHRASES


def _is_unclear_go_ahead(message: str) -> bool:
    """
    Report whether a short reply to the review reads as a yes carrying
    something else. "yes please send it to support asap" adds nothing to the
    ticket while "yes, and the course is 18.01" does, and nothing here can tell
    them apart - so neither is guessed at.
    """
    words = _normalised(message).split()
    if not words or len(words) > SHORT_REPLY_MAX_WORDS:
        return False
    return words[0] in AFFIRMATIVE_OPENERS or _normalised(message) in (
        NOTHING_MORE_PHRASES
    )


def _support_description(prior_messages: list, message: str, *, since: int = 0) -> str:
    """
    Join everything the learner typed about the problem. An address they gave is
    the reply-to, and a bare "send" is an answer to us, so neither belongs in
    what support reads.

    The opening message is kept whatever it looks like. Nobody has been asked
    for an address at that point, so "I cannot log in as x@y.com" is the
    problem even though it reads like the answer to that question - and
    dropping it filed an empty ticket.

    ``since`` is where the current request starts: a cancellation withdraws
    every word before it, so those are not this request's to file.
    """
    prior_messages = prior_messages[since:]
    typed = [
        msg.content
        for position, msg in enumerate(prior_messages)
        if isinstance(msg, HumanMessage)
        and msg.content
        # Dropped only where we came back and asked what it meant, so the
        # chatter we could not read stays out while the same words typed
        # anywhere else are still the learner's problem.
        and SUPPORT_SEND_UNCLEAR
        not in [
            getattr(reply, "content", "")
            for reply in prior_messages[position + 1 : position + 2]
        ]
    ]
    typed.append(message)
    return "\n\n".join(
        text
        for position, text in enumerate(typed)
        if text
        and (
            position == 0
            or not (
                _is_bare_reply_to(text) or _is_filler(text) or _is_cancellation(text)
            )
        )
    )


def _cancelled_after(prior_state: dict) -> int:
    """
    Where the current request starts in the thread. Latest wins: each
    cancellation moves the line forward past the one before it.
    """
    return (prior_state.get("cancelled_after") or [0])[-1]


def _accumulated(prior_state: dict, extra_state: dict, key: str) -> str:
    """
    Latest non-empty value for a key, across the thread and this request.

    The widget sends the page URL and the email with the opening message, not
    with the turn that files, so both have to come back out of accumulated
    state. Latest wins, because the earliest may be the one being corrected.
    """
    return _latest((prior_state.get(key) or []) + (extra_state.get(key) or []))


def _support_email(prior_state: dict, extra_state: dict, message: str) -> str:
    """
    Decide who the support team should reply to.

    A learner who mistyped their address can only fix it by typing it again.
    One an authenticated session vouched for is not theirs to change: that
    would let the chat name a stranger as the requester.
    """
    # On the opening message an address is part of the problem ("can't log in
    # as x@y.com"), not a reply-to, so it only counts once we have asked.
    parsed_reply_to = _reply_to_email(message) if prior_state.get("messages") else ""
    return (
        _accumulated(prior_state, extra_state, "verified_email")
        or parsed_reply_to
        or _accumulated(prior_state, extra_state, "user_email")
    )


def _support_review(description: str, page_url: str, email: str) -> str:
    """Show the learner the ticket before it exists, in their own words."""
    lines = [SUPPORT_TICKET_REVIEW_HEADER, "", f"**Your message:** {description}"]
    if page_url:
        lines += ["", f"**Where you were:** {page_url}"]
    lines += ["", f"**Reply to:** {email}", "", SUPPORT_TICKET_REVIEW_FOOTER]
    return "\n".join(lines)


def _ticket_subject(description: str) -> str:
    """Build a subject from the learner's own words, no model involved."""
    subject = " ".join((description or "").split())
    if len(subject) > ZENDESK_SUBJECT_MAX_LENGTH:
        subject = f"{subject[: ZENDESK_SUBJECT_MAX_LENGTH - 1].rstrip()}…"
    return subject or "Support request"


class SupportBot(TruncatingChatbot):
    """Service class for the support intake agent"""

    PROMPT_TEMPLATE = "support"
    TASK_NAME = "SUPPORT_TASK"
    JOB_ID = "SUPPORT_JOB"
    STATE_CLASS = SupportAgentState
    TRACE_STATE_KEYS = ("page_url",)
    # Every branch of get_completion ends at _reply_without_the_model.
    CALLS_THE_MODEL = False

    def __init__(  # noqa: PLR0913
        self,
        user_id: str,
        checkpointer: BaseCheckpointSaver,
        *,
        name: str = "MIT Open Learning Support Chatbot",
        model: str | None = None,
        temperature: float | None = None,
        instructions: str | None = None,
        thread_id: str | None = None,
    ):
        super().__init__(
            user_id,
            name=name,
            checkpointer=checkpointer,
            model=model or settings.AI_DEFAULT_SUPPORT_MODEL,
            temperature=temperature,
            instructions=instructions,
            thread_id=thread_id,
        )
        self.agent = self.create_agent_graph()

    def create_tools(self):
        """
        No tools: the bot takes the problem in, it does not answer it, and the
        ticket is filed deterministically in get_completion. A second,
        model-driven filing path could only disagree with that one.
        """
        return []

    async def get_tool_metadata(self) -> str:
        """Return no metadata - this agent has no tools"""
        return ""

    async def _prior_state(self) -> dict:
        """Return the graph state already accumulated on this thread."""
        state = await self.agent.aget_state(self.config)
        return (state.values if state else None) or {}

    async def _reply_without_the_model(
        self, message: str, reply: str, extra_state: dict
    ) -> AsyncGenerator[str, None]:
        """
        Yield a reply the model had no part in, and leave the same trace behind
        as a normal turn: the exchange written to the thread, and the metadata
        comment the caller reads the thread_id out of.
        """
        yield reply
        # as_node is required: with no checkpoint to infer from, or more than one
        # candidate, LangGraph refuses the write as ambiguous.
        await self.agent.aupdate_state(
            self.config,
            {"messages": [HumanMessage(message), AIMessage(reply)], **extra_state},
            as_node="agent",
        )
        metadata = await self.get_metadata()
        yield f"\n\n<!-- {metadata} -->\n\n"

    async def _send_to_zendesk(self, *, session, ticket: dict) -> tuple[str, dict]:
        """Call Zendesk and return the reply that reports how it went."""
        result = await file_support_ticket(
            subject=_ticket_subject(ticket["description"]),
            # Zendesk de-duplicates on this, which is the only thing that can
            # tell a retry after a lost response from a second request.
            thread_id=self.thread_id,
            **ticket,
        )
        if result.get("reference") is None:
            if result.get("uncertain"):
                # A ticket may exist that we never learned the id of, so the
                # thread stops here rather than risk filing a second one.
                return SUPPORT_TICKET_MAYBE_FILED, {
                    "awaiting_send": [False],
                    "filing_uncertain": [True],
                }
            if result.get("reason") == TICKET_BAD_EMAIL:
                # The request is fine, the address is not, so collection
                # resumes and the next message is read as the new reply-to.
                return SUPPORT_TICKET_BAD_EMAIL, {"awaiting_send": [False]}
            if result.get("reason") == TICKET_REJECTED:
                # Zendesk will answer the same way every time, so offering a
                # retry only walks the learner around the same loop.
                return SUPPORT_TICKET_REJECTED, {
                    "awaiting_send": [False],
                    "filing_blocked": [True],
                }
            # awaiting_send stays set, so the learner's next message retries
            # rather than starting the review over.
            return SUPPORT_TICKET_FAILED, {}
        if session:
            session.support_ticket_reference = str(result["reference"])
            session.support_ticket_email = ticket["email"]
            await session.asave(
                update_fields=["support_ticket_reference", "support_ticket_email"]
            )
        return SUPPORT_TICKET_FILED.format(
            reference=result["reference"], email=ticket["email"]
        ), {}

    async def _file_the_ticket(self, *, session, ticket: dict) -> tuple[str, dict]:
        """File one ticket per thread, however many turns ask at once."""
        cache = get_django_cache()
        lock_key = f"{SUPPORT_FILING_LOCK_PREFIX}{self.thread_id}"
        if not await cache.aadd(lock_key, value=True, timeout=_filing_lock_ttl()):
            return SUPPORT_TICKET_SENDING, {}
        try:
            if session:
                # The already-filed check ran before this turn did its work, so
                # an overlapping turn may have filed since.
                await session.arefresh_from_db(
                    fields=["support_ticket_reference", "support_ticket_email"]
                )
                if session.support_ticket_reference:
                    return SUPPORT_TICKET_ALREADY_FILED.format(
                        reference=session.support_ticket_reference,
                        email=session.support_ticket_email,
                    ), {}
            return await self._send_to_zendesk(session=session, ticket=ticket)
        finally:
            await cache.adelete(lock_key)

    def _collect_the_request(
        self, message: str, *, extra_state: dict, prior_state: dict
    ) -> tuple[str, dict]:
        """
        Ask for what is still missing, then show the request back.

        The intake conversation is scripted, not model-driven: it is a fixed
        form (collect the problem and a reply-to, show it back, file), so a
        model only adds a dependency and a way for every intake to fail.
        """
        prior_messages = prior_state.get("messages") or []
        email = _support_email(prior_state, extra_state, message)
        if not email:
            return (
                SUPPORT_ASK_EMAIL_AGAIN if prior_messages else SUPPORT_ASK_EMAIL
            ), {}
        if not prior_messages:
            # An authenticated learner's address arrives with the opening
            # message; invite more detail before showing the review.
            return SUPPORT_ASK_MORE, {"user_email": [email]}
        return _support_review(
            _support_description(
                prior_messages, message, since=_cancelled_after(prior_state)
            ),
            _accumulated(prior_state, extra_state, "page_url"),
            email,
            # An address parsed out of this turn reaches state no other way,
            # and the turn that files has to still know it.
        ), {"user_email": [email], "awaiting_send": [True]}

    async def _send_edit_or_cancel(
        self, message: str, *, extra_state: dict, prior_state: dict, session, confirmed
    ) -> tuple[str, dict]:
        """Act on what the learner said about the request they were shown."""
        prior_messages = prior_state.get("messages") or []
        # The question on screen here is whether to send, so a plain "no"
        # answers it. Cancelling costs a turn to undo; filing cannot be undone.
        if _is_cancellation(message) or _is_decline(message):
            return SUPPORT_TICKET_CANCELLED, {
                "awaiting_send": [False],
                # This turn's own exchange is written after the reply, so the
                # next request starts past both of those too.
                "cancelled_after": [len(prior_messages) + 2],
            }

        email = _support_email(prior_state, extra_state, message)
        page_url = _accumulated(prior_state, extra_state, "page_url")
        since = _cancelled_after(prior_state)

        # confirm is the widget's Send button; the phrases are for learners who
        # type instead. Anything else is an edit to the request, so it is shown
        # back rather than sent - the learner has not read this version yet.
        if not (confirmed or _is_affirmative(message)):
            if _is_unclear_go_ahead(message):
                return SUPPORT_SEND_UNCLEAR, {"awaiting_send": [True]}
            return _support_review(
                _support_description(prior_messages, message, since=since),
                page_url,
                email,
            ), {"user_email": [email], "awaiting_send": [True]}

        reply, filing_updates = await self._file_the_ticket(
            session=session,
            ticket={
                # What files is what the learner reviewed. This turn only
                # approved it, so it stays out of the ticket whether they typed
                # "send" or pressed a button that carried some words along.
                "description": _support_description(prior_messages, "", since=since),
                "page_url": page_url,
                "email": email,
                "name": _accumulated(prior_state, extra_state, "user_name"),
                "username": _accumulated(prior_state, extra_state, "user_username"),
                # Only this request's session can vouch for the address. Reading
                # the thread's accumulated verifications would let any later turn
                # inherit a sign-in it never presented.
                "email_verified": bool(email)
                and email in (extra_state.get("verified_email") or []),
            },
        )
        # A retry after a Zendesk failure reads the address back out of state,
        # so a correction made on this turn has to land there.
        return reply, {"user_email": [email], **filing_updates}

    async def get_completion(
        self,
        message: str,
        *,
        extra_state: dict[str, Any] | None = None,
        **kwargs,  # noqa: ARG002
    ) -> AsyncGenerator[str, None]:
        """
        Show the learner what will be sent, then file it on the turn after.
        Both steps happen in code: a model asked to call a tool does it only
        when it feels like it, and a learner who was told their request was
        sent has no way to tell that nothing was filed.
        """
        extra_state = dict(extra_state or {})
        # Popped, not stored: it is this turn's Send button, and a checkpointed
        # one would approve every turn after it.
        confirmed = any(extra_state.pop("confirm", None) or [])
        session = await UserChatSession.objects.filter(
            thread_id=self.thread_id
        ).afirst()
        prior_state = await self._prior_state()

        if session and session.support_ticket_reference:
            reply, updates = (
                SUPPORT_TICKET_ALREADY_FILED.format(
                    reference=session.support_ticket_reference,
                    email=session.support_ticket_email,
                ),
                {},
            )
        # Zendesk may hold a ticket from a call it never answered, so nothing
        # on this thread files again - there is no way to tell from here.
        elif (prior_state.get("filing_uncertain") or [False])[-1]:
            reply, updates = SUPPORT_TICKET_MAYBE_FILED, {}
        # Zendesk turned this thread's request down on its merits, so the same
        # words sent again get the same answer.
        elif (prior_state.get("filing_blocked") or [False])[-1]:
            reply, updates = SUPPORT_TICKET_REJECTED, {}
        # Latest wins, not any(): a cancellation appends False, and reading the
        # whole list would let the send the learner called off come back.
        elif (prior_state.get("awaiting_send") or [False])[-1]:
            reply, updates = await self._send_edit_or_cancel(
                message,
                extra_state=extra_state,
                prior_state=prior_state,
                session=session,
                confirmed=confirmed,
            )
        else:
            reply, updates = self._collect_the_request(
                message, extra_state=extra_state, prior_state=prior_state
            )

        async for chunk in self._reply_without_the_model(
            message, reply, {**extra_state, **updates}
        ):
            yield chunk
