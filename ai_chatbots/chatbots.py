"""Agent service classes for the AI chatbots"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from operator import add
from typing import Annotated, Any, Optional
from uuid import uuid4

import posthog
from django.conf import settings
from django.utils.module_loading import import_string
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.tools.base import BaseTool
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from langgraph.prebuilt.chat_agent_executor import AgentState
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
from posthog.ai.langchain import CallbackHandler

from ai_chatbots import tools
from ai_chatbots.api import (
    DjangoCheckpoint,
    MessageTruncationNode,
    create_tutorbot_output_and_checkpoints,
    get_search_tool_metadata,
    query_tutorbot_output,
)
from ai_chatbots.posthog import TokenTrackingCallbackHandler
from ai_chatbots.prompts import SYSTEM_PROMPT_MAPPING
from ai_chatbots.utils import (
    async_request_with_token,
    get_django_cache,
)

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

    def __init__(  # noqa: PLR0913
        self,
        user_id: str,
        checkpointer: BaseCheckpointSaver,
        *,
        name: str = "MIT Open Learning Chatbot",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        instructions: Optional[str] = None,
        thread_id: Optional[str] = None,
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
        llm = ChatLiteLLM(
            model=f"{self.proxy_prefix}{self.model}",
            streaming=True,
            **(self.proxy.get_api_kwargs() if self.proxy else {}),
            **(self.proxy.get_additional_kwargs(self) if self.proxy else {}),
            **kwargs,
        )
        # Set the temperature if it's supported by the model
        if self.temperature and self.model not in settings.AI_UNSUPPORTED_TEMP_MODELS:
            llm.temperature = self.temperature
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

    async def _get_latest_checkpoint_id(self) -> Optional[str]:
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

    async def set_callbacks(
        self, properties: Optional[dict] = None
    ) -> list[CallbackHandler]:
        """Set callbacks for the agent LLM"""
        if settings.POSTHOG_PROJECT_API_KEY and settings.POSTHOG_API_HOST:
            hog_client = posthog.Posthog(
                settings.POSTHOG_PROJECT_API_KEY, host=settings.POSTHOG_API_HOST
            )
            model_parts = self.model.rsplit("/", 1)
            extra_props = properties or {}
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
            return [callback_handler]
        return []

    async def get_completion(
        self,
        message: str,
        *,
        extra_state: Optional[dict[str, Any]] = None,
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
            self.config["callbacks"] = await self.set_callbacks()
            state = {
                "messages": [HumanMessage(message)],
                **(extra_state or {}),
            }
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
                    full_response += chunk[0].content
                    yield chunk[0].content
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
            yield f"<!-- {json.dumps(error)} -->"
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
            return json.dumps(metadata)
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
        checkpointer: Optional[BaseCheckpointSaver] = None,
        *,
        name: str = "MIT Open Learning Chatbot",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        instructions: Optional[str] = None,
        thread_id: Optional[str] = None,
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


class SyllabusAgentState(SummaryState):
    """
    State for the syllabus bot. Passes course_id and
    collection_name to the associated tool function.
    """

    course_id: Annotated[list[str], add]
    collection_name: Annotated[list[str], add]
    related_courses: Annotated[list[str], add]
    # str representation of a boolean value, because the
    # langgraph JsonPlusSerializer can't handle booleans
    exclude_canvas: Annotated[Optional[list[str]], add]


class SyllabusBot(TruncatingChatbot):
    """Service class for the AI syllabus agent"""

    PROMPT_TEMPLATE = "syllabus"
    TASK_NAME = "SYLLABUS_TASK"
    JOB_ID = "SYLLABUS_JOB"
    STATE_CLASS = SyllabusAgentState

    def __init__(  # noqa: PLR0913
        self,
        user_id: str,
        checkpointer: BaseCheckpointSaver,
        *,
        name: str = "MIT Open Learning Syllabus Chatbot",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        instructions: Optional[str] = None,
        thread_id: Optional[str] = None,
        enable_related_courses: Optional[bool] = False,
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
        self.agent = self.create_agent_graph()

    def create_tools(self):
        """Create tools required by the agent"""
        bot_tools = [tools.search_content_files]
        if self.enable_related_courses:
            bot_tools.append(tools.search_related_course_content_files)
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
        checkpointer: Optional[BaseCheckpointSaver] = BaseCheckpointSaver,
        *,
        name: str = "MIT Open Learning Tutor Chatbot",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        thread_id: Optional[str] = None,
        block_siblings: Optional[list[str]] = None,
        edx_module_id: Optional[str] = None,
        run_readable_id: Optional[str] = None,
        problem_set_title: Optional[str] = None,
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
        self.run_readable_id = run_readable_id
        self.problem_set_title = problem_set_title

        if not self.edx_module_id:
            self.variant = "canvas"
            self.problem = ""
        else:
            self.variant = "edx"
            self.problem = None

        self.problem_set = None
        self.problem_data_loaded = False

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
        extra_state: Optional[dict[str, Any]] = None,  # noqa: ARG002
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
            additional_metadata = {
                "edx_module_id": self.edx_module_id,
                "tutor_model": self.model,
                "problem_set_title": self.problem_set_title,
                "run_readable_id": self.run_readable_id,
            }
            json_output = tutor_output_to_json(
                new_history,
                new_intent_history,
                new_assessment_history,
                additional_metadata,
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

    response = await async_request_with_token(api_url, params, timeout=10)

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

    response = await async_request_with_token(api_url, {}, timeout=10)
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

    def __init__(  # noqa: PLR0913
        self,
        user_id: str,
        checkpointer: BaseCheckpointSaver,
        *,
        name: str = "MIT Open Learning VideoGPT Chatbot",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        instructions: Optional[str] = None,
        thread_id: Optional[str] = None,
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
