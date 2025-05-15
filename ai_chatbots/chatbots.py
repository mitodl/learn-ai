"""Agent service classes for the AI chatbots"""

import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from operator import add
from typing import Annotated, Optional
from uuid import uuid4

import posthog
import requests
from asgiref.sync import async_to_sync
from channels.db import database_sync_to_async
from django.conf import settings
from django.utils.module_loading import import_string
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.tools.base import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt.chat_agent_executor import AgentState
from open_learning_ai_tutor.message_tutor import message_tutor
from open_learning_ai_tutor.prompts import get_system_prompt
from open_learning_ai_tutor.tools import tutor_tools
from open_learning_ai_tutor.utils import (
    json_to_intent_list,
    json_to_messages,
    tutor_output_to_json,
)
from openai import BadRequestError
from typing_extensions import TypedDict

from ai_chatbots import tools
from ai_chatbots.api import get_search_tool_metadata
from ai_chatbots.models import TutorBotOutput
from ai_chatbots.prompts import PROMPT_MAPPING
from ai_chatbots.tools import get_video_transcript_chunk, search_content_files
from ai_chatbots.utils import get_django_cache

log = logging.getLogger(__name__)


class SummarizedState(AgentState):
    """Advanced state container with summary management capabilities.

    This implementation demonstrates three key features:
    1. Message storage with LangGraph annotations
    2. Dynamic summary tracking
    3. Configurable history window

    Attributes:
        messages: Annotated list of conversation messages
        summary: Running summary of conversation context
        window_size: Maximum messages to maintain before summarization

    Note:
        The summary field is crucial for maintaining context when
        older messages are pruned from the window.
    """

    messages: Annotated[list[str], add]
    summary: str


class BaseChatbot(ABC):
    """
    Base AI chatbot class
    """

    PROMPT_TEMPLATE = "base"

    # For LiteLLM tracking purposes
    TASK_NAME = "BASE_TASK"
    JOB_ID = "BASECHAT_JOB"
    STATE_CLASS = SummarizedState

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
            or get_system_prompt(self.PROMPT_TEMPLATE, PROMPT_MAPPING, get_django_cache)
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

    def create_agent_graph(self) -> CompiledGraph:  # noqa: C901
        """
        Return a graph for the relevant LLM and tools, summarizing the
        conversation history if needed.

        An easy way to create a basic graph is to use the prebuilt create_react_agent:

            from langgraph.prebuilt import create_react_agent

            return create_react_agent(
                self.llm,
                tools=self.tools,
                checkpointer=self.checkpointer,
                state_modifier=self.instructions,
            )

        """

        # Names of nodes in the graph
        summarizer_node = "summarizer"
        agent_node = "agent"
        tools_node = "tools"

        async def summarize(state: SummarizedState) -> str:
            """Summarize the conversation history with a summarizer agent"""
            log.info("SUMMARY STATE INPUT MSG COUNT: %d", len(state["messages"]))
            summary_bot = ChatSummarizerBot(
                self.user_id,
                checkpointer=self.checkpointer,
                model=settings.AI_DEFAULT_SUMMARY_MODEL,
                temperature=self.temperature,
                thread_id=self.thread_id,
            )
            summary_state = await summary_bot.agent.ainvoke(state)
            # Remove all but deleted messages
            log.info("SUMMARY STATE OUTPUT MSG COUNT: %d", len(summary_state["messages"]))
            log.info("SUMMARY STATE OUTPUT REMOVE COUNT: %d", len([s for s in summary_state["messages"] if isinstance(s, RemoveMessage)]))
            summary_state["messages"] = [
                m for m in summary_state["messages"] if isinstance(m, RemoveMessage)
            ]
            return summary_state

        def tool_calling_llm(state: SummarizedState) -> SummarizedState:
            """Call the LLM, injecting system prompt"""
            summary = state.get("summary")
            log.info("TOOL_CALLING_LLM summary? %s", summary)
            removed_message_ids = [
                m.id for m in state["messages"] if isinstance(m, RemoveMessage)
            ]
            valid_messages = [
                m
                for m in state["messages"]
                if not isinstance(m, RemoveMessage) and m.id not in removed_message_ids
            ]
            if not isinstance(valid_messages[0], SystemMessage):
                # inject the system prompt if not there
                valid_messages.insert(0, SystemMessage(self.instructions))
            for idx, m in enumerate(state["messages"]):
                log.info(
                    "STATE MSG %d: %s: %s: %s",
                    idx,
                    m.__class__.__name__,
                    m.id,
                    m.content[0:10] if hasattr(m, "content") else "",
                )
            if summary:
                # Add the summary to the messages
                if not isinstance(valid_messages[1], AIMessage):
                    valid_messages.insert(1, AIMessage(content=summary))
                else:
                    valid_messages[1].content = summary
            for idx, m in enumerate(valid_messages):
                log.info(
                    "VALID MSG %d: %s: %s: %s",
                    idx,
                    m.__class__.__name__,
                    m.id,
                    m.content[0:10] if hasattr(m, "content") else "",
                )
            return self.STATE_CLASS(messages=[self.llm.invoke(valid_messages)])

        def summarizing_llm(state: SummarizedState) -> SummarizedState:
            """Call the LLM to summarize the conversation if needed"""
            if len(state["messages"]) > settings.AI_MAX_CHAT_SESSION_LENGTH:
                log.info(
                    "NEED TO SUMMARIZE %d out of %d MESSAGES",
                    len(state),
                    len(state["messages"]),
                )
                new_state = async_to_sync(summarize)(state)
                log.info("NEW STATE # MESSAGES: %d", len(new_state["messages"]))
                log.info("SUMMARY? %s", new_state["summary"])
                return new_state
            else:
                log.info("NO NEED TO SUMMARIZE %d MESSAGES", len(state["messages"]))
                return {"summary": state.get("summary")}

        agent_graph = StateGraph(self.STATE_CLASS)
        # Summarize the conversation history
        agent_graph.add_node(summarizer_node, summarizing_llm)
        agent_graph.add_edge(summarizer_node, agent_node)
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
        agent_graph.set_entry_point(summarizer_node)

        # compile and return the agent graph
        return agent_graph.compile(checkpointer=self.checkpointer)

    async def get_latest_history(self) -> dict:
        """Get the most recent state history, summarize if needed"""
        latest_state = None
        async for state in self.agent.aget_state_history(self.config):
            if state:
                return state
        return None

    async def send_posthog_event(
        self, message: str, full_response: str, metadata: dict
    ) -> None:
        """
        Send a posthog event with the user message, AI response, and metadata
        """
        if settings.POSTHOG_PROJECT_API_KEY:
            try:
                hog_client = posthog.Posthog(
                    settings.POSTHOG_PROJECT_API_KEY, host=settings.POSTHOG_API_HOST
                )
                hog_client.capture(
                    self.user_id,
                    event=self.JOB_ID,
                    properties={
                        "question": message,
                        "answer": full_response,
                        "metadata": metadata,
                        "user": self.user_id,
                    },
                )
            except:  # noqa: E722
                log.exception("Error sending posthog event")

    async def get_completion(
        self,
        message: str,
        *,
        extra_state: Optional[TypedDict] = None,
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
                if isinstance(chunk[0], AIMessageChunk):
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
        metadata = await self.get_tool_metadata()
        if debug:
            yield f"\n\n<!-- {metadata} -->\n\n"
        await self.send_posthog_event(message, full_response, metadata)

    @abstractmethod
    async def get_tool_metadata(self) -> str:
        """
        Return metadata JSON about the response
        """
        raise NotImplementedError


class RecommendationAgentState(SummarizedState):
    """
    State for the recommendation bot. Passes search url
    to the associated tool function.
    """

    search_url: Annotated[list[str], add]


class ResourceRecommendationBot(BaseChatbot):
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
        return [tools.search_courses]

    async def get_tool_metadata(self) -> str:
        """Return the metadata for the search tool"""
        thread_id = self.config["configurable"]["thread_id"]
        latest_state = await self.get_latest_history()
        return get_search_tool_metadata(thread_id, latest_state)


class SyllabusAgentState(SummarizedState):
    """
    State for the syllabus bot. Passes course_id and
    collection_name to the associated tool function.
    """

    course_id: Annotated[list[str], add]
    collection_name: Annotated[list[str], add]


class SyllabusBot(BaseChatbot):
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
    ):
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
        return [search_content_files]

    async def get_tool_metadata(self) -> str:
        """Return the metadata for the search tool"""
        thread_id = self.config["configurable"]["thread_id"]
        latest_state = await self.get_latest_history()
        return get_search_tool_metadata(thread_id, latest_state)


@database_sync_to_async
def create_tutorbot_output(thread_id, chat_json):
    return TutorBotOutput.objects.create(thread_id=thread_id, chat_json=chat_json)


@database_sync_to_async
def get_history(thread_id):
    return TutorBotOutput.objects.filter(thread_id=thread_id).last()


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
        self.problem, self.problem_set = get_problem_from_edx_block(
            edx_module_id, block_siblings
        )

    async def get_tool_metadata(self) -> str:
        """Return the metadata for the  tool"""
        return json.dumps(
            {
                "edx_module_id": self.edx_module_id,
                "block_siblings": self.block_siblings,
                "problem": self.problem,
                "problem_set": self.problem_set,
            }
        )

    async def get_completion(
        self,
        message: str,
        *,
        extra_state: Optional[TypedDict] = None,  # noqa: ARG002
        debug: bool = settings.AI_DEBUG,  # noqa: ARG002
    ) -> AsyncGenerator[str, None]:
        """Call message_tutor with the user query and return the response"""

        history = await get_history(self.thread_id)

        if history:
            json_history = json.loads(history.chat_json)
            chat_history = json_to_messages(  # noqa: RUF005
                json_history.get("chat_history", [])
            ) + [HumanMessage(content=message)]

            intent_history = json_to_intent_list(json_history["intent_history"])
            assessment_history = json_to_messages(json_history["assessment_history"])

        else:
            chat_history = [HumanMessage(content=message)]
            intent_history = []
            assessment_history = []

        response = ""

        try:
            new_history, new_intent_history, new_assessment_history = message_tutor(
                self.problem,
                self.problem_set,
                self.llm,
                [HumanMessage(content=message)],
                chat_history,
                assessment_history,
                intent_history,
                tools=tutor_tools,
            )

            metadata = {"edx_module_id": self.edx_module_id, "tutor_model": self.model}
            json_output = tutor_output_to_json(
                new_history, new_intent_history, new_assessment_history, metadata
            )
            await create_tutorbot_output(self.thread_id, json_output)
            response = new_history[-1].content
            yield replace_math_tags(response)
            await self.send_posthog_event(
                message, response, await self.get_tool_metadata()
            )

        except Exception:
            yield '<!-- {"error":{"message":"An error occurred, please try again"}} -->'
            log.exception("Error running AI agent")


INLINE_MATH_REGEX = re.compile(r"\\\((.*?)\\\)")
DISPLAY_MATH_REGEX = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)


# react-markdown expects Mathjax deliminators to be $...$ or $$...$$
# the prompt for the tutorbot asks for Mathjax tags with $ format but
# the LLM does not get it right all the time
# this function replaces the Mathjax tags with the correct format
# eventually we will probably be able to remove this as LLMs get better
def replace_math_tags(input_string):
    r"""
    Replace instances of \(...\) and \[...\] Mathjax tags with $...$
    and $$...$$ tags.
    """
    input_string = re.sub(INLINE_MATH_REGEX, r"$\1$", input_string)
    return re.sub(DISPLAY_MATH_REGEX, r"$$\1$$", input_string)


def get_problem_from_edx_block(edx_module_id: str, block_siblings: list[str]):
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
    response = requests.get(api_url, params=params, timeout=10)

    response = response.json()

    problem = get_matching_content(response, edx_module_id)

    problem_set = ""

    for sibling_module_id in block_siblings:
        problem_set += get_matching_content(response, sibling_module_id)
    return problem, problem_set


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


class VideoGPTAgentState(SummarizedState):
    """
    State for the video GPT bot. Passes transcript_asset_id
    to the associated tool function.
    """

    transcript_asset_id: Annotated[list[str], add]


class VideoGPTBot(BaseChatbot):
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
        return [get_video_transcript_chunk]

    async def get_tool_metadata(self) -> str:
        """Return the metadata for the search tool"""
        thread_id = self.config["configurable"]["thread_id"]
        latest_state = await self.get_latest_history()
        return get_search_tool_metadata(thread_id, latest_state)


class ChatSummarizerBot(BaseChatbot):
    """
    Chatbot that summarizes a user's chat history
    """

    PROMPT_TEMPLATE = "summarizer"
    TASK_NAME = "CHAT_SUMMARIZER_TASK"
    JOB_ID = "CHAT_SUMMARIZER_JOB"
    STATE_CLASS = SummarizedState

    def __init__(  # noqa: PLR0913
        self,
        user_id: str,
        checkpointer: BaseCheckpointSaver,
        *,
        name: str = "MIT Open Learning Chat Summarizer",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        instructions: Optional[str] = None,
        thread_id: Optional[str] = None,
    ):
        super().__init__(
            user_id,
            name=name,
            checkpointer=checkpointer,
            model=model or settings.AI_DEFAULT_CHAT_SUMMARIZER_MODEL,
            temperature=temperature,
            instructions=instructions,
            thread_id=thread_id,
        )
        self.agent = self.create_agent_graph()

    def create_agent_graph(self) -> CompiledGraph:
        """
        Return a graph that generates a summary of the conversation.
        """

        # Names of nodes in the graph
        summarizer_node = "summarizer"

        def summarizing_llm(state: SummarizedState) -> SummarizedState:
            """Call the LLM to summarize the conversation"""
            log.info("State KEYS: %s", state.keys() if state else "No state")
            valid_messages = state["messages"]
            log.info("SUMMARIZE LLM MSG COUNT: %d", len(valid_messages))
            previous_summary = state.get("summary")
            if isinstance(previous_summary, dict):
                log.info("PREVIOUS SUMMARY IS A DICT: %s", previous_summary.keys())
                previous_summary = previous_summary.get("summary")
            log.info("PREVIOUS SUMMARY: %s", previous_summary)
            if previous_summary:
                # Add the previous summary to the messages
                valid_messages.insert(1, AIMessage(content=previous_summary))
            log.info("SUMMARY MESSAGE: %s", self.instructions)
            # Add prompt to our history
            valid_messages.append(HumanMessage(content=self.instructions))
            response = self.llm.invoke(valid_messages, stream=False)
            log.info("RESPONSE LLM SUMMARY: %s", response.content)

            # Delete all but the initial system prompt and the most recent message
            #log.info("Last 3 messages: %s", valid_messages[-3])
            log.info("Last 2 messages: %s", valid_messages[-2])
            #log.info("Last 1 messages: %s", valid_messages[-1])
            # Delete all but the system prompt and 2nd to last message
            last_user_message = [msg for msg in valid_messages if isinstance(msg, HumanMessage)][-2]
            delete_messages = [
                RemoveMessage(id=m.id)
                for m in state["messages"]
                if not isinstance(m, SystemMessage) and m != last_user_message
            ]
            log.info("last_human_message: %s", last_user_message)
            log.info("delete_messages count: %d", len(delete_messages))
            return {"summary": response.content, "messages": delete_messages}

        agent_graph = StateGraph(self.STATE_CLASS)
        # Summarize the conversation history
        agent_graph.add_node(summarizer_node, summarizing_llm)
        agent_graph.add_edge(summarizer_node, END)
        agent_graph.set_entry_point(summarizer_node)

        # compile and return the agent graph
        return agent_graph.compile(checkpointer=self.checkpointer)

    async def get_tool_metadata(self) -> str:
        """Return the metadata for the summarizer tool, in this case, nothing"""
        return ""

