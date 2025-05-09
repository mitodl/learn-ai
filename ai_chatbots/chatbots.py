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
from channels.db import database_sync_to_async
from django.conf import settings
from django.utils.module_loading import import_string
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.tools.base import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from langgraph.prebuilt.chat_agent_executor import AgentState
from open_learning_ai_tutor.message_tutor import message_tutor
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
from ai_chatbots.tools import get_video_transcript_chunk, search_content_files

log = logging.getLogger(__name__)


class BaseChatbot(ABC):
    """
    Base AI chatbot class
    """

    INSTRUCTIONS = "Provide instructions for the LLM"

    # For LiteLLM tracking purposes
    TASK_NAME = "BASE_TASK"
    JOB_ID = "BASECHAT_JOB"

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
        self.instructions = instructions or self.INSTRUCTIONS
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

    def create_agent_graph(self) -> CompiledGraph:
        """
        Return a graph for the relevant LLM and tools.

        An easy way to create a graph is to use the prebuilt create_react_agent:

            from langgraph.prebuilt import create_react_agent

            return create_react_agent(
                self.llm,
                tools=self.tools,
                checkpointer=self.checkpointer,
                state_modifier=self.instructions,
            )

        The base implementation here accomplishes the same thing but a
        bit more explicitly, to give a better idea of what's happening
        and how it can be customized.
        """

        # Names of nodes in the graph
        agent_node = "agent"
        tools_node = "tools"

        def tool_calling_llm(state: MessagesState) -> MessagesState:
            """Call the LLM, injecting system prompt"""
            if len(state["messages"]) == 1:
                # New chat, so inject the system prompt
                state["messages"].insert(0, SystemMessage(self.instructions))
            return MessagesState(messages=[self.llm.invoke(state["messages"])])

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


class RecommendationAgentState(AgentState):
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

    TASK_NAME = "RECOMMENDATION_TASK"
    JOB_ID = "RECOMMENDATION_JOB"

    INSTRUCTIONS = """You are an assistant helping users find courses from a catalog
of learning resources. Users can ask about specific topics, levels, or recommendations
based on their interests or goals.  Do not answer questions that are not related to
educational resources at MIT.

Your job:
1. Understand the user's intent AND BACKGROUND based on their message.
2. Use the available function to gather information or recommend courses.
3. Provide a clear, user-friendly explanation of your recommendations if search results
are found.


Run the tool to find learning resources that the user is interested in,
and answer only based on the function search
results.

VERY IMPORTANT: NEVER USE ANY INFORMATION OUTSIDE OF THE MIT SEARCH RESULTS TO
ANSWER QUESTIONS.

If no results are returned, say you could not find any relevant
resources.  Don't say you're going to try again.  Ask the user if they would like to
modify their preferences or ask a different question.

Respond in this format:
- If the user's intent is unclear, ask clarifying questions about users preference on
price, certificate
- Understand user background from the message history, like their level of education.
- After the function executes, rerank results based on user background and return
only the top 1 or 2 of the results to the user.
- Make the title of each resource a clickable link.

VERY IMPORTANT: NEVER USE ANY INFORMATION OUTSIDE OF THE MIT SEARCH RESULTS TO ANSWER
QUESTIONS.

Here are some sample user prompts, each with a guide on how to respond to them:

Prompt: “I\'d like to learn some advanced mathematics that I may not have had exposure
to before, as a physics major.”
Expected Response: Ask some follow-ups about particular interests (e.g., set theory,
analysis, topology. Maybe ask whether you are more interested in applied math or theory.
Then perform the search based on those interests and send the most relevant results back
based on the user's answers.

Prompt: “As someone with a non-science background, what courses can I take that will
prepare me for the AI future.”
Expected Output: Maybe ask whether the user wants to learn how to program, or just use
AI in their discipline - does this person want to study machine learning? More info
needed. Then perform a relevant search and send back the best results.


AGAIN: NEVER USE ANY INFORMATION OUTSIDE OF THE MIT SEARCH RESULTS TO
ANSWER QUESTIONS.
    """

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

    def create_agent_graph(self) -> CompiledGraph:
        """
        Generate a standard react agent graph for the recommendation agent.
        Use the custom RecommendationAgentState to pass search_url
        to the associated tool function.
        """
        return create_react_agent(
            self.llm,
            tools=self.tools,
            checkpointer=self.checkpointer,
            state_schema=RecommendationAgentState,
            state_modifier=self.instructions,
        )


class SyllabusAgentState(AgentState):
    """
    State for the syllabus bot. Passes course_id and
    collection_name to the associated tool function.
    """

    course_id: Annotated[list[str], add]
    collection_name: Annotated[list[str], add]


class SyllabusBot(BaseChatbot):
    """Service class for the AI syllabus agent"""

    TASK_NAME = "SYLLABUS_TASK"
    JOB_ID = "SYLLABUS_JOB"

    INSTRUCTIONS = """You are an assistant helping users answer questions related
to a syllabus.

Your job:
1. Use the available function to gather relevant information about the user's question.
2. Provide a clear, user-friendly summary of the information retrieved by the tool to
answer the user's question.

Always use the tool results to answer questions, and answer only based on the tool
output. Do not include the course id in the query parameter.
VERY IMPORTANT: NEVER USE ANY INFORMATION OUTSIDE OF THE TOOL OUTPUT TO
ANSWER QUESTIONS.  If no results are returned, say you could not find any relevant
information.
    """

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

    def create_agent_graph(self) -> CompiledGraph:
        """
        Generate a standard react agent graph for the syllabus agent.
        Use the custom SyllabusAgentState to pass course_id and collection_name
        to the associated tool function.
        """
        return create_react_agent(
            self.llm,
            tools=self.tools,
            checkpointer=self.checkpointer,
            state_schema=SyllabusAgentState,
            state_modifier=self.instructions,
        )

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


class VideoGPTAgentState(AgentState):
    """
    State for the video GPT bot. Passes transcript_asset_id
    to the associated tool function.
    """

    transcript_asset_id: Annotated[list[str], add]


class VideoGPTBot(BaseChatbot):
    """Service class for the AI video chat agent"""

    TASK_NAME = "VIDEO_GPT_TASK"
    JOB_ID = "VIDEO_GPT_JOB"

    INSTRUCTIONS = """You are an assistant helping users answer questions related
to a video transcript.
Your job:
1. Use the available function to gather relevant information about the user's question.
2. Provide a clear, user-friendly summary of the information retrieved by the tool to
answer the user's question.
3. Do not specify the answer is from a transcript, instead say it's from video.
Always use the tool results to answer questions, and answer only based on the tool
output. Do not include the xblock video id in the query parameter.
VERY IMPORTANT: NEVER USE ANY INFORMATION OUTSIDE OF THE TOOL OUTPUT TO
ANSWER QUESTIONS.  If no results are returned, say you could not find any relevant
information.
"""

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

    def create_agent_graph(self) -> CompiledGraph:
        """
        Generate a standard react agent graph for the video gpt agent.
        Use the custom VideoGPTAgentState to pass transcript_asset_id
        to the associated tool function.
        """
        return create_react_agent(
            self.llm,
            tools=self.tools,
            checkpointer=self.checkpointer,
            state_schema=VideoGPTAgentState,
            state_modifier=self.instructions,
        )

    async def get_tool_metadata(self) -> str:
        """Return the metadata for the search tool"""
        thread_id = self.config["configurable"]["thread_id"]
        latest_state = await self.get_latest_history()
        return get_search_tool_metadata(thread_id, latest_state)
