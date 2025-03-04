"""Agent service classes for the AI chatbots"""

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from operator import add
from typing import Annotated, Optional
from uuid import uuid4
from asgiref.sync import sync_to_async
from channels.db import database_sync_to_async

import posthog
from django.conf import settings
from django.utils.module_loading import import_string
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.tools.base import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from langgraph.prebuilt.chat_agent_executor import AgentState
from openai import BadRequestError
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from open_learning_ai_tutor.problems import get_pb_sol
from open_learning_ai_tutor.StratL import message_tutor, process_StratL_json_output
from open_learning_ai_tutor.tools import tutor_tools
from open_learning_ai_tutor.utils import  messages_to_json, json_to_messages, intent_list_to_json
from ai_chatbots.models import TutorBotOutput
from ai_chatbots import tools
from ai_chatbots.api import get_search_tool_metadata
from ai_chatbots.tools import search_content_files

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
            # Format and yield an error message inside a hidden comment
            if hasattr(error, "response"):
                error = error.response.json()
            else:
                error = {
                    "error": {"message": "An error has occurred, please try again"}
                }
            if (
                error["error"]["message"].startswith("Budget has been exceeded")
                and not settings.AI_DEBUG
            ):  # Friendlier message for end user
                error["error"]["message"] = (
                    "You have exceeded your AI usage limit. Please try again later."
                )
            yield f"<!-- {json.dumps(error)} -->".encode()
        except Exception:
            yield '<!-- {"error":{"message":"An error occurred, please try again"}} -->'
            log.exception("Error running AI agent")
        if debug:
            yield f"\n\n<!-- {await self.get_tool_metadata()} -->\n\n"
        if settings.POSTHOG_PROJECT_API_KEY:
            hog_client = posthog.Posthog(
                settings.POSTHOG_PROJECT_API_KEY, host=settings.POSTHOG_API_HOST
            )
            hog_client.capture(
                self.user_id,
                event=self.JOB_ID,
                properties={
                    "question": message,
                    "answer": full_response,
                    "metadata": await self.get_tool_metadata(),
                    "user": self.user_id,
                },
            )

    @abstractmethod
    async def get_tool_metadata(self) -> str:
        """
        Yield markdown comments to send hidden metadata in the response
        """
        raise NotImplementedError


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

Always run the tool to answer questions, and answer only based on the tool
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
    def __init__(  # noqa: PLR0913
        self,
        user_id: str,
        checkpointer: Optional[BaseCheckpointSaver] = BaseCheckpointSaver,
        *,
        name: str = "MIT Open Learning Tutor Chatbot",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        thread_id: Optional[str] = None,
        problem_code: Optional[str] = None,
    ):
        super().__init__(
            user_id,
            name=name,
            checkpointer=checkpointer,
            temperature=temperature,
            thread_id=thread_id,
            model=model or settings.AI_DEFAULT_TUTOR_MODEL,
        )
        self.problem, self.solution = get_pb_sol(problem_code)
    
    def get_llm(self, **kwargs) -> BaseChatModel:
        """
        Return the LLM instance for the chatbot.
        Set it up to use a proxy, with required proxy kwargs, if applicable.
        """
        llm = ChatOpenAI(
            model=f"{self.proxy_prefix}{self.model}",
            **(self.proxy.get_api_kwargs(base_url_key="base_url", api_key_key="openai_api_key") if self.proxy else {}),
            **(self.proxy.get_additional_kwargs(self) if self.proxy else {}),
            **kwargs,
        )
        # Set the temperature if it's supported by the model
        if self.temperature and self.model not in settings.AI_UNSUPPORTED_TEMP_MODELS:
            llm.temperature = self.temperature
        return llm


    async def get_tool_metadata(self) -> str:
        """Return the metadata for the  tool"""
        return None
        
    async def get_completion(
        self,
        message: str,
        *,
        extra_state: Optional[TypedDict] = None,
        debug: bool = settings.AI_DEBUG,
    ) -> AsyncGenerator[str, None]:
        """Call message_tutor with the user query and return the response"""

        history = await get_history(self.thread_id)

        if history:
            json_history = json.loads(history.chat_json)
            self.chat_history = json_to_messages(json_history.get('chat_history', []))+[HumanMessage(content=message)]
            self.intent_history =  json_history.get('intent_history', [])
            self.assessment_history =  json_history.get('assessment_history', [])
        else:
            self.chat_history = [HumanMessage(content=message)]
            self.intent_history = '[]'
            self.assessment_history = ''

        response = ""

        try:
            json_output = message_tutor(
                self.problem, 
                self.solution, 
                self.llm, 
                messages_to_json([HumanMessage(content=message)]), 
                messages_to_json(self.chat_history), 
                self.assessment_history, 
                self.intent_history, 
                {"assessor_client": self.llm},
                tools=tutor_tools
            )

            await create_tutorbot_output(self.thread_id, json_output)
            prossessed = process_StratL_json_output(json_output)
            response = "An error has occurred, please try again"
            for index, msg in enumerate(prossessed[0]):
                if isinstance(msg, ToolMessage) and msg.name == 'text_student':
                    response = prossessed[0][index-1].tool_calls[0]['args']['message_to_student']

            yield response

        except Exception:
            yield '<!-- {"error":{"message":"An error occurred, please try again"}} -->'
            log.exception("Error running AI agent")

