"""Agent service classes for the AI chatbots"""

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Optional
from uuid import uuid4

import posthog
from django.conf import settings
from django.utils.module_loading import import_string
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.tools.base import BaseTool
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from openai import BadRequestError

from ai_chatbots import tools
from ai_chatbots.api import ChatMemory
from ai_chatbots.constants import LLMClassEnum

log = logging.getLogger(__name__)

DEFAULT_TEMPERATURE = 0.1
CHECKPOINTER = ChatMemory().checkpointer


class BaseChatbot(ABC):
    """
    Base AI chatbot class
    """

    INSTRUCTIONS = "Provide instructions for the LLM"

    # For LiteLLM tracking purposes
    JOB_ID = "BASECHAT_JOB"

    def __init__(  # noqa: PLR0913
        self,
        user_id: str,
        *,
        name: str = "MIT Open Learning Chatbot",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        instructions: Optional[str] = None,
        thread_id: Optional[str] = None,
    ):
        """Initialize the AI chat agent service"""
        self.bot_name = name
        self.model = model or settings.AI_MODEL
        self.temperature = temperature or DEFAULT_TEMPERATURE
        self.instructions = instructions or self.INSTRUCTIONS
        self.user_id = user_id
        self.config = {"configurable": {"thread_id": thread_id or uuid4().hex}}
        self.memory = CHECKPOINTER  # retain chat history
        if settings.AI_PROXY_CLASS:
            self.proxy = import_string(
                f"ai_chatbots.proxies.{settings.AI_PROXY_CLASS}"
            )()
            self.proxy.create_proxy_user(self.user_id)
        else:
            self.proxy = None
        self.llm = self.get_llm()
        self.agent = None

    def create_tools(self):
        """Create any tools required by the agent"""
        return []

    def get_llm(self, **kwargs) -> BaseChatModel:
        try:
            llm_class = LLMClassEnum[settings.AI_PROVIDER].value
        except KeyError:
            raise NotImplementedError from KeyError
        llm = llm_class(
            model=self.model,
            **(self.proxy.get_api_kwargs() if self.proxy else {}),
            **(self.proxy.get_additional_kwargs(self) if self.proxy else {}),
            **kwargs,
        )
        if self.temperature:
            llm.temperature = self.temperature
        return llm

    def create_agent_graph(self) -> CompiledGraph:
        """
        Return a graph for the relevant LLM and tools.

        The default implementation uses the prebuilt create_react_agent
        function.  Subclasses can override this method to create custom
        graphs.
        """

        return create_react_agent(
            self.llm,
            tools=self.create_tools(),
            checkpointer=self.memory,
            state_modifier=self.instructions,
        )

    @abstractmethod
    async def get_comment_metadata(self) -> str:
        """Yield markdown comments to send hidden metdata in the response"""

    async def get_completion(
        self, message: str, *, debug: bool = settings.AI_DEBUG
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
            response_generator = self.agent.astream(
                {"messages": [{"role": "user", "content": message}]},
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
            yield f"\n\n<!-- {await self.get_comment_metadata()} -->\n\n"
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
                    "metadata": await self.get_comment_metadata(),
                    "user": self.user_id,
                },
            )


class ResourceRecommendationBot(BaseChatbot):
    """
    Chatbot that searches for learning resources in the MIT Learn catalog,
    then recommends the best results to the user based on their query.
    """

    TASK_NAME = "RECOMMENDATION_TASK"

    INSTRUCTIONS = """You are an assistant helping users find courses from a catalog
of learning resources. Users can ask about specific topics, levels, or recommendations
based on their interests or goals.

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

Here are some guidelines on when to use the possible filters in the search function:

q: The area of interest requested by the user.  NEVER INCLUDE WORDS SUCH AS "advanced"
or "introductory" IN THIS PARAMETER! If the user asks for introductory, intermediate,
or advanced courses, do not include that in the search query, but examine the search
results to determine which most closely match the user's desired education level and/or
their educational background (if either is provided) and choose those results to return
to the user.  If the user asks what other courses are taught by a particular instructor,
search the catalog for courses taught by that instructor using the instructor's name
as the value for this parameter.

offered_by: If a user asks for resources "offered by" or "from" an institution,
you should include this parameter based on the following
dictionary:

    mitx = "MITx"
    ocw = "MIT OpenCourseWare"
    bootcamps = "Bootcamps"
    xpro = "MIT xPRO"
    mitpe = "MIT Professional Education"
    see = "MIT Sloan Executive Education"

DON'T USE THE offered_by FILTER OTHERWISE. Combine 2+ offered_by values in 1 query.

certification: true if the user is interested in resources that offer certificates,
false if the user does not want resources with a certificate offered.  Do not use
this filter if the user does not indicate a preference.

free: true if the user is interested in free resources, false if the user is only
interested in paid resources. Do not used this filter if the user does not indicate
a preference.

resource_type: If the user mentions courses, programs, videos, or podcasts in
particular, filter the search by this parameter.  DO NOT USE THE resource_type FILTER
OTHERWISE. You MUST combine multiple resource types in one request like this:
"resource_type=course&resource_type=program". Do not attempt more than one query per
user message. If the user asks for podcasts, filter by both "podcast" and
"podcast_episode".

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

Here are some recommended tool parameters to apply for sample user prompts:

User: "I am interested in learning advanced AI techniques for free"
Search parameters: q="AI techniques", free=true

User: "I am curious about AI applications for business"
Search parameters: q="AI business"

User: "I want free basic courses about biology from OpenCourseware"
Search parameters: q="biology", resource_type=["course"], offered_by: ["ocw"]

User: "I want to learn some advanced mathematics from MITx or OpenCourseware"
Search parameters: q="mathematics", , offered_by: ["ocw", "mitx]

AGAIN: NEVER USE ANY INFORMATION OUTSIDE OF THE MIT SEARCH RESULTS TO
ANSWER QUESTIONS.
    """

    def __init__(  # noqa: PLR0913
        self,
        user_id: str,
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
            model=model,
            temperature=temperature,
            instructions=instructions,
            thread_id=thread_id,
        )
        self.agent = self.create_agent_graph()

    def create_tools(self) -> list[BaseTool]:
        """Create tools required by the agent"""
        return [tools.search_courses]

    async def get_latest_history(self) -> dict:
        async for state in self.agent.aget_state_history(self.config):
            if state:
                return state
        return None

    async def get_comment_metadata(self) -> str:
        """
        Yield markdown comments to send hidden metadata in the response
        """
        thread_id = self.config["configurable"]["thread_id"]
        metadata = {"thread_id": thread_id}
        latest_state = await self.get_latest_history()
        tool_messages = (
            []
            if not latest_state
            else [
                t
                for t in latest_state.values.get("messages", [])
                if t and t.__class__ == ToolMessage
            ]
        )
        if tool_messages:
            content = json.loads(tool_messages[-1].content or {})
            metadata = {
                "metadata": {
                    "search_parameters": content.get("metadata", {}).get(
                        "parameters", []
                    ),
                    "search_results": content.get("results", []),
                    "thread_id": thread_id,
                }
            }
        return json.dumps(metadata)
