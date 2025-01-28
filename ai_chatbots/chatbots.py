"""Agent service classes for the AI chatbots"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

import pydantic
import requests
from django.conf import settings
from django.core.cache import caches
from django.utils.module_loading import import_string
from llama_index.core.agent import AgentRunner
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.constants import DEFAULT_TEMPERATURE
from llama_index.core.tools import FunctionTool, ToolMetadata
from openai import BadRequestError
from pydantic import Field

from ai_chatbots.api import get_agent, get_llm
from ai_chatbots.constants import LearningResourceType, OfferedBy
from ai_chatbots.utils import enum_zip

log = logging.getLogger(__name__)


class BaseChatbot(ABC):
    """
    Base AI chatbot class

    Llamaindex was chosen to implement this because it provides
    a far easier framework than native OpenAi or LiteLLM to
    handle function calling completions.  With LiteLLM/OpenAI,
    the first response may or may not be the result of a
    function call, so it's necessary to check the response.
    If it did call a function, then a second completion is needed
    to get the final response with the function call result added
    to the chat history.  Llamaindex handles this automatically.

    For comparison see:
    https://docs.litellm.ai/docs/completion/function_call
    """

    INSTRUCTIONS = "You are a friendly chatbot, answer the user's questions"

    # For LiteLLM tracking purposes
    TASK_NAME = "BASECHAT_TASK"
    CACHE_PREFIX = "base_ai_"

    def __init__(
        self,
        user_id: str,
        *,
        name: str = "MIT Open Learning Chatbot",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        instructions: Optional[str] = None,
    ):
        """Initialize the AI chatbot"""
        self.user_id = user_id
        self.assistant_name = name
        self.model = model or settings.AI_MODEL
        self.temperature = temperature or DEFAULT_TEMPERATURE
        self.instructions = instructions or self.INSTRUCTIONS
        if settings.AI_PROXY_CLASS and settings.AI_PROXY_URL:
            self.proxy = import_string(f"ai_chatbots.proxy.{settings.AI_PROXY_CLASS}")(
                user_id=user_id, task_id=self.TASK_NAME
            )
        else:
            self.proxy = None
        self.agent = None
        self.save_history = settings.AI_CACHE_HISTORY and self.user_id
        if self.save_history:
            self.cache = caches[settings.AI_CACHE]
            self.cache_timeout = settings.AI_CACHE_TIMEOUT
            self.cache_key = f"{self.CACHE_PREFIX}{self.user_id}"

    def get_or_create_chat_history_cache(self) -> None:
        """
        Get the user chat history from the cache and load it into the
        llamaindex agent's chat history (agent.chat_history).
        Create an empty cache key if it doesn't exist.
        """
        if self.cache_key in self.cache:
            try:
                for message in json.loads(self.cache.get(self.cache_key)):
                    self.agent.chat_history.append(ChatMessage(**message))
            except json.JSONDecodeError:
                self.cache.set(self.cache_key, "[]", timeout=self.cache_timeout)
        else:
            if self.proxy:
                self.proxy.create_proxy_user(self.user_id)
            self.cache.set(self.cache_key, "[]", timeout=self.cache_timeout)

    def save_chat_history(self) -> None:
        """Save the agent chat history to the cache"""
        chat_history = [
            message.dict()
            for message in self.agent.chat_history
            if message.role != "tool" and message.content
        ]
        self.cache.set(
            self.cache_key, json.dumps(chat_history), timeout=settings.AI_CACHE_TIMEOUT
        )

    @abstractmethod
    def create_agent(self) -> AgentRunner:
        """Create an AgentRunner for the relevant AI source"""

    def create_tools(self):
        """Create any tools required by the agent"""
        return []

    def clear_chat_history(self) -> None:
        """Clear the chat history from the cache"""
        self.agent.chat_history.clear()
        if self.save_history:
            self.cache.delete(self.cache_key)
            self.get_or_create_chat_history_cache()

    @abstractmethod
    def get_comment_metadata(self):
        """Yield markdown comments to send hidden metdata in the response"""

    def get_completion(self, message: str, *, debug: bool = settings.AI_DEBUG) -> str:
        """
        Send the user message to the agent and yield the response as
        it comes in.

        Append the response with debugging metadata and/or errors.
        """
        if not self.agent:
            error = "Create agent before running"
            raise ValueError(error)
        try:
            response = self.agent.stream_chat(
                message,
            )
            response_gen = response.response_gen
            yield from response_gen
        except BadRequestError as error:
            # Format and yield an error message inside a hidden comment
            if hasattr(error, "response"):
                error = error.response.json()
            else:
                error = {"error": {"message": str(error)}}
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
            yield f"\n\n<!-- {self.get_comment_metadata()} -->\n\n"
        if self.save_history:
            self.save_chat_history()


class FunctionCallingChatbot(BaseChatbot):
    """Function calling chatbot, using a FunctionCallingAgent"""

    TASK_NAME = "FUNCTION_CALL_TASK"

    def create_agent(self) -> AgentRunner:
        """
        Create a function calling agent
        """
        llm = get_llm(self.model, self.proxy)
        self.agent = get_agent().from_tools(
            tools=self.create_tools(),
            llm=llm,
            verbose=True,
            system_prompt=self.instructions,
        )
        if self.save_history:
            self.get_or_create_chat_history_cache()
        return self.agent


class ResourceRecommendationBot(FunctionCallingChatbot):
    """
    Chatbot that searches for learning resources in the MIT Learn catalog,
    then recommends the best results to the user based on their query.
    """

    TASK_NAME = "RECOMMENDATION_TASK"

    INSTRUCTIONS = f"""You are an assistant helping users find courses from a catalog
of learning resources. Users can ask about specific topics, levels, or recommendations
based on their interests or goals.

Your job:
1. Understand the user's intent AND BACKGROUND based on their message.
2. Use the available function to gather information or recommend courses.
3. Provide a clear, user-friendly explanation of your recommendations if search results
are found.


Always run the tool to answer questions, and answer only based on the function search
results. VERY IMPORTANT: NEVER USE ANY INFORMATION OUTSIDE OF THE MIT SEARCH RESULTS TO
ANSWER QUESTIONS.  If no results are returned, say you could not find any relevant
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
dictionary: {OfferedBy.as_dict()}  DO NOT USE THE offered_by FILTER OTHERWISE.

certificate: true if the user is interested in resources that offer certificates, false
if the user does not want resources with a certificate offered.  Do not used this filter
if the user does not indicate a preference.

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
- After the function executes, rerank results based on user background and recommend
1 or 2 courses to the user
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

And here are some recommended search parameters to apply for sample user prompts:

User: "I am interested in learning advanced AI techniques"
Search parameters: {{"q": "AI techniques"}}

User: "I am curious about AI applications for business"
Search parameters: {{"q": "AI business"}}

User: "I want free basic courses about climate change from OpenCourseware"
Search parameters: {{"q": "climate change", "free": true, "resource_type": ["course"],
"offered_by": "ocw"}}

User: "I want to learn some advanced mathematics"
Search parameters: {{"q": "mathematics"}}
    """

    class SearchToolSchema(pydantic.BaseModel):
        """Schema for searching MIT learning resources.

        Attributes:
            q: The search query string
            resource_type: Filter by type of resource (course, program, etc)
            free: Filter for free resources only
            certification: Filter for resources offering certificates
            offered_by: Filter by institution offering the resource
        """

        q: str = Field(
            description=(
                "Query to find resources. Never use level terms like 'advanced' here"
            )
        )
        resource_type: Optional[
            list[enum_zip("resource_type", LearningResourceType)]
        ] = Field(
            default=None,
            description="Type of resource to search for: course, program, video, etc",
        )
        free: Optional[bool] = Field(
            default=None,
            description="Whether the resource is free to access, true|false",
        )
        certification: Optional[bool] = Field(
            default=None,
            description=(
                "Whether the resource offers a certificate upon completion, true|false"
            ),
        )
        offered_by: Optional[enum_zip("offered_by", OfferedBy)] = Field(
            default=None,
            description="Institution that offers the resource: ocw, mitxonline, etc",
        )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "q": "machine learning",
                    "resource_type": ["course"],
                    "free": True,
                    "certification": False,
                    "offered_by": "MIT",
                }
            ]
        }
    }

    def __init__(
        self,
        user_id: str,
        *,
        name: Optional[str] = "Learning Resource Search AI Assistant",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        instructions: Optional[str] = None,
    ):
        """Initialize the chatbot"""
        super().__init__(
            user_id,
            name=name,
            model=model or settings.AI_MODEL,
            temperature=temperature,
            instructions=instructions,
        )
        self.search_parameters = []
        self.search_results = []
        super().create_agent()

    def search_courses(self, q: str, **kwargs) -> str:
        """
        Query the MIT API for learning resources, and
        return simplified results as a JSON string
        """

        params = {"q": q, "limit": settings.AI_MIT_SEARCH_LIMIT}

        valid_params = {
            "resource_type": kwargs.get("resource_type"),
            "free": kwargs.get("free"),
            "offered_by": kwargs.get("offered_by"),
            "certificate": kwargs.get("certificate"),
        }
        params.update({k: v for k, v in valid_params.items() if v is not None})
        self.search_parameters.append(params)
        try:
            response = requests.get(
                settings.AI_MIT_SEARCH_URL, params=params, timeout=30
            )
            response.raise_for_status()
            raw_results = response.json().get("results", [])
            # Simplify the response to only include the main properties
            main_properties = [
                "title",
                "url",
                "description",
                "offered_by",
                "free",
                "certification",
                "resource_type",
            ]
            simplified_results = []
            for result in raw_results:
                simplified_result = {k: result.get(k) for k in main_properties}
                # Instructors and level will be in the runs data if present
                next_date = result.get("next_start_date", None)
                raw_runs = result.get("runs", [])
                best_run = None
                if next_date:
                    runs = [run for run in raw_runs if run["start_date"] == next_date]
                    if runs:
                        best_run = runs[0]
                elif raw_runs:
                    best_run = raw_runs[-1]
                if best_run:
                    for attribute in ("level", "instructors"):
                        simplified_result[attribute] = best_run.get(attribute, [])
                simplified_results.append(simplified_result)
            self.search_results.extend(simplified_results)
            return json.dumps(simplified_results)
        except requests.exceptions.RequestException as e:
            log.exception("Error querying MIT API")
            return json.dumps({"error": str(e)})

    def create_tools(self):
        """Create tools required by the agent"""
        return [self.create_search_tool()]

    def create_search_tool(self) -> FunctionTool:
        """Create the search tool for the AI agent"""
        metadata = ToolMetadata(
            name="search_courses",
            description="Search for learning resources in the MIT catalog",
            fn_schema=self.SearchToolSchema,
        )
        return FunctionTool.from_defaults(
            fn=self.search_courses, tool_metadata=metadata
        )

    def get_comment_metadata(self) -> str:
        """
        Yield markdown comments to send hidden metadata in the response
        """
        metadata = {
            "metadata": {
                "search_parameters": self.search_parameters,
                "search_results": self.search_results,
                "system_prompt": self.instructions,
            }
        }
        return json.dumps(metadata)

class TutorChatbot():
    """
    Chatbot that assists with problem sets
    """

