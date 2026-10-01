"""AI-specific functions for ai_chatbots."""

import json
import logging
from typing import Any, Union
from uuid import uuid4

from channels.db import database_sync_to_async
from django.conf import settings
from django.db import transaction
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.utils.runnable import RunnableCallable
from langsmith import Client as LangsmithClient
from typing_extensions import TypedDict

from ai_chatbots.constants import WRITES_MAPPING
from ai_chatbots.models import DjangoCheckpoint, TutorBotOutput, UserChatSession
from ai_chatbots.utils import get_django_cache, get_sync_http_client
from main.utils import now_in_utc

log = logging.getLogger(__name__)


def get_search_tool_metadata(thread_id: str, latest_state: TypedDict) -> str:
    """
    Return the metadata for a bot search tool.
    """
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
        msg_content = tool_messages[-1].content
        try:
            content = json.loads(msg_content or "{}")
        except json.JSONDecodeError:
            log.exception(
                "Error parsing tool metadata, not valid JSON: %s", msg_content
            )
            return {"error": "Error parsing tool metadata", "content": msg_content}
        metadata = {
            "search_url": content.get("metadata", {}).get("search_url"),
            "search_parameters": content.get("metadata", {}).get("parameters", []),
            "search_results": content.get("results", []),
            "citation_sources": content.get("citation_sources", []),
            "thread_id": thread_id,
        }
        if content.get("error"):
            # a search that failed and one that matched nothing look the same
            # otherwise
            metadata["error"] = content["error"]
        return {"metadata": metadata}
    else:
        return {}


def get_langsmith_prompt(prompt_name: str) -> str:
    """Get the text of a prompt from LangSmith by its name."""
    if settings.LANGSMITH_API_KEY:
        client = LangsmithClient(api_key=settings.LANGSMITH_API_KEY)
        try:
            prompt_template = client.pull_prompt(prompt_name)
        except Exception:
            log.exception("Error retrieving prompt '%s' from LangSmith", prompt_name)
            return None
        if prompt_template:
            return prompt_template.messages[0].prompt.template
        else:
            log.warning("Prompt '%s' not found in LangSmith.", prompt_name)
    return None


def serialize_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """
    Transform LangChain tool call format to OpenAI function call format.
    """
    return [
        {
            "id": tc.get("id", ""),
            "type": tc.get("type", "tool_call"),
            "function": {
                "name": tc.get("name", ""),
                "arguments": tc.get("args", {}),
            },
        }
        for tc in tool_calls
    ]


@database_sync_to_async
def query_tutorbot_output(thread_id: str) -> TutorBotOutput | None:
    """Return the latest TutorBotOutput for a given thread_id"""
    return TutorBotOutput.objects.filter(thread_id=thread_id).last()


@database_sync_to_async
def create_tutorbot_output_and_checkpoints(
    thread_id: str, chat_json: Union[str, dict], edx_module_id: str | None
) -> tuple[TutorBotOutput, list[DjangoCheckpoint]]:
    """Atomically create both TutorBotOutput and DjangoCheckpoint objects"""
    with transaction.atomic():
        # Get the previous TutorBotOutput to compare messages
        previous_output = (
            TutorBotOutput.objects.filter(thread_id=thread_id).order_by("-id").first()
        )
        previous_chat_json = previous_output.chat_json if previous_output else None

        # Create TutorBotOutput
        tutorbot_output = TutorBotOutput.objects.create(
            session=UserChatSession.objects.get(thread_id=thread_id),
            thread_id=thread_id,
            chat_json=chat_json,
            edx_module_id=edx_module_id or "",
        )

        checkpoints = create_tutor_checkpoints(thread_id, chat_json, previous_chat_json)

        return tutorbot_output, checkpoints


def _should_create_checkpoint(msg: dict) -> bool:
    """Determine if a message should have a checkpoint created for it."""
    # Skip ToolMessage type or tool_calls
    return not (msg.get("type") == "ToolMessage" or msg.get("tool_calls"))


def _identify_new_messages(
    filtered_messages: list[dict], previous_chat_json: Union[str, dict] | None
) -> list[dict]:
    """Identify which messages are new by comparing with previous chat data."""
    if not previous_chat_json:
        return filtered_messages

    previous_chat_data = (
        json.loads(previous_chat_json)
        if isinstance(previous_chat_json, str)
        else previous_chat_json
    )
    previous_messages = previous_chat_data.get("chat_history", [])

    # Get set of existing message IDs from previous chat
    existing_message_ids = {
        msg.get("id")
        for msg in previous_messages
        if _should_create_checkpoint(msg) and msg.get("id")
    }

    # Find messages with IDs that don't exist in previous chat
    return [msg for msg in filtered_messages if msg["id"] not in existing_message_ids]


def _create_langchain_message(message: dict) -> dict:
    """Create a message in LangChain format."""
    return {
        "id": ["langchain", "schema", "messages", message["type"]],
        "lc": 1,
        "type": "constructor",
        "kwargs": {
            "id": message["id"],
            "type": message["type"].lower().replace("message", ""),
            "content": message["content"],
        },
    }


def _create_checkpoint_data(checkpoint_id: str, step: int, chat_data: dict) -> dict:
    """Create the checkpoint data structure."""
    return {
        "v": 4,
        "id": checkpoint_id,
        "ts": now_in_utc().isoformat(),
        "pending_sends": [],
        "versions_seen": {
            "__input__": {},
            "__start__": {"__start__": step + 1} if step >= 0 else {},
        },
        "channel_values": {
            "messages": [
                _create_langchain_message(msg)
                for msg in chat_data.get("chat_history", [])
            ],
            # Preserve tutor-specific data
            "intent_history": chat_data.get("intent_history"),
            "assessment_history": chat_data.get("assessment_history"),
            # Include metadata for reference
            "tutor_metadata": chat_data.get("metadata", {}),
            # Add other channel values that might be needed
            "branch:to:pre_model_hook": None,
        },
        "channel_versions": {"messages": len(chat_data.get("messages", []))},
    }


def _create_checkpoint_metadata(
    tutor_meta: dict, message: dict, step: int, thread_id: str
) -> dict:
    """Create metadata for the checkpoint based on message type."""
    writes = None
    message_type = message.get("kwargs", {}).get("type", None)
    source = "input" if message_type == HumanMessage.__name__ else "loop"
    container = WRITES_MAPPING.get(message_type, None)
    if container:
        writes = {container: {"messages": [message], **tutor_meta}}

    return {
        "step": step,
        "source": source,
        "writes": writes,
        "parents": {},
        "thread_id": thread_id,
    }


def create_tutor_checkpoints(
    thread_id: str,
    chat_json: Union[str, dict],
    previous_chat_json: Union[str, dict] | None = None,
) -> list[DjangoCheckpoint]:
    """Create DjangoCheckpoint records from tutor chat data (synchronous)"""
    # Get the associated session
    try:
        session = UserChatSession.objects.get(thread_id=thread_id)
    except UserChatSession.DoesNotExist:
        return []

    # Parse and validate chat data
    chat_data = json.loads(chat_json) if isinstance(chat_json, str) else chat_json
    messages = chat_data.get("chat_history", [])

    if not messages:
        return []

    # Filter out ToolMessage types and AI messages with tool_calls
    filtered_messages = [msg for msg in messages if _should_create_checkpoint(msg)]
    if not filtered_messages:
        return []

    # Get previous checkpoint if any
    latest_checkpoint = (
        DjangoCheckpoint.objects.filter(
            thread_id=thread_id,
            checkpoint__channel_values__tutor_metadata__isnull=False,
        )
        .only("checkpoint_id")
        .order_by("-id")
        .first()
    )
    parent_checkpoint_id = (
        latest_checkpoint.checkpoint_id if latest_checkpoint else None
    )

    # Determine new messages by comparing message IDs
    new_messages = _identify_new_messages(filtered_messages, previous_chat_json)
    if not new_messages:
        return []  # No new messages to checkpoint

    # Calculate starting step based on previous checkpoint if any
    step = latest_checkpoint.metadata.get("step", -1) + 1 if latest_checkpoint else 0
    checkpoints_created = []

    # Create checkpoints only for the NEW messages

    for message in new_messages:
        checkpoint_id = str(uuid4())

        # Create checkpoint data structure
        checkpoint_data = _create_checkpoint_data(checkpoint_id, step, chat_data)

        # Create message with LangChain format and add to cumulative history
        langchain_message = _create_langchain_message(message)

        # Create metadata for this step
        metadata = _create_checkpoint_metadata(
            chat_data.get("metadata", {}), langchain_message, step, thread_id
        )

        # Create and save the checkpoint
        checkpoint, _ = DjangoCheckpoint.objects.update_or_create(
            session=session,
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            defaults={
                "checkpoint_ns": "",
                "parent_checkpoint_id": parent_checkpoint_id,
                "type": "msgpack",
                "checkpoint": checkpoint_data,
                "metadata": metadata,
            },
        )
        parent_checkpoint_id = checkpoint_id
        checkpoints_created.append(checkpoint)
        step += 1

    return checkpoints_created


class MessageTruncationNode(RunnableCallable):
    """
    Truncates message history to keep only the last N human messages and
    their responses.

    Preserves all messages in the database, but only sends recent messages
    to the LLM to keep context window size small. Counts only human messages
    to determine truncation point, then includes all associated AI and tool
    messages.
    """

    def __init__(
        self,
        *,
        max_human_messages: int = settings.AI_HUMAN_MAX_CONVERSATION_MEMORY,
        input_messages_key: str = "messages",
        output_messages_key: str = "llm_input_messages",
        name: str = "truncation",
    ) -> None:
        """
        Initialize the MessageTruncationNode.

        Args:
            max_human_messages: Maximum number of human messages to keep.
            input_messages_key: Key to read messages from state.
            output_messages_key: Key to write truncated messages to state.
            name: Name of the node.
        """
        super().__init__(self._func, name=name, trace=False)
        self.max_human_messages = max_human_messages
        self.input_messages_key = input_messages_key
        self.output_messages_key = output_messages_key

    def _func(self, node_input: dict[str, Any]) -> dict[str, Any]:
        """Truncate messages to last N human messages and their responses."""
        messages = node_input.get(self.input_messages_key, [])

        if not messages:
            return {self.output_messages_key: messages}

        system_msg = None
        other_messages = messages

        if messages and isinstance(messages[0], SystemMessage):
            system_msg = messages[0]
            other_messages = messages[1:]

        human_message_count = sum(
            1 for msg in other_messages if isinstance(msg, HumanMessage)
        )
        if human_message_count <= self.max_human_messages:
            return {self.output_messages_key: messages}
        truncation_index = self.find_nth_human_message_from_end(
            other_messages, self.max_human_messages
        )

        truncated = other_messages[truncation_index:]
        result = [system_msg, *truncated] if system_msg else truncated

        return {self.output_messages_key: result}

    def find_nth_human_message_from_end(
        self, messages: list[BaseMessage], n: int
    ) -> int:
        """
        Find the index of the Nth human message from the end.

        Args:
            messages: List of messages to search.
            n: Number of human messages to count from the end.

        Returns:
            Index of the Nth human message from the end.
        """
        human_indices = [
            i for i, msg in enumerate(messages) if isinstance(msg, HumanMessage)
        ]
        if len(human_indices) < n:
            return 0
        return human_indices[-n]


RESOURCE_FACTS_CACHE_PREFIX = "resource_facts_"

RESOURCE_FACTS_HEADER = (
    "Facts about this resource from MIT Learn.  They are authoritative, so use "
    "them to answer questions about price, dates, format, duration and "
    "instructors instead of searching:"
)

RESOURCE_FACTS_MULTI_PLATFORM_HEADER = RESOURCE_FACTS_HEADER.rstrip(":") + (
    ".  The same course is offered on more than one platform and the learner "
    "could be asking about either, so give the facts for both unless they say "
    "which platform they are on:"
)


RESOURCE_FACTS_FOOTER = (
    "These facts describe the resource, not the learner's account, so trouble "
    "signing in, paying or enrolling is still a question for "
    "search_support_articles."
)


def _resource_run_instructors(resource: dict) -> str:
    """Return the instructors of the resource's best run, if any"""
    runs = resource.get("runs") or []
    best_run = next(
        (run for run in runs if run.get("id") == resource.get("best_run_id")),
        runs[0] if runs else {},
    )
    return ", ".join(
        instructor["full_name"]
        for instructor in (best_run.get("instructors") or [])
        if instructor.get("full_name")
    )


def _resource_fact_lines(resource: dict) -> list[str]:
    """Return the fact lines for a single learning resource"""
    prices = ", ".join(f"${price}" for price in (resource.get("prices") or []))
    delivery = ", ".join(
        item["name"]
        for key in ("delivery", "format", "pace")
        for item in (resource.get(key) or [])
        if item.get("name")
    )
    platform = (resource.get("platform") or {}).get("name")
    facts = {
        "Title": resource.get("title"),
        "Platform": platform,
        "Free": "yes" if resource.get("free") else "no",
        "Price": prices,
        "Certificate": (resource.get("certification_type") or {}).get("name")
        if resource.get("certification")
        else "none",
        "Format": delivery,
        "Duration": resource.get("duration") or resource.get("time_commitment"),
        "Next start date": (resource.get("next_start_date") or "")[:10],
        "Instructors": _resource_run_instructors(resource),
        "Url": resource.get("url") or resource.get("learn_url"),
    }
    return [f"- {label}: {value}" for label, value in facts.items() if value]


def _resource_facts_block(resources: list[dict]) -> str:
    """Render the facts of every resource matching a readable id"""
    blocks = [
        "\n".join(_resource_fact_lines(resource))
        for resource in resources
        if resource.get("title")
    ]
    if not blocks:
        return ""
    header = (
        RESOURCE_FACTS_MULTI_PLATFORM_HEADER
        if len(blocks) > 1
        else RESOURCE_FACTS_HEADER
    )
    return "\n\n".join([header, *blocks, RESOURCE_FACTS_FOOTER])


def get_resource_facts(readable_id: str, platform: str | None = None) -> str:
    """
    Return a block of authoritative facts about a learning resource, for
    inclusion in a chatbot system prompt.  Cached, and empty if the resource
    cannot be looked up: the content search still covers those questions.
    """
    if not readable_id:
        return ""
    cache = get_django_cache()
    cache_key = f"{RESOURCE_FACTS_CACHE_PREFIX}{readable_id}:{platform or ''}"
    cached_facts = cache.get(cache_key)
    if cached_facts is not None:
        return cached_facts
    params = {"readable_id": readable_id, "limit": 5}
    if platform:
        params["platform"] = platform
    try:
        # Resources are public, so the lookup works without a token; httpx
        # rejects an empty bearer header, so only send one when configured.
        headers = (
            {"Authorization": f"Bearer {settings.LEARN_ACCESS_TOKEN}"}
            if settings.LEARN_ACCESS_TOKEN
            else {}
        )
        response = get_sync_http_client().get(
            settings.AI_MIT_LEARNING_RESOURCES_URL,
            params=params,
            headers=headers,
            timeout=settings.AI_COURSE_PLATFORM_LOOKUP_TIMEOUT,
        )
        response.raise_for_status()
        facts = _resource_facts_block(response.json().get("results", []))
    except Exception:
        log.exception("Error looking up the facts of resource %s", readable_id)
        # Cache the failure briefly, so an unreachable API is not retried in
        # front of every message while it is down.
        cache.set(cache_key, "", settings.AI_COURSE_PLATFORM_ERROR_CACHE_DURATION)
        return ""
    cache.set(cache_key, facts, settings.AI_RESOURCE_FACTS_CACHE_DURATION)
    return facts
