"""AI-specific functions for ai_chatbots."""

import json
import logging

from langchain_core.messages import ToolMessage
from typing_extensions import TypedDict

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
            metadata = {
                "metadata": {
                    "search_url": content.get("metadata", {}).get("search_url"),
                    "search_parameters": content.get("metadata", {}).get(
                        "parameters", []
                    ),
                    "search_results": content.get("results", []),
                    "thread_id": thread_id,
                }
            }
            return json.dumps(metadata)
        except json.JSONDecodeError:
            log.exception(
                "Error parsing tool metadata, not valid JSON: %s", msg_content
            )
            return json.dumps(
                {"error": "Error parsing tool metadata", "content": msg_content}
            )
    else:
        return "{}"
